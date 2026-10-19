"""Parallel, resumable conversion using the SAME pixel/action format as the pilot.

Run explicitly in Ubuntu/WSL. This module never starts training. Raw trajectories
and the optional reuse cache are read-only. One worker owns one output HDF5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing, contextmanager
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import time
import uuid

import h5py
import numpy as np

from wm_training.prepare_data import (
    ACTION_COLUMNS, ACTION_NAMES, FORMAT, IMAGE_MEAN, IMAGE_STD, action_statistics,
    contained_path, convert_episode, make_split, read_json, scene_signature,
    sha256, validate_arrays,
)


REVISION = 1
SPLITS = ("train", "val", "test")
PLAN_KEYS = ("format", "source_root", "source_manifest_sha256", "seed", "split_ids",
             "image_size", "resize", "pixel_mean", "pixel_std", "source_action_columns",
             "action_names", "frameskip", "control_dt", "scene")


def atomic_json(path, value, *, replace=False):
    """Atomic commit inside this run only; final index/plan are never replaced."""
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary.replace(path)


@contextmanager
def run_lock(output):
    """OS advisory lock releases after a crash; stale lock files need no deletion."""
    try:
        import fcntl
    except ImportError as error:
        raise RuntimeError("Run the converter in Ubuntu/WSL, not Windows Python") from error
    path = contained_path(output, ".prepare.lock")
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another converter is using this output directory. Do not run two copies.") from error
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def build_plan(source, *, seed, image_size, train_episodes, val_episodes, full):
    manifest_path = source / "manifest.json"
    digest = sha256(manifest_path)
    manifest = read_json(manifest_path)
    splits = make_split(manifest, seed)
    if full:
        selected = {name: list(splits[name]) for name in SPLITS}
    else:
        if train_episodes > len(splits["train"]) or val_episodes > len(splits["val"]):
            raise ValueError("Requested size exceeds its pool")
        selected = {"train": splits["train"][:train_episodes], "val": splits["val"][:val_episodes], "test": []}
    rows = {row["id"]: row for row in manifest["episodes"]}
    first = contained_path(source, rows[selected["train"][0]]["path"])
    scene = scene_signature(read_json(contained_path(first, "metadata.json")))
    plan = {"format": FORMAT, "source_root": str(source), "source_manifest_sha256": digest,
            "seed": seed, "split_ids": splits, "selected_ids": selected,
            "split_method": "mode-interleaved shuffled whole episodes; 90/5/5",
            "image_size": image_size, "resize": "LANCZOS letterbox; no crop; RGB ImageNet-mean padding",
            "pixel_mean": IMAGE_MEAN, "pixel_std": IMAGE_STD,
            "source_action_columns": list(ACTION_COLUMNS), "action_names": ACTION_NAMES,
            "frameskip": 1, "control_dt": scene["control_dt"], "scene": scene,
            "raw_label_semantics": manifest["class_names"], "labels_used_for_wm_training": False,
            "label_audit_scope": "stored flag consistency only; not proof of simulator-label correctness",
            "reserved_test_not_read": not bool(selected["test"]), "test_used_for_training": False,
            "parallel_converter_revision": REVISION,
            "conversion_source_sha256": {p.name: sha256(p) for p in
                                          (Path(__file__), Path(__file__).with_name("prepare_data.py"))}}
    return plan, rows


def reuse_mapping(root, plan):
    if root is None:
        return {}, None
    index = read_json(root / "index.json")
    if index.get("status") != "complete":
        raise ValueError("Reuse cache is not complete")
    for key in PLAN_KEYS:
        if index.get(key) != plan[key]:
            raise ValueError(f"Reuse cache has different {key}; cannot mix data/preprocessing")
    mapping, seen = {}, set()
    for split in SPLITS:
        for row in index["episodes"].get(split, []):
            episode_id = row["id"]
            if episode_id in seen or episode_id not in plan["split_ids"][split]:
                raise ValueError("Incorrect or duplicate split membership in reuse cache")
            seen.add(episode_id)
            mapping[episode_id] = str(contained_path(root, row["path"]))
    return mapping, index


def check_cached_episode(path, source, row, image_size, scene):
    """Check reused/resumed output against current raw NPZ/metadata and pixel digest.

    Does NOT reread the original PNGs. Assumes raw PNGs have remained immutable
    since initial conversion; the v1 pilot did not store per-PNG source hashes.
    """
    episode = contained_path(source, row["path"])
    meta_path, npz_path = contained_path(episode, "metadata.json"), contained_path(episode, "trajectory.npz")
    metadata = read_json(meta_path)
    meta_hash, npz_hash = sha256(meta_path), sha256(npz_path)
    if scene_signature(metadata) != scene:
        raise ValueError(f"Changed source scene for {row['id']}")
    with np.load(npz_path, allow_pickle=False) as raw, h5py.File(path, "r") as handle:
        action = validate_arrays(raw, metadata, row)
        n_states = len(action) + 1
        if (handle.attrs.get("format") != FORMAT or not handle.attrs.get("complete", False) or
                handle.attrs.get("episode_id") != row["id"] or
                handle.attrs.get("source_metadata_sha256") != meta_hash or
                handle.attrs.get("source_npz_sha256") != npz_hash or
                handle.attrs.get("alignment") != "pixels[t] -- action[t] --> pixels[t+1]" or
                json.loads(handle.attrs["metadata_json"]) != metadata or
                handle["pixels"].shape != (n_states, image_size, image_size, 3) or
                handle["pixels"].dtype != np.dtype("uint8") or
                handle["action"].dtype != np.dtype("float32") or
                not np.array_equal(handle["action"][:], action)):
            raise ValueError(f"Unusable cached episode {row['id']}: {path}")
        numeric_fields = set()
        for key in raw.files:
            values = raw[key]
            if values.dtype.kind in "biuf":
                numeric_fields.add(key)
                if (key not in handle["diagnostics"] or not np.isfinite(values).all() or
                        not np.array_equal(handle["diagnostics"][key][()], values)):
                    raise ValueError(f"Cached diagnostic mismatch {row['id']}/{key}")
        if set(handle["diagnostics"]) != numeric_fields:
            raise ValueError(f"Cached diagnostic fields changed for {row['id']}")
        digest = hashlib.sha256()
        for start in range(0, n_states, 16):
            digest.update(handle["pixels"][start:start + 16].tobytes())
        if digest.hexdigest() != handle.attrs.get("pixels_sha256"):
            raise ValueError(f"Cached pixels failed checksum for {row['id']}")
        counts = Counter(int(x) for x in raw["class_label"])
    if sha256(meta_path) != meta_hash or sha256(npz_path) != npz_hash:
        raise RuntimeError("Source changed during cache validation")
    info = {"id": row["id"], "mode": row["mode"], "path": f"episodes/episode_{row['id']:04d}.h5",
            "num_actions": len(action), "num_states": n_states,
            "class_counts": {str(i): counts[i] for i in range(3)},
            "source_npz_sha256": npz_hash, "source_metadata_sha256": meta_hash,
            "file_bytes": Path(path).stat().st_size}
    return info, action.astype(np.float64)


def ensure_space(output, states, image_size, workers=1):
    # Guard each new episode even after the initial size estimate. New cache
    # disk-full errors must leave old data intact and existing work resumable.
    required = int(states * image_size**2 * 3 * 1.2 * workers + 512 * 1024**2)
    if shutil.disk_usage(output).free < required:
        raise OSError(f"Insufficient free space; need {required/2**30:.2f} GiB for workers + reserve")


def work_episode(job):
    source, output = Path(job["source"]), Path(job["output"])
    row, size, scene = job["row"], job["image_size"], job["scene"]
    destination = contained_path(output, f"episodes/episode_{row['id']:04d}.h5")
    if destination.exists():
        info, actions = check_cached_episode(destination, source, row, size, scene)
        origin = "verified-existing"
    else:
        ensure_space(output, row["num_states"], size, job["workers"])
        temporary = contained_path(output, f"partial/episode_{row['id']:04d}-{uuid.uuid4().hex}.h5")
        candidate = job.get("reuse_path")
        if candidate is not None:
            candidate = Path(candidate)
            before = sha256(candidate)
            info, actions = check_cached_episode(candidate, source, row, size, scene)
            # Copy, never hardlink: changes to this cache cannot affect the pilot.
            with candidate.open("rb") as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            if sha256(temporary) != before or sha256(candidate) != before:
                raise RuntimeError("Reuse cache changed during copy")
            origin = "reused-pilot"
        else:
            info, actions = convert_episode(source, temporary, row, size, scene)
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            origin = "converted"
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing cache: {destination}")
        # A crash before this point leaves a uniquely named partial, not a final
        # file. --resume never opens partial files or overwrites them.
        temporary.rename(destination)
        info["path"] = f"episodes/{destination.name}"
        info["file_bytes"] = destination.stat().st_size
    stats = {"sum": actions.sum(0).tolist(), "sq_sum": (actions**2).sum(0).tolist(), "count": len(actions)}
    return {"info": info, "stats": stats, "origin": origin, "split": job["split"]}


def worker_init():
    # Parent handles Ctrl+C, stops scheduling and lets <= workers active jobs
    # finish their single episode. It does not wait for all remaining episodes.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def duration(seconds):
    if seconds is None:
        return "estimating"
    seconds = max(0, int(seconds))
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"


class Progress:
    def __init__(self, total, total_frames):
        self.total, self.total_frames = total, total_frames
        self.start = time.monotonic()
        self.done, self.frames, self.converted_frames = 0, 0, 0
        self.origins = Counter()

    def record(self, result):
        self.done += 1
        frames = result["info"]["num_states"]
        self.frames += frames
        self.origins[result["origin"]] += 1
        if result["origin"] == "converted":
            self.converted_frames += frames

    def snapshot(self):
        elapsed = time.monotonic() - self.start
        fps = self.converted_frames / max(elapsed, 1e-6)
        eta = 0.0 if self.done == self.total else ((self.total_frames - self.frames) / fps if fps > 0 else None)
        return {"done": self.done, "total": self.total, "frames_done": self.frames,
                "total_frames": self.total_frames, "elapsed_seconds": elapsed,
                "conversion_frames_per_second": fps, "eta_seconds": eta, "origins": dict(self.origins)}

    def show(self, detail):
        state = self.snapshot()
        print(f"[{self.done}/{self.total}] {detail} | frames={self.frames}/{self.total_frames} "
              f"| elapsed={duration(state['elapsed_seconds'])} "
              f"| new frames/s={state['conversion_frames_per_second']:.1f} "
              f"| ETA~{duration(state['eta_seconds'])}", flush=True)


def jobs_parallel(jobs, workers, heartbeat):
    if workers == 1:
        for job in jobs:
            yield work_episode(job)
        return
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=worker_init)
    pending, iterator = set(), iter(jobs)
    last_report = time.monotonic()
    try:
        for _ in range(workers):
            job = next(iterator, None)
            if job is not None:
                pending.add(pool.submit(work_episode, job))
        while pending:
            ready, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
            if not ready and time.monotonic() - last_report > 10:
                heartbeat()
                last_report = time.monotonic()
            for future in ready:
                result = future.result()
                yield result
                job = next(iterator, None)
                if job is not None:
                    pending.add(pool.submit(work_episode, job))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def estimate_budget(rows, selected, reuse_index, output, image_size):
    remaining = [rows[i] for ids in selected.values() for i in ids
                 if not contained_path(output, f"episodes/episode_{i:04d}.h5").exists()]
    frames = sum(row["num_states"] for row in remaining)
    if reuse_index is not None:
        ratios = [row["file_bytes"] / row["num_states"]
                  for rs in reuse_index["episodes"].values() for row in rs]
    else:
        ratios = []
    if ratios:
        budget = int(frames * max(ratios) * 2 + 2 * 1024**3)
        method = "2x largest pilot bytes/frame + 2 GiB; estimate, not a guarantee"
    else:
        budget = int(frames * image_size**2 * 3 * 1.2 + 512 * 1024**2)
        method = "uncompressed pixel budget + 20% + 512 MiB (no pilot size reference)"
    free = shutil.disk_usage(output).free
    print(f"Storage estimate {budget/2**30:.1f} GiB; filesystem free {free/2**30:.1f} GiB ({method})", flush=True)
    if free < budget:
        raise OSError("Not enough free space for the estimated remaining cache; no files were deleted")


def prepare_parallel(source, output, *, workers=4, resume=False, reuse_cache=None,
                     full=False, train_episodes=80, val_episodes=20, seed=20260921, image_size=224):
    if workers not in range(1, 9) or seed < 0 or train_episodes < 2 or val_episodes < 1:
        raise ValueError("Require workers 1..8, seed>=0, train>=2 and val>=1")
    if image_size < 28 or image_size % 14:
        raise ValueError("Image size must be a multiple of 14 and >=28")
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    reuse_cache = Path(reuse_cache).expanduser().resolve() if reuse_cache is not None else None
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Raw data and output must be separate directories")
    if reuse_cache is not None and (output.is_relative_to(reuse_cache) or reuse_cache.is_relative_to(output)):
        raise ValueError("Reuse cache and output must be separate directories")
    if output.exists() and not resume:
        raise FileExistsError(f"Output exists; use --resume for this converter's run: {output}")
    if resume and not output.is_dir():
        raise FileNotFoundError("--resume requires an existing output directory")
    plan, rows = build_plan(source, seed=seed, image_size=image_size,
                            train_episodes=train_episodes, val_episodes=val_episodes, full=full)
    reuse, reuse_index = reuse_mapping(reuse_cache, plan)
    if resume:
        previous = read_json(output / "plan.json")
        if previous != plan:
            raise ValueError("Resume plan/source/code has changed. Do not mix runs; request a migration or use a new output.")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        if not resume:
            atomic_json(output / "plan.json", plan)
        for name in ("episodes", "partial"):
            contained_path(output, name).mkdir(exist_ok=True)
        if (output / "index.json").exists():
            index = read_json(output / "index.json")
            if index.get("status") != "complete" or any(index.get(k) != v for k, v in plan.items()):
                raise ValueError("Existing index is not a matching complete run")
            print(f"Already complete: {output / 'index.json'} (no conversion started)", flush=True)
            return index
        leftovers = list((output / "partial").iterdir())
        if leftovers:
            print(f"Keeping {len(leftovers)} old partial file(s); they will NOT be reused or deleted.", flush=True)
        selected = plan["selected_ids"]
        estimate_budget(rows, selected, reuse_index, output, image_size)
        jobs = [{"source": str(source), "output": str(output), "row": rows[i], "split": split,
                 "image_size": image_size, "scene": plan["scene"], "workers": workers,
                 "reuse_path": reuse.get(i)} for split in SPLITS for i in selected[split]]
        progress = Progress(len(jobs), sum(job["row"]["num_states"] for job in jobs))
        results = {}
        print(f"Converting {len(jobs)} episodes with {workers} workers; "
              f"train/val/test={len(selected['train'])}/{len(selected['val'])}/{len(selected['test'])}. "
              "Ctrl+C stops scheduling; active episodes may need a moment to finish.", flush=True)
        try:
            with closing(jobs_parallel(jobs, workers, lambda: progress.show("workers busy"))) as completed:
                for result in completed:
                    info = result["info"]
                    results[info["id"]] = result
                    progress.record(result)
                    progress.show(f"{result['split']} episode_{info['id']:04d} {result['origin']}")
                    atomic_json(output / "progress.json", {"status": "converting", **progress.snapshot()}, replace=True)
            if sha256(source / "manifest.json") != plan["source_manifest_sha256"]:
                raise RuntimeError("Raw manifest changed during conversion")
            prepared = {split: [results[i]["info"] for i in selected[split]] for split in SPLITS}
            # Deterministic ID order, regardless of worker completion order. No
            # validation/test actions are fitted into these statistics.
            sums, sq_sums, count = np.zeros(4), np.zeros(4), 0
            for i in selected["train"]:
                stats = results[i]["stats"]
                sums += stats["sum"]
                sq_sums += stats["sq_sum"]
                count += stats["count"]
            index = {**plan, "status": "complete", "episodes": prepared,
                     "normalization": action_statistics(sums, sq_sums, count),
                     "total_bytes": sum(row["file_bytes"] for rs in prepared.values() for row in rs),
                     "execution": {"workers": workers, "resumed": resume, **progress.snapshot()}}
            atomic_json(output / "index.json", index)
            atomic_json(output / "progress.json", {"status": "complete", **progress.snapshot()}, replace=True)
            print(f"Cache ready: {output / 'index.json'}", flush=True)
            return index
        except BaseException as error:
            # Files completed by in-flight workers remain discoverable even if
            # this parent's progress count predates their completion.
            atomic_json(output / "INCOMPLETE.json", {"error": f"{type(error).__name__}: {error}",
                        "note": "Completed episode files are resumable; raw/reuse data is untouched.",
                        **progress.snapshot()}, replace=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-cache", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full", action="store_true", help="Convert all train/val/test episodes without training on test")
    parser.add_argument("--train-episodes", type=int, default=80, help="Smoke size when --full is absent")
    parser.add_argument("--val-episodes", type=int, default=20, help="Smoke size when --full is absent")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    if args.full and (args.train_episodes != 80 or args.val_episodes != 20):
        parser.error("Use --full alone, or specify pilot sizes without --full")
    prepare_parallel(args.source, args.output, workers=args.workers, resume=args.resume,
                     reuse_cache=args.reuse_cache, full=args.full, train_episodes=args.train_episodes,
                     val_episodes=args.val_episodes, seed=args.seed, image_size=args.image_size)


if __name__ == "__main__":
    main()
