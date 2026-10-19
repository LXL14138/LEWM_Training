"""Convert danger-benchmark datasets into an EVALUATION cache (same layout as the training cache).

Differences from prepare_full: no split is drawn (every episode goes to `val`,
with bench_dev / bench_test membership copied from benchmark_split.json), and
the action/pixel normalization is COPIED from --reference-cache (the cache the
WM was trained on) instead of being fitted. Episode ids become global:
dataset_index * 100000 + local id. Raw data and the reference cache are read-only.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import os
from pathlib import Path
import time

import numpy as np

from wm_training.prepare_data import (
    ACTION_COLUMNS, ACTION_NAMES, FORMAT, contained_path, read_json, scene_signature, sha256, write_new_json,
)
from wm_training.prepare_full import Progress, atomic_json, jobs_parallel, work_episode

EVAL_CACHE_REVISION = 2                    # 2: upstream provenance of counterfactual branch datasets in the index
GLOBAL_ID_STRIDE = 100000
SPLIT_FORMAT = "danger_bench_split_v1"
PROVENANCE_KEYS = ("benchmark_schedule", "counterfactual_branches")   # what fixed the attempts of a dataset
BRANCH_MEASURE_FORMAT = "counterfactual_branch_measure_v1"            # counterfactual_split.MEASURE_FORMAT (split file of branch datasets)
BRANCH_EPISODE_KEYS = ("part", "branch", "lead", "prefix_frames", "restore_frame", "source_dataset", "source_episode_id", "source_path")
BRANCH_PARTS = ("dev", "test")


def branch_provenance(manifest_record, split, dataset_index, dataset_path):
    """Upstream provenance of a counterfactual branch dataset, copied into the eval-cache index so that training guards
    can see where the branches were restored from without opening the raw dataset: the upstream (source) dataset
    roots and split file of the branch job, and per branch episode the SOURCE part (dev = validation half, test =
    training half of the training split), branch, lead and prefix. The split file must be the counterfactual_split
    output (its per-episode records were validated against every episode's metadata) and must agree with the job
    record's part."""
    if split.get("measure_format") != BRANCH_MEASURE_FORMAT or not isinstance(split.get("episodes"), list):
        raise ValueError(f"{dataset_path}: a counterfactual branch dataset needs a counterfactual_split split file "
                         f"(measure_format {BRANCH_MEASURE_FORMAT!r} with per-episode records)")
    roots = [str(d.get("path")) for d in (manifest_record.get("source_datasets") or []) if d.get("path")]
    job_split = manifest_record.get("split") or {}
    if not roots or not job_split.get("path"):
        raise ValueError(f"{dataset_path}: counterfactual_branches record names no source datasets / split file")
    job_part = job_split.get("part")
    per_episode = {}
    for entry in split["episodes"]:
        if int(entry["dataset"]) != dataset_index:
            continue
        missing = [k for k in BRANCH_EPISODE_KEYS if k not in entry]
        if missing:
            raise ValueError(f"{dataset_path}: split episode record lacks {missing}")
        if entry["part"] not in BRANCH_PARTS:
            raise ValueError(f"{dataset_path}: episode {entry['episode_id']} has an unknown source part {entry['part']!r}")
        if job_part in BRANCH_PARTS and entry["part"] != job_part:
            raise ValueError(f"{dataset_path}: episode {entry['episode_id']} comes from part {entry['part']!r} but the job "
                             f"was restricted to part {job_part!r}")
        per_episode[int(entry["episode_id"])] = {
            "source_part": str(entry["part"]), "branch": str(entry["branch"]), "lead": int(entry["lead"]),
            "prefix_frames": int(entry["prefix_frames"]), "restore_frame": int(entry["restore_frame"]),
            "upstream_source": {"dataset": int(entry["source_dataset"]), "episode_id": int(entry["source_episode_id"]),
                                "path": str(entry["source_path"])}}
    return {"upstream_source_roots": roots, "upstream_split_file": str(job_split["path"]), "upstream_split_sha256": job_split.get("sha256"),
            "job_part": job_part, "episodes": per_episode}


def provenance_record(manifest):
    """(key, record) of the manifest record that fixed the dataset's attempts: the benchmark schedule of a danger
    benchmark / targeted training collection, or the branch job record of a counterfactual dataset. Both carry a sha256."""
    for key in PROVENANCE_KEYS:
        record = manifest.get(key)
        if isinstance(record, dict) and record.get("sha256"):
            return key, record
    raise ValueError(f"manifest needs one of {PROVENANCE_KEYS} with a sha256")


def global_id(dataset_index, local_id):
    if not 0 <= local_id < GLOBAL_ID_STRIDE:
        raise ValueError("Local episode id too large for the global id scheme")
    return dataset_index * GLOBAL_ID_STRIDE + local_id


def bench_plan(split_path, reference, image_size):
    split = read_json(split_path)
    if split.get("format") != SPLIT_FORMAT:
        raise ValueError("Unrecognized benchmark split file")
    reference = Path(reference).resolve()
    ref_index = read_json(reference / "index.json")
    if ref_index.get("status") != "complete" or ref_index.get("format") != FORMAT or ref_index.get("eval_cache"):
        raise ValueError("--reference-cache must be a complete TRAINING cache")
    if ref_index["image_size"] != image_size:
        raise ValueError("Image size must equal the reference cache's")
    datasets, rows_by_gid, sources = [], {}, {}
    for entry in split["datasets"]:
        path = Path(entry["path"]).resolve()
        manifest_path = path / "manifest.json"
        if sha256(manifest_path) != entry["manifest_sha256"]:
            raise ValueError(f"{path}: manifest changed since the split was written")
        manifest = read_json(manifest_path)
        interrupted_ok = bool(split.get("allow_interrupted")) and manifest.get("status") == "interrupted"
        if (manifest.get("status") != "complete" and not interrupted_ok) or manifest.get("interaction_revision", 1) < 3:
            raise ValueError(f"{path}: not a complete benchmark dataset")
        provenance_key, provenance = provenance_record(manifest)
        index = entry["index"]
        upstream = branch_provenance(provenance, split, index, path) if provenance_key == "counterfactual_branches" else None
        datasets.append({"index": index, "path": str(path), "manifest_sha256": entry["manifest_sha256"],
                         "schedule_sha256": provenance["sha256"], "provenance": provenance_key, "status": manifest.get("status"),
                         "upstream": None if upstream is None else {k: v for k, v in upstream.items() if k != "episodes"}})
        for row in manifest["episodes"]:
            if row.get("status") != "complete":
                if interrupted_ok:
                    continue
                raise ValueError(f"{path}: incomplete episode {row['id']}")
            gid = global_id(index, int(row["id"]))
            rows_by_gid[gid] = {**row, "id": gid}
            sources[gid] = {"dataset": index, "local_id": int(row["id"]), "path": str(path), "provenance": provenance_key}
            if upstream is not None:
                if int(row["id"]) not in upstream["episodes"]:
                    raise ValueError(f"{path}: branch episode {row['id']} has no record in the split file")
                sources[gid].update(upstream["episodes"][int(row["id"])])
        first = contained_path(path, manifest["episodes"][0]["path"])
        if scene_signature(read_json(contained_path(first, "metadata.json"))) != ref_index["scene"]:
            raise ValueError(f"{path}: scene/camera/action settings differ from the reference cache")
    bench_split = {"bench_dev": [], "bench_test": []}
    for name, part in split["strata"].items():
        for key in ("dev", "test"):
            for ref in part[key]:
                gid = global_id(ref["dataset"], ref["episode_id"])
                if gid not in rows_by_gid:
                    raise ValueError(f"Split references an unknown episode {ref}")
                bench_split[f"bench_{key}"].append(gid)
                sources[gid]["stratum"] = name
    for key in bench_split:
        bench_split[key] = sorted(bench_split[key])
    if set(bench_split["bench_dev"]) & set(bench_split["bench_test"]):
        raise ValueError("Episode appears in both bench_dev and bench_test")
    gids = sorted(set(bench_split["bench_dev"]) | set(bench_split["bench_test"]))
    if set(gids) != set(rows_by_gid):
        raise ValueError("Split does not cover exactly the complete episodes of its datasets")
    # Branch episodes inherit their SOURCE event's part: dev-part branches must sit in bench_dev, test-part in bench_test.
    for key, part in (("bench_dev", "dev"), ("bench_test", "test")):
        wrong = [g for g in bench_split[key] if sources[g].get("source_part") not in (None, part)]
        if wrong:
            raise ValueError(f"{len(wrong)} branch episode(s) in {key} come from the {sources[wrong[0]]['source_part']!r} part of the "
                             f"training split (first: {wrong[:5]}); the split file is inconsistent")
    upstream_roots = sorted({r for d in datasets if d["upstream"] for r in d["upstream"]["upstream_source_roots"]})
    upstream_splits = sorted({d["upstream"]["upstream_split_file"] for d in datasets if d["upstream"]})
    ref_sha = sha256(reference / "index.json")
    plan = {"format": FORMAT, "eval_cache": True, "eval_cache_revision": EVAL_CACHE_REVISION,
            "source_root": datasets[0]["path"], "source_roots": [d["path"] for d in datasets],
            "source_manifest_sha256": {d["path"]: d["manifest_sha256"] for d in datasets},
            "source_provenance": {d["path"]: d["provenance"] for d in datasets},
            "upstream_source_roots": upstream_roots, "upstream_split_files": upstream_splits,
            "upstream_by_dataset": {d["path"]: d["upstream"] for d in datasets},
            "source_parts": {part: sum(1 for s in sources.values() if s.get("source_part") == part) for part in BRANCH_PARTS},
            "split_file": str(Path(split_path).resolve()), "split_sha256": sha256(split_path),
            "reference_cache": {"path": str(reference), "index_sha256": ref_sha},
            "seed": split["seed"], "split_ids": {"train": [], "val": gids, "test": []},
            "selected_ids": {"train": [], "val": gids, "test": []},
            "split_method": "danger benchmark: every episode in val; bench_dev/bench_test copied from benchmark_split.json",
            "bench_split": bench_split, "episode_sources": {str(g): sources[g] for g in gids},
            "image_size": image_size, "resize": ref_index["resize"],
            "pixel_mean": ref_index["pixel_mean"], "pixel_std": ref_index["pixel_std"],
            "source_action_columns": list(ACTION_COLUMNS), "action_names": ACTION_NAMES,
            "frameskip": 1, "control_dt": ref_index["control_dt"], "scene": ref_index["scene"],
            "raw_label_semantics": ref_index["raw_label_semantics"], "labels_used_for_wm_training": False,
            "label_audit_scope": "stored flag consistency only; not proof of simulator-label correctness",
            "reserved_test_not_read": True, "test_used_for_training": False,
            "conversion_source_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("prepare_data.py"),
                                                                   Path(__file__).with_name("prepare_full.py"))}}
    for key in ("image_size", "resize", "pixel_mean", "pixel_std", "source_action_columns", "frameskip", "control_dt"):
        if plan[key] != ref_index[key]:
            raise ValueError(f"Preprocessing contract differs from the reference cache: {key}")
    normalization = {**ref_index["normalization"], "copied_from_reference": True, "reference_index_sha256": ref_sha}
    return plan, rows_by_gid, normalization, datasets


def prepare_bench(split_path, reference, output, *, workers=4, image_size=224):
    if workers not in range(1, 9):
        raise ValueError("workers must be 1..8")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    plan, rows, normalization, datasets = bench_plan(split_path, reference, image_size)
    for root in [Path(d["path"]) for d in datasets] + [Path(reference).resolve()]:
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output must be separate from the datasets and the reference cache")
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "plan.json", plan)
    for name in ("episodes", "partial"):
        (output / name).mkdir()
    jobs = [{"source": plan["episode_sources"][str(gid)]["path"], "output": str(output), "row": rows[gid], "split": "val",
             "image_size": image_size, "scene": plan["scene"], "workers": workers, "reuse_path": None}
            for gid in plan["split_ids"]["val"]]
    progress = Progress(len(jobs), sum(job["row"]["num_states"] for job in jobs))
    results = {}
    print(f"Converting {len(jobs)} benchmark episodes with {workers} workers into an evaluation cache "
          f"(normalization copied from {plan['reference_cache']['path']}).", flush=True)
    try:
        with closing(jobs_parallel(jobs, workers, lambda: progress.show("workers busy"))) as completed:
            for result in completed:
                info = result["info"]
                results[info["id"]] = result
                progress.record(result)
                progress.show(f"episode_{info['id']} {result['origin']}")
                atomic_json(output / "progress.json", {"status": "converting", **progress.snapshot()}, replace=True)
        for d in datasets:
            if sha256(Path(d["path"]) / "manifest.json") != d["manifest_sha256"]:
                raise RuntimeError("Raw manifest changed during conversion")
        prepared = [results[gid]["info"] for gid in plan["split_ids"]["val"]]
        for info in prepared:
            source = plan["episode_sources"][str(info["id"])]
            info.update(source_dataset=source["dataset"], source_id=source["local_id"], stratum=source["stratum"],
                        source_part=source.get("source_part"), branch=source.get("branch"), lead=source.get("lead"),
                        prefix_frames=source.get("prefix_frames"))
        index = {**plan, "status": "complete", "episodes": {"train": [], "val": prepared, "test": []},
                 "normalization": normalization,
                 "total_bytes": sum(row["file_bytes"] for row in prepared),
                 "execution": {"workers": workers, "resumed": False, **progress.snapshot()}}
        atomic_json(output / "index.json", index)
        atomic_json(output / "progress.json", {"status": "complete", **progress.snapshot()}, replace=True)
        print(f"Evaluation cache ready: {output / 'index.json'}", flush=True)
        return index
    except BaseException as error:
        atomic_json(output / "INCOMPLETE.json", {"error": f"{type(error).__name__}: {error}", **progress.snapshot()}, replace=True)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True, help="benchmark_split.json")
    p.add_argument("--reference-cache", type=Path, required=True, help="training cache whose normalization is copied")
    p.add_argument("--output", type=Path, required=True, help="NEW directory")
    p.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    p.add_argument("--image-size", type=int, default=224)
    a = p.parse_args(argv)
    prepare_bench(a.split, a.reference_cache, a.output, workers=a.workers, image_size=a.image_size)


if __name__ == "__main__":
    main()
