"""Prepare a NEW, episode-separated LeWM cache; never modify raw trajectories.

This is our adapter's HDF5 layout, not stable_worldmodel's built-in HDF5 format.
Run this file explicitly to create files. Importing it performs no conversion.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from PIL import Image


FORMAT = "three_blocks_lewm_v1"
ACTION_COLUMNS = (0, 1, 2, 4)
ACTION_NAMES = ["dx", "dy", "dz", "dgripper"]
MODES = {"free", "grasp", "carry", "guard"}
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root, relative):
    """Reject path escapes, including symlinks resolving outside the input root."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Path escapes the dataset directory: {relative}")
    return path


def make_split(manifest, seed=20260921):
    """Fixed 90/5/5 episode split, interleaved across collection modes.

    No selection by success, safety, length or future labels. The held-out test
    IDs are recorded but their images/NPZ are not read by the pilot converter.
    """
    if (manifest.get("status") != "complete" or
            manifest.get("schema_version") != 3 or
            manifest.get("dataset_kind") != "guided_exploration_v1"):
        raise ValueError("Expected a complete schema-3 guided-exploration dataset")
    rows = manifest["episodes"]
    if len(rows) != manifest["requested_episodes"]:
        raise ValueError("Manifest episode count is incomplete")
    ids = [row["id"] for row in rows]
    paths = [row["path"] for row in rows]
    if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
        raise ValueError("Duplicate episode IDs or paths")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        if (type(row["id"]) is not int or row["id"] < 0 or
                row["path"] != f"episode_{row['id']:04d}" or
                row["status"] != "complete" or row["mode"] not in MODES or
                row["num_actions"] < 3 or row["num_states"] != row["num_actions"] + 1):
            raise ValueError(f"Invalid/incomplete episode row: {row['id']}")
        groups[row["mode"]].append(row["id"])
    rng = np.random.default_rng(seed)
    for mode in sorted(groups):
        rng.shuffle(groups[mode])
    ordered = [int(i) for group in itertools.zip_longest(
        *(groups[mode] for mode in sorted(groups))
    ) for i in group if i is not None]
    n_val = max(1, round(len(ordered) * 0.05))
    n_test = max(1, round(len(ordered) * 0.05))
    if len(ordered) - n_val - n_test < 2:
        raise ValueError("Too few episodes for independent train/val/test pools")
    return {"train": ordered[n_val + n_test:],
            "val": ordered[:n_val], "test": ordered[n_val:n_val + n_test]}


def letterbox(image, size):
    """Keep the whole camera image, preserve aspect ratio, never crop guards."""
    width, height = image.size
    scale = size / max(width, height)
    shape = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.convert("RGB").resize(shape, Image.Resampling.LANCZOS)
    fill = tuple(round(value * 255) for value in IMAGE_MEAN)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(image, ((size - shape[0]) // 2, (size - shape[1]) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def validate_arrays(arrays, metadata, row):
    actions = np.asarray(arrays["actions"])
    count = row["num_actions"]
    if actions.shape != (count, 5) or not np.isfinite(actions).all():
        raise ValueError("Expected T finite, normalized five-dimensional actions")
    if np.any(np.abs(actions) > 1 + 1e-7) or np.any(actions[:, 3] != 0):
        raise ValueError("Out-of-range action or nonzero yaw: refusing to discard it")
    if metadata.get("state_alignment") != "state[i] -- actions[i] --> state[i+1]":
        raise ValueError("Unrecognized state/action alignment")
    if metadata.get("action_order") != ["dx", "dy", "dz", "yaw_zero_only", "dgripper"]:
        raise ValueError("Unrecognized source action order")
    if (metadata.get("status") != "complete" or
            metadata.get("num_actions") != count or
            metadata.get("num_states") != count + 1):
        raise ValueError("Episode metadata/count mismatch")
    time = np.asarray(arrays["time"])
    dt = float(metadata["control_dt"])
    if (time.shape != (count + 1,) or not np.isfinite(time).all() or dt <= 0 or
            not np.allclose(np.diff(time), dt, atol=1e-7, rtol=0)):
        raise ValueError("Invalid timestamps or transition alignment")
    for field in ("class_label", "unsafe", "hover_conditions_met", "hover_success", "episode_failed"):
        values = np.asarray(arrays[field])
        if values.shape != (count + 1,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid state-aligned field: {field}")
    if not np.isin(arrays["class_label"], [0, 1, 2]).all():
        raise ValueError("Unexpected class label")
    expected = np.where(arrays["unsafe"], 1, np.where(arrays["hover_conditions_met"], 0, 2))
    if not np.array_equal(arrays["class_label"], expected):
        raise ValueError("Stored labels disagree with stored current-state flags")
    # This is consistency checking, NOT a fresh validation of the simulator's
    # tilt/hold calculations. Known scene-test failures remain a separate task.
    goals = np.asarray(arrays["hover_goal_xyz"])
    goal = np.asarray(metadata["hover_goal_xyz"])
    if (goals.shape != (count + 1, 3) or goal.shape != (3,) or
            not np.isfinite(goals).all() or
            not np.allclose(goals, goal, atol=1e-9, rtol=0)):
        raise ValueError("Hover goal is missing, nonfinite or changes within an episode")
    return actions[:, ACTION_COLUMNS].astype(np.float32)


def scene_signature(metadata):
    keys = ("schema_version", "dataset_kind", "image", "control_dt", "physics_dt",
            "action_order", "action_low", "action_high", "fixed_yaw_deg", "blocks",
            "unsafe_tilt_deg", "displacement_check_enabled", "safety_scope")
    image = metadata["image"]
    if image.get("camera") != "front_pixels" or image.get("overlay") is not False:
        raise ValueError("Pilot requires clean fixed-front camera images without overlays")
    return {key: metadata[key] for key in keys}


def action_statistics(action_sum, action_sq_sum, count):
    if count < 2:
        raise ValueError("Not enough TRAIN actions to fit a normalizer")
    mean = action_sum / count
    variance = np.maximum((action_sq_sum - count * mean**2) / (count - 1), 0)
    std = np.sqrt(variance)
    constant = std < 1e-6
    # A constant channel is centered but never divided by zero.
    scale = np.where(constant, 1.0, std)
    return {"fit_split": "train", "num_actions": count, "names": ACTION_NAMES,
            "mean": mean.tolist(), "std": std.tolist(), "scale": scale.tolist(),
            "constant_channels": np.flatnonzero(constant).tolist()}


def convert_episode(source, destination, row, image_size, reference_scene):
    episode = contained_path(source, row["path"])
    meta_path = contained_path(episode, "metadata.json")
    npz_path = contained_path(episode, "trajectory.npz")
    metadata = read_json(meta_path)
    if scene_signature(metadata) != reference_scene:
        raise ValueError(f"Mixed camera/scene/action settings at episode {row['id']}")
    meta_hash, npz_hash = sha256(meta_path), sha256(npz_path)
    with np.load(npz_path, allow_pickle=False) as raw:
        action = validate_arrays(raw, metadata, row)
        n_states = len(action) + 1
        images_dir = contained_path(episode, "images")
        pngs = list(images_dir.glob("*.png"))
        if len(pngs) != n_states:
            raise ValueError(f"Image count mismatch at episode {row['id']}")
        with h5py.File(destination, "x") as output:
            output.attrs.update(format=FORMAT, complete=False, episode_id=row["id"],
                                source_metadata_sha256=meta_hash, source_npz_sha256=npz_hash,
                                metadata_json=json.dumps(metadata, ensure_ascii=False),
                                alignment="pixels[t] -- action[t] --> pixels[t+1]")
            pixels = output.create_dataset("pixels", shape=(n_states, image_size, image_size, 3),
                                           dtype="u1", chunks=(1, image_size, image_size, 3),
                                           compression="lzf")
            output.create_dataset("action", data=action)
            # Preserve diagnostic truth separately, never returned as WM input.
            truth = output.create_group("diagnostics")
            for key in raw.files:
                values = raw[key]
                if values.dtype.kind in "biuf":
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite raw diagnostic field {key}")
                    truth.create_dataset(key, data=values, compression="lzf" if values.ndim else None)
            rgb_digest = hashlib.sha256()
            for i in range(n_states):
                path = contained_path(episode, f"images/{i:06d}.png")
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (metadata["image"]["width"], metadata["image"]["height"]):
                        raise ValueError(f"Unexpected raw image: {path}")
                    frame = letterbox(image, image_size)
                pixels[i] = frame
                rgb_digest.update(frame.tobytes())
            if sha256(meta_path) != meta_hash or sha256(npz_path) != npz_hash:
                raise RuntimeError("Source changed while converting; cache is incomplete")
            output.attrs["pixels_sha256"] = rgb_digest.hexdigest()
            output.attrs["complete"] = True
        counts = Counter(int(i) for i in raw["class_label"])
    info = {"id": row["id"], "mode": row["mode"], "path": f"episodes/{destination.name}",
            "num_actions": len(action), "num_states": n_states,
            "class_counts": {str(i): counts[i] for i in range(3)},
            "source_npz_sha256": npz_hash, "source_metadata_sha256": meta_hash,
            "file_bytes": destination.stat().st_size}
    return info, action.astype(np.float64)


def prepare(source, output, *, train_episodes=80, val_episodes=20,
            seed=20260921, image_size=224):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; use a NEW directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Cache must be separate from the raw dataset, not inside it or its parent")
    if train_episodes < 2 or val_episodes < 1 or seed < 0:
        raise ValueError("Require train>=2, val>=1, seed>=0")
    if image_size < 28 or image_size % 14:
        raise ValueError("Image size must be a positive multiple of LeWM's patch size 14")
    manifest_path = source / "manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = read_json(manifest_path)
    splits = make_split(manifest, seed)
    if train_episodes > len(splits["train"]) or val_episodes > len(splits["val"]):
        raise ValueError("Requested pilot size exceeds its reserved split; do not borrow test episodes")
    selected = {"train": splits["train"][:train_episodes], "val": splits["val"][:val_episodes]}
    rows = {row["id"]: row for row in manifest["episodes"]}
    first = contained_path(source, rows[selected["train"][0]]["path"])
    signature = scene_signature(read_json(contained_path(first, "metadata.json")))
    # Conservative uncompressed-pixels budget plus 20% and 512 MiB headroom.
    total_states = sum(rows[i]["num_states"] for ids in selected.values() for i in ids)
    estimated_bytes = int(total_states * image_size**2 * 3 * 1.2 + 512 * 1024**2)
    existing_parent = output.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < estimated_bytes:
        raise OSError(f"Insufficient free space for conservative cache budget {estimated_bytes / 2**30:.1f} GiB")
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    plan = {"format": FORMAT, "source_root": str(source), "source_manifest_sha256": manifest_hash,
            "seed": seed, "split_method": "mode-interleaved shuffled whole episodes; 90/5/5",
            "split_ids": splits, "selected_ids": selected, "reserved_test_not_read": True,
            "image_size": image_size, "resize": "LANCZOS letterbox; no crop; RGB ImageNet-mean padding",
            "pixel_mean": IMAGE_MEAN, "pixel_std": IMAGE_STD,
            "source_action_columns": list(ACTION_COLUMNS), "action_names": ACTION_NAMES,
            "frameskip": 1, "control_dt": signature["control_dt"], "scene": signature,
            "raw_label_semantics": manifest["class_names"],
            "labels_used_for_wm_training": False,
            "label_audit_scope": "stored flag consistency only; not proof of simulator-label correctness"}
    write_new_json(output / "plan.json", plan)
    prepared = {"train": [], "val": []}
    action_sum, action_sq_sum, count = np.zeros(4), np.zeros(4), 0
    completed, total = 0, train_episodes + val_episodes
    try:
        for split, ids in selected.items():
            for episode_id in ids:
                path = output / "episodes" / f"episode_{episode_id:04d}.h5"
                info, actions = convert_episode(source, path, rows[episode_id], image_size, signature)
                prepared[split].append(info)
                if split == "train":
                    action_sum += actions.sum(axis=0)
                    action_sq_sum += (actions**2).sum(axis=0)
                    count += len(actions)
                completed += 1
                print(f"[{completed}/{total}] {split} episode_{episode_id:04d}: "
                      f"{info['num_states']} frames, {info['file_bytes']/2**20:.1f} MiB", flush=True)
        if sha256(manifest_path) != manifest_hash:
            raise RuntimeError("Source manifest changed while preparing")
        result = {**plan, "status": "complete", "episodes": prepared,
                  "normalization": action_statistics(action_sum, action_sq_sum, count),
                  "total_bytes": sum(r["file_bytes"] for rs in prepared.values() for r in rs)}
        # This is the commit marker: training refuses directories without it.
        write_new_json(output / "index.json", result)
        print(f"Cache ready: {output / 'index.json'}", flush=True)
        return result
    except BaseException as error:
        write_new_json(output / "INCOMPLETE.json", {"error": str(error), "completed_episodes": completed,
                                                    "note": "Not usable for training; raw data is untouched."})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-episodes", type=int, default=80)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    prepare(args.source, args.output, train_episodes=args.train_episodes,
            val_episodes=args.val_episodes, seed=args.seed, image_size=args.image_size)


if __name__ == "__main__":
    main()
