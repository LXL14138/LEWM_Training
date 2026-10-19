"""Convert RRT-law branch rollouts (rrt_branches_v1, collect_rrt_branches.py) into a fine-tune cache.

TRAINING venv (~/lewm-train/.venv: numpy, h5py, torch). Implements runs/rrt_data_plan_005/plan.json 'conversion'
exactly; Chinese doc: CONVERT_RRT_BRANCHES.md.

Roles: --role train (data/rrt_branches_001 -> data/wm_rrt_branches_001, rows part 'test' / bench_test = training rows of
finetune_tilt --extra-cache) and --role dev (data/rrt_branches_dev_001 -> data/wm_rrt_dev_001, rows part 'dev' / bench_dev =
finetune_tilt --extra-dev-cache). The confirmation set (role 'confirm', random_branches_*) is NEVER converted.

One cache episode per LAW rollout ('recorded' duplicates its source episode and is skipped):
  states  = prefix_pixels[8:11] (frames a-2, a-1, a) + pixels[r, 1:33]                  -> 35 states
  actions = prefix_actions[8:10] + actions[r], columns [0, 1, 2, 4] as float32            -> 34 rows (env commands;
            normalised later by the index normalization, which is COPIED from data/wm_full_001)
  alignment pixels[t] -- action[t] --> pixels[t+1]; prefix_frames = 2 (state 2 is the anchor a)
  labels  = per-state diagnostics prefix_<field>[8:11] + <field>[r, 1:33]. G anchors (stratum G) whose goal_source is
            'goal_image_achieved_com': hover_error_m / hover_conditions_met / class 0 are RECOMPUTED against goal_xyz
            (grasped, |green COM - goal| <= 1.5 cm, green speed <= 2 cm/s, the scene's 1e-9 slack; class 1 unchanged).
            Every other anchor is labelled against its hover goal and the recomputation must reproduce the recorded values
            bit for bit (checked). That includes T / F / M anchors whose episode has a linked goal image: the planner gives
            them goal_source 'goal_image_achieved_com' too, but plan.json says '(G)' and PLAN_ZH.md says T / F / M keep the
            hover goal, 'only recorded, not used for the goal labels' (label_goal_source = 'hover_goal' on those rows).
            diagnostics/hover_goal_xyz = the label goal point; source_hover_goal_xyz and recorded_* keep the originals.
Episode id = anchor_id * 16 + rollout; source_npz_sha256 = sha256(ascii '<anchor npz sha256>:<rollout>') (unique per
rollout; it names no raw file, so finetune_tilt's raw-hash overlap checks cannot see the upstream episode - the converter
checks the upstream raw trajectory.npz sha256 itself, before writing and again from the written index); source_metadata_sha256
= sha256 of the dataset manifest row (JSON, sorted keys).
The index is an eval-cache index like prepare_bench (eval_cache_revision 2): contract and normalization copied from the
reference cache, every episode in val, bench_split by role, source_provenance 'rrt_branches', upstream_source_roots = the
raw datasets the anchors were restored from, upstream_split_files = [the danger_train split file] (always, as the plan
says), per-row source_part / branch / prefix_frames (finetune_tilt's guard reads them), anchor file + npz sha256 + the
upstream raw sha256 per row and in episode_sources.
Datasets outside the stage-2 plan (the pilots; plan.json conversion.outputs.never_converted) are converted only with
--check-only, into a directory OUTSIDE the data root (the parent of --reference-cache); their index says check_only true.

Checks: dataset manifest complete and bound to the collector report; every anchor npz sha256; the dataset's role and
every start training-side (train) / dev-side (dev) by source name, raw-dataset family, split ids and raw sha256 (never a
val REPORT / TEST / bench_test / danger_bench episode); wm_action / prefix_wm_action reproduced bit for bit from the
executed actions with the index normalization; state 2 = pixels[r, 0]; safety_classifier.audit_labels per episode;
afterwards eval_cache.check_eval_cache against the reference cache, no raw-episode overlap with it, TiltWindows(stride 3)
opens every row (8 windows per episode), a pixel / action / label alignment spot check through TiltWindows, and
finetune_tilt.extra_cache_guard(role) with the bench and hold-out caches. The plan's datasets (rrt_branches_NNN /
rrt_branches_dev_NNN) are converted whole into the plan's names (or the next free number) with --stage2-plan; any other
dataset (the pilots) only into other names. NEW output directory only; FAILED.json on error (index.json renamed to
index.failed.json so a failed cache cannot be used); summary.json only on success.
release: this copy ships without the training modules (eval_cache, finetune_dataset, finetune_tilt). audit_labels comes from
wm_training/label_audit.py (a verbatim extract of safety_classifier.py); the post-conversion audit (check_eval_cache, TiltWindows,
the alignment spot check, extra_cache_guard) is skipped when those modules are absent, one console line says so and
summary.json records verification.post_conversion_audit = 'skipped: training modules not shipped in this release'.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util                                       # release: to detect the absent training modules
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import numpy as np

import rrt_branches_plan as rp
from wm_training.danger_strata import native
from wm_training.prepare_data import FORMAT as CACHE_FORMAT, contained_path, read_json, sha256, write_new_json
from wm_training.label_audit import audit_labels           # release: verbatim extract of wm_training.safety_classifier.audit_labels

FORMAT = "rrt_branches_cache_v1"
DATASET_FORMAT = "rrt_branches_v1"
ROLES = ("train", "dev")
ROLE_PART = {"train": "test", "dev": "dev"}                 # finetune_tilt guard vocabulary: test = training half, dev = validation half
ROLE_BENCH_PART = {"train": "bench_test", "dev": "bench_dev"}
PREFIX_SLICE = slice(8, 11)                                 # prefix frames a-2, a-1, a (prefix index 10 = the anchor)
PREFIX_ACTION_SLICE = slice(8, 10)                          # recorded actions a-2, a-1
PREFIX_FRAMES = 2
HORIZON = 32
STATES, ACTIONS = PREFIX_FRAMES + 1 + HORIZON, PREFIX_FRAMES + HORIZON
ID_STRIDE = 16
WM_COLUMNS = [0, 1, 2, 4]
# collect_random_branches.STEP_FIELDS and collect_rrt_branches.CONTACT_FIELDS (those modules import MuJoCo; the lists are
# repeated here and checked against the npz)
STEP_FIELDS = ("guard_tilt_deg", "unsafe", "class_label", "hover_error_m", "target_speed_mps", "hover_conditions_met", "grasped",
               "object_com_xyz", "object_quat_wxyz", "pinch_xyz", "guard_contact_force_n", "guard_contact_peak_last_step_n",
               "green_guard_contact_peak_last_step_n", "physics_state")
CONTACT_FIELDS = ("robot_env_contact", "robot_green_contact", "green_env_contact")
NPZ_KEYS = ("pixels", "prefix_pixels", "actions", "wm_action", "prefix_actions", "prefix_wm_action", "families", "anchor_id", "stratum",
            "set", "goal_xyz", "goal_source", "goal_image_id", "hover_goal_xyz", "source_frame", "cache_episode_id", "cache",
            *STEP_FIELDS, *CONTACT_FIELDS, *(f"prefix_{k}" for k in STEP_FIELDS + CONTACT_FIELDS))
# exploration.py (ExplorationScene.post_step): GOAL_TOLERANCE_M, GOAL_SPEED_MPS and the 1e-9 comparison slack
HOVER_TOLERANCE_M, HOVER_SPEED_MPS, RULE_EPS = 0.015, 0.02, 1e-9
GOAL_SOURCES = ("hover_goal", "goal_image_achieved_com")
# plan.json training_collection / branch_dev_set datasets and conversion outputs (checked against the plan when given)
PLAN_DATASETS = {"train": ("data/rrt_branches_001", "data/wm_rrt_branches_001"), "dev": ("data/rrt_branches_dev_001", "data/wm_rrt_dev_001")}
BYTES_PER_STATE = 19381                                     # plan.json conversion.sizes.bytes_per_state_measured (rounded up)
DISK_RESERVE_BYTES = 5 * 1024**3
ALIGNMENT_SPOT_EPISODES = 8
IMPLEMENTATION_ROOT = ("convert_rrt_branches.py", "rrt_branches_plan.py")
IMPLEMENTATION_WM = ("prepare_data.py", "label_audit.py", "dataset.py", "danger_strata.py", "provenance.py")   # release: shipped modules only
AUDIT_MODULES = ("wm_training.eval_cache", "wm_training.finetune_dataset", "wm_training.finetune_tilt")       # release: not shipped
AUDIT_SKIPPED = "skipped: training modules not shipped in this release"                                        # release:


# ----- labels -------------------------------------------------------------------------------------------------------
def goal_labels(green_com, speed, grasped, unsafe, goal):
    """(hover_error_m, hover_conditions_met, class_label) of every state against ``goal`` with the scene's own arithmetic
    (ExplorationScene.post_step: float(np.linalg.norm(xipos - goal)), <= tol + 1e-9, grasped & in_goal & slow;
    class 1 if unsafe else 0 if conditions else 2)."""
    goal = np.asarray(goal, dtype=np.float64)
    com = np.asarray(green_com, dtype=np.float64)
    if goal.shape != (3,) or com.ndim != 2 or com.shape[1] != 3:
        raise ValueError("goal_labels: need (3,) goal and (N, 3) green COM")
    error = np.array([float(np.linalg.norm(c - goal)) for c in com], dtype=np.float64)
    in_goal = error <= HOVER_TOLERANCE_M + RULE_EPS
    slow = np.asarray(speed, dtype=np.float64) <= HOVER_SPEED_MPS + RULE_EPS
    conditions = np.asarray(grasped, bool) & in_goal & slow
    classes = np.where(np.asarray(unsafe, bool), 1, np.where(conditions, 0, 2))
    return error, conditions, classes


def label_goal_rule(stratum, goal_source):
    """Which goal point the class-0 labels of an anchor are computed against: the goal image's achieved COM only for G
    anchors that carry one (plan.json conversion.labels '(G)'; PLAN_ZH.md: T / F / M keep the trajectory's hover goal,
    'only recorded, not used for the goal labels'); the hover goal (= the recorded labels, checked bit for bit) otherwise."""
    if goal_source not in GOAL_SOURCES:
        raise ValueError(f"unknown goal_source {goal_source!r}")
    return "goal_image_achieved_com" if (stratum == "G" and goal_source == "goal_image_achieved_com") else "hover_goal"


def rollout_source_sha256(anchor_sha256, rollout):
    return hashlib.sha256(f"{anchor_sha256}:{int(rollout)}".encode("ascii")).hexdigest()


def manifest_row_sha256(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def episode_states(z, r):
    """Per-state arrays (35 states: a-2, a-1, a, rollout states 1..32) of law rollout r, pixels and 34 raw actions."""
    diag = {k: np.concatenate([np.asarray(z[f"prefix_{k}"])[PREFIX_SLICE], np.asarray(z[k])[r, 1:]]) for k in STEP_FIELDS + CONTACT_FIELDS}
    pixels = np.concatenate([np.asarray(z["prefix_pixels"])[PREFIX_SLICE], np.asarray(z["pixels"])[r, 1:]])
    actions5 = np.concatenate([np.asarray(z["prefix_actions"], np.float64)[PREFIX_ACTION_SLICE], np.asarray(z["actions"], np.float64)[r]])
    wm = np.concatenate([np.asarray(z["prefix_wm_action"])[PREFIX_ACTION_SLICE], np.asarray(z["wm_action"])[r]]).astype(np.float32)
    return diag, pixels, actions5, wm


# ----- one anchor (worker) --------------------------------------------------------------------------------------------
def read_anchor(dataset, row):
    """anchor npz read once from bytes after its sha256 matched the manifest row."""
    path = contained_path(dataset, row["file"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError(f"{path}: sha256 differs from the dataset manifest")
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        missing = [k for k in NPZ_KEYS if k not in npz.files]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing[:6]} (the prefix_* diagnostics exist from the 2026-09-28 collector on; "
                             "rrt_branches_pilot_001 / _002 cannot be converted)")
        return {k: np.asarray(npz[k]) for k in NPZ_KEYS}


def convert_anchor(job):
    """Every law rollout of one anchor -> cache episodes. Returns (index rows, per-anchor statistics)."""
    dataset, output, row, size = Path(job["dataset"]), Path(job["output"]), job["row"], int(job["image_size"])
    mean, scale = np.asarray(job["mean"], np.float32), np.asarray(job["scale"], np.float32)
    z = read_anchor(dataset, row)
    families = [str(f) for f in z["families"]]
    R = len(families)
    if not families or families[0] != "recorded" or R > ID_STRIDE:
        raise ValueError(f"anchor {row['id']}: rollout 0 must be 'recorded' and at most {ID_STRIDE} rollouts")
    meta = {"anchor_id": int(z["anchor_id"]), "stratum": str(z["stratum"]), "set": str(z["set"])}
    if meta != {"anchor_id": int(row["id"]), "stratum": row["stratum"], "set": row["set"]}:
        raise ValueError(f"anchor {row['id']}: npz metadata {meta} differs from the manifest row")
    shapes = {"pixels": (R, HORIZON + 1, size, size, 3), "prefix_pixels": (rp.PREFIX + 1, size, size, 3), "actions": (R, HORIZON, 5),
              "wm_action": (R, HORIZON, 4), "prefix_actions": (rp.PREFIX, 5), "prefix_wm_action": (rp.PREFIX, 4), "goal_xyz": (3,),
              "hover_goal_xyz": (3,), "guard_tilt_deg": (R, HORIZON + 1, 2), "prefix_guard_tilt_deg": (rp.PREFIX + 1, 2)}
    for k, shape in shapes.items():
        if z[k].shape != shape:
            raise ValueError(f"anchor {row['id']}: {k} has shape {z[k].shape}, expected {shape}")
    if z["pixels"].dtype != np.uint8 or z["prefix_pixels"].dtype != np.uint8:
        raise ValueError(f"anchor {row['id']}: pixels must be uint8")
    if not (z["pixels"][:, 0] == z["prefix_pixels"][rp.PREFIX][None]).all():
        raise ValueError(f"anchor {row['id']}: pixels[:, 0] differ from the anchor frame prefix_pixels[10]")
    goal_source = str(z["goal_source"])
    goal, hover_goal = np.asarray(z["goal_xyz"], np.float64), np.asarray(z["hover_goal_xyz"], np.float64)
    if goal_source not in GOAL_SOURCES:
        raise ValueError(f"anchor {row['id']}: unknown goal_source {goal_source!r}")
    if goal_source == "hover_goal" and not np.array_equal(goal, hover_goal):
        raise ValueError(f"anchor {row['id']}: goal_source hover_goal but goal_xyz != hover_goal_xyz")
    label_goal_source = label_goal_rule(meta["stratum"], goal_source)
    label_goal = goal if label_goal_source == "goal_image_achieved_com" else hover_goal
    plan_anchor = row.get("plan_anchor") or {}
    src = plan_anchor.get("source") or {}
    anchor_meta_sha = manifest_row_sha256(row)
    out_rows, relabelled_frames, relabelled_episodes = [], 0, 0
    for r, family in enumerate(families):
        if family == "recorded":
            continue
        eid = int(row["id"]) * ID_STRIDE + r
        diag, pixels, actions5, wm_stored = episode_states(z, r)
        if not np.isfinite(actions5).all() or np.any(np.abs(actions5) > 1 + 1e-7) or np.any(actions5[:, 3] != 0):
            raise ValueError(f"anchor {row['id']} rollout {r}: actions must be normalised commands with yaw 0")
        action = actions5[:, WM_COLUMNS].astype(np.float32)
        if not np.array_equal(((action - mean) / scale).astype(np.float32), wm_stored):
            raise ValueError(f"anchor {row['id']} rollout {r}: the index normalization does not reproduce the collector's wm_action")
        if not np.array_equal(pixels[PREFIX_FRAMES], z["pixels"][r, 0]):
            raise ValueError(f"anchor {row['id']} rollout {r}: state 2 differs from pixels[r, 0]")
        error, conditions, classes = goal_labels(diag["object_com_xyz"][:, 0], diag["target_speed_mps"], diag["grasped"], diag["unsafe"], label_goal)
        recorded = {k: diag[k].copy() for k in ("hover_error_m", "hover_conditions_met", "class_label")}
        classes = classes.astype(recorded["class_label"].dtype)
        if label_goal_source == "hover_goal":
            same = (np.array_equal(error, recorded["hover_error_m"]) and np.array_equal(conditions, recorded["hover_conditions_met"])
                    and np.array_equal(classes, recorded["class_label"]))
            if not same:
                raise ValueError(f"anchor {row['id']} rollout {r}: recomputing the goal labels against the hover goal does not reproduce "
                                 "the recorded hover_error_m / hover_conditions_met / class_label bit for bit")
        changed = int((classes != recorded["class_label"]).sum())
        relabelled_frames += changed
        relabelled_episodes += int(changed > 0)
        diag.update(hover_error_m=error, hover_conditions_met=conditions, class_label=classes,
                    hover_goal_xyz=np.tile(label_goal, (STATES, 1)), source_hover_goal_xyz=np.tile(hover_goal, (STATES, 1)),
                    recorded_hover_error_m=recorded["hover_error_m"], recorded_hover_conditions_met=recorded["hover_conditions_met"],
                    recorded_class_label=recorded["class_label"], step_from_anchor=np.arange(-PREFIX_FRAMES, HORIZON + 1, dtype=np.int64))
        for k, v in diag.items():
            if len(v) != STATES or (v.dtype.kind == "f" and not np.isfinite(v).all()):
                raise ValueError(f"anchor {row['id']} rollout {r}: diagnostics {k} not state-aligned / finite")
        audit_labels(diag, STATES)                                                    # quaternion -> tilt -> 30-degree rule
        source_sha = rollout_source_sha256(row["sha256"], r)
        metadata = {"format": FORMAT, "dataset": str(dataset), "anchor_file": row["file"], "anchor_npz_sha256": row["sha256"], "anchor_id": int(row["id"]),
                    "rollout": r, "family": family, "family_set": plan_anchor.get("family_set"), "stratum": row["stratum"], "set": row["set"],
                    "source_cache": str(z["cache"]), "source_episode_id": int(z["cache_episode_id"]), "source_frame": int(z["source_frame"]),
                    "goal_source": goal_source, "goal_xyz": goal.tolist(), "goal_image_id": int(z["goal_image_id"]),
                    "label_goal_source": label_goal_source, "label_goal_xyz": label_goal.tolist(), "hover_goal_xyz": hover_goal.tolist(),
                    "states": "prefix frames a-2, a-1, a (index 0..2) + rollout states 1..32 (index 3..34)", "prefix_frames": PREFIX_FRAMES,
                    "role": job["role"], "check_only": bool(job.get("check_only"))}
        dest = output / "episodes" / f"episode_{eid:07d}.h5"
        digest = hashlib.sha256()
        with h5py.File(dest, "x") as f:
            f.attrs.update(format=CACHE_FORMAT, complete=False, episode_id=eid, source_npz_sha256=source_sha,
                           source_metadata_sha256=anchor_meta_sha, metadata_json=json.dumps(metadata, ensure_ascii=False),
                           alignment="pixels[t] -- action[t] --> pixels[t+1]")
            px = f.create_dataset("pixels", shape=(STATES, size, size, 3), dtype="u1", chunks=(1, size, size, 3), compression="lzf")
            for i in range(STATES):
                px[i] = pixels[i]
                digest.update(np.ascontiguousarray(pixels[i]).tobytes())
            f.create_dataset("action", data=action)
            truth = f.create_group("diagnostics")
            for k, v in diag.items():
                truth.create_dataset(k, data=v, compression="lzf" if v.ndim else None)
            f.attrs["pixels_sha256"] = digest.hexdigest()
            f.attrs["complete"] = True
        counts = Counter(int(c) for c in classes)
        out_rows.append({"id": eid, "mode": family, "path": f"episodes/{dest.name}", "num_actions": ACTIONS, "num_states": STATES,
                         "class_counts": {str(i): counts[i] for i in range(3)}, "source_npz_sha256": source_sha,
                         "source_metadata_sha256": anchor_meta_sha, "file_bytes": dest.stat().st_size, "anchor_file": row["file"],
                         "anchor_npz_sha256": row["sha256"], "anchor_id": int(row["id"]), "rollout": r, "source_part": ROLE_PART[job["role"]],
                         "branch": family, "lead": None, "prefix_frames": PREFIX_FRAMES, "stratum": row["stratum"], "set": row["set"],
                         "family_set": plan_anchor.get("family_set"), "goal_source": goal_source, "label_goal_source": label_goal_source,
                         "relabelled_class_frames": changed, "upstream_npz_sha256": src.get("npz_sha256"),
                         "source": {"cache": str(z["cache"]), "episode_id": int(z["cache_episode_id"]), "frame": int(z["source_frame"]),
                                    "episode_dir": src.get("episode_dir"), "npz_sha256": src.get("npz_sha256")}})
    return out_rows, {"anchor": int(row["id"]), "episodes": len(out_rows), "stratum": meta["stratum"], "goal_source": goal_source,
                      "label_goal_source": label_goal_source, "relabelled_frames": relabelled_frames, "relabelled_episodes": relabelled_episodes,
                      "label_goal_offset_m": float(np.linalg.norm(goal - hover_goal))}


# ----- inputs -------------------------------------------------------------------------------------------------------------
def load_dataset(dataset, report):
    """(manifest, sha256) of a complete rrt_branches_v1 dataset bound to its collector report."""
    dataset, report = Path(dataset), Path(report)
    manifest_path = dataset / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("format") != DATASET_FORMAT or manifest.get("status") != "complete" or (dataset / "FAILED.json").exists():
        raise ValueError(f"{dataset}: not a complete {DATASET_FORMAT} dataset")
    digest = sha256(manifest_path)
    summary = read_json(report / "summary.json")
    if summary.get("manifest_sha256") != digest or (report / "FAILED.json").exists():
        raise ValueError(f"{report}: collector summary not bound to {dataset} (manifest sha256) or the run failed")
    if (prefix := int(manifest["prefix"])) != rp.PREFIX or int(manifest["horizon"]) != HORIZON:
        raise ValueError(f"{dataset}: prefix {prefix} / horizon {manifest['horizon']} differ from this converter")
    return manifest, digest


def dataset_role(manifest, dataset):
    """The dataset's collection role; the confirmation set (role 'confirm', or any random_branches_* dataset, or any
    val REPORT / bench_test start) is never converted."""
    role = manifest.get("role", "train")
    sets = {str(r["set"]) for r in manifest["anchors"]}
    if role == "confirm" or re.fullmatch(r"random_branches_\d{3}", Path(dataset).name) or sets & set(rp.ROLE_SOURCES["confirm"]):
        raise ValueError(f"{dataset}: the confirmation set is never converted (role {role!r}, sets {sorted(sets)})")
    if role not in ROLES:
        raise ValueError(f"{dataset}: unknown role {role!r}")
    return role


def check_sources(manifest, role, ref_index, bench_index, holdout_index, classifier, bench_split):
    """Every start is from the role's side: source names, raw-dataset family, split ids, and the raw trajectory.npz
    sha256 equal to the cache row it came from and absent from the hold-out (real benchmark) cache. Returns the upstream
    raw roots and counts."""
    train_ids = {int(r["id"]) for r in ref_index["episodes"]["train"]}
    val_rows = {int(r["id"]): r for r in ref_index["episodes"]["val"]}
    train_rows = {int(r["id"]): r for r in ref_index["episodes"]["train"]}
    held = {int(r["id"]) for r in ref_index["episodes"].get("test", [])} | {int(i) for i in ref_index["split_ids"].get("test", [])}
    dev_ids, report_ids = {int(i) for i in classifier["dev_episode_ids"]}, {int(i) for i in classifier["report_episode_ids"]}
    bench_rows = {int(r["id"]): r for r in bench_index["episodes"]["val"]}
    bench_parts = {k: {int(i) for i in v} for k, v in bench_index["bench_split"].items()}
    hold_hashes = {r.get("source_npz_sha256") for part in holdout_index["episodes"].values() for r in part}
    split_roots = {str(Path(d["path"])) for d in bench_split["datasets"]}
    ref_root = str(Path(ref_index["source_root"]))
    roots, counts = set(), Counter()
    for row in manifest["anchors"]:
        pa = row.get("plan_anchor") or {}
        src = pa.get("source") or {}
        name = src.get("source")
        if row["set"] != name:
            raise ValueError(f"anchor {row['id']}: manifest set {row['set']!r} differs from its plan source {name!r}")
        rp.check_anchor_side(role, pa)                                         # role sources + raw-dataset family
        eid = int(src["episode_id"])
        if name in ("full_train", "val_dev"):
            if name == "full_train":
                ok, cache_row = eid in train_ids and eid not in held and eid not in val_rows, train_rows.get(eid)
            else:
                ok, cache_row = eid in dev_ids and eid not in report_ids and eid in val_rows and eid not in held, val_rows.get(eid)
            root_ok = str(Path(src["episode_dir"]).parent) == ref_root
        else:
            part = "bench_test" if name == "bench_train" else "bench_dev"
            ok, cache_row = eid in bench_parts.get(part, set()) and all(eid not in v for k, v in bench_parts.items() if k != part), bench_rows.get(eid)
            root_ok = str(Path(src["episode_dir"]).parent) in split_roots
        if not ok or cache_row is None or not root_ok:
            raise ValueError(f"anchor {row['id']}: {name} episode {eid} is not a {role}-side start of the reference caches")
        if cache_row.get("source_npz_sha256") != src["npz_sha256"]:
            raise ValueError(f"anchor {row['id']}: raw sha256 differs from the cache row it names")
        if src["npz_sha256"] in hold_hashes:
            raise ValueError(f"anchor {row['id']}: its raw episode belongs to the hold-out (real benchmark) cache")
        roots.add(str(Path(src["episode_dir"]).parent))
        counts[name] += 1
    return sorted(roots), dict(counts)


def plan_dataset_rule(dataset, output, role, stage2_plan, anchor_ids, check_only=False, data_root=None):
    """The plan's datasets are converted whole into the plan's names (with --stage2-plan). Every other dataset (the pilots:
    plan.json conversion.outputs.never_converted) only with --check-only, into a directory outside ``data_root`` (the
    parent of --reference-cache) and never into the plan's output names; its index is marked check_only (not training or
    selection material). Returns the record for run.json."""
    planned_in, planned_out = PLAN_DATASETS[role]
    stage2 = None
    if stage2_plan is not None:
        doc, stage2 = rp.load_stage2_plan(stage2_plan)
        section = doc[rp.STAGE2_SECTIONS[role]]
        outputs = doc["conversion"]["outputs"][role]
        if section["dataset"] != planned_in or not str(outputs).startswith(planned_out + " "):
            raise ValueError("the stage-2 plan's dataset / conversion names differ from this converter's")
    in_stem, out_stem = Path(planned_in).name[:-3], Path(planned_out).name[:-3]
    is_plan_dataset = re.fullmatch(re.escape(in_stem) + r"\d{3}", Path(dataset).name) is not None
    if is_plan_dataset:
        if check_only:
            raise ValueError(f"{dataset} is a stage-2 plan dataset: it is converted whole for its role, never as a --check-only conversion")
        if stage2 is None:
            raise ValueError(f"{dataset} is a stage-2 plan dataset: pass --stage2-plan runs/rrt_data_plan_005/plan.json")
        if anchor_ids is not None:
            raise ValueError("a stage-2 plan dataset is converted whole (no --anchor-ids)")
        return {"plan_dataset": True, "check_only": False, "stage2_plan": stage2, "output_name": rp.check_stage2_name(output, planned_out),
                "input_name": rp.check_stage2_name(dataset, planned_in)}
    if not check_only:
        raise ValueError(f"{dataset} is not a stage-2 plan dataset ({planned_in} or the next number): plan.json conversion.outputs."
                         "never_converted lists the pilots; such a dataset is converted only with --check-only (a check, never a cache "
                         "for finetune_tilt)")
    if re.fullmatch(r"wm_rrt_(branches|dev)_\d{3}", Path(output).name):
        raise ValueError(f"{output}: wm_rrt_branches_NNN / wm_rrt_dev_NNN are reserved for the stage-2 plan's datasets")
    if data_root is not None and Path(output).resolve().is_relative_to(Path(data_root).resolve()):
        raise ValueError(f"{output}: a --check-only conversion is written outside the data root {data_root} (e.g. a scratch directory), "
                         "so it cannot be mistaken for a training / selection cache")
    return {"plan_dataset": False, "check_only": True, "stage2_plan": stage2, "output_name": None, "input_name": None,
            "data_root": None if data_root is None else str(Path(data_root).resolve())}


def upstream_audit(index, role, ref_index, bench_index, holdout_index, classifier):
    """From the WRITTEN index: every row's upstream raw trajectory.npz sha256 (episode_sources[id].upstream_source.npz_sha256
    and the row's upstream_npz_sha256) lies in the role's own pools (train: wm_full_001 train + bench_test rows of the bench
    cache; dev: the val DEV half + bench_dev rows) and in none of: the hold-out (real benchmark) cache, wm_full_001 test /
    val REPORT half, the other bench part. finetune_tilt's own hash checks see only the per-rollout source_npz_sha256, so
    this is the check that ties the cache to real raw episodes. Returns counts (raises on any violation)."""
    def hashes(rows):
        return {str(r["source_npz_sha256"]) for r in rows if r.get("source_npz_sha256")}
    dev, report = {int(i) for i in classifier["dev_episode_ids"]}, {int(i) for i in classifier["report_episode_ids"]}
    val = ref_index["episodes"]["val"]
    by_id = {int(r["id"]): r for r in bench_index["episodes"]["val"]}
    bench = {part: hashes([by_id[int(i)] for i in ids if int(i) in by_id]) for part, ids in bench_index["bench_split"].items()}
    pools = {"train": hashes(ref_index["episodes"]["train"]), "test": hashes(ref_index["episodes"].get("test", [])),
             "val_dev": hashes([r for r in val if int(r["id"]) in dev]), "val_report": hashes([r for r in val if int(r["id"]) in report]),
             "bench_test": bench.get("bench_test", set()), "bench_dev": bench.get("bench_dev", set()),
             "holdout": {str(r.get("source_npz_sha256")) for part in holdout_index["episodes"].values() for r in part}}
    allowed = {"train": ("train", "bench_test"), "dev": ("val_dev", "bench_dev")}[role]
    sources = index.get("episode_sources") or {}
    seen, violations = set(), []
    for row in index["episodes"]["val"]:
        entry = sources.get(str(row["id"])) or {}
        digest = (entry.get("upstream_source") or {}).get("npz_sha256")
        if digest is None or digest != row.get("upstream_npz_sha256"):
            violations.append((row["id"], "no / inconsistent upstream raw sha256"))
            continue
        seen.add(digest)
        inside = [name for name in allowed if digest in pools[name]]
        outside = [name for name in pools if name not in allowed and digest in pools[name]]
        if not inside or outside:
            violations.append((row["id"], f"in {inside or 'none of ' + str(list(allowed))}, also in {outside}"))
    if violations:
        raise ValueError(f"{len(violations)} cache row(s) name an upstream raw episode outside the {role} side: {violations[:5]}")
    return {"rows": len(index["episodes"]["val"]), "distinct_upstream_episodes": len(seen), "allowed_pools": list(allowed),
            "in_holdout": 0, "in_other_pools": 0,
            "rule": "episode_sources upstream trajectory.npz sha256 vs the pools of wm_full_001 / the bench cache / the hold-out cache"}


# ----- verification (torch venv) -------------------------------------------------------------------------------------------
def expected_states(z, r):
    diag, pixels, _, wm = episode_states(z, r)
    return pixels, wm, diag["guard_tilt_deg"].max(1).astype(np.float32), diag["unsafe"].astype(np.uint8)


def check_alignment(cache, dataset, rows_by_id, episode_ids, stride=3, horizon=10):
    """Through finetune_dataset.TiltWindows (the fine-tune's reader): every window of the given episodes returns the npz
    frames, the collector's wm_action and the audited tilt / unsafe of the same states, i.e. image[i] -- action[i] -->
    image[i+1] exactly as the npz alignment pixels[r, i] -- actions[r, i] --> pixels[r, i+1]. Returns counts."""
    from wm_training.finetune_dataset import TiltWindows
    windows = TiltWindows(cache, "val", 3, horizon, stride, 1, episode_ids=list(episode_ids))
    anchors = {}
    checked = 0
    try:
        for index in range(len(windows)):
            item = windows[index]
            eid = int(item["episode_id"])
            aid, r = divmod(eid, ID_STRIDE)
            if aid not in anchors:
                anchors[aid] = read_anchor(dataset, rows_by_id[aid])
            pixels, wm, tilt, unsafe = expected_states(anchors[aid], r)
            s = int(item["start"])
            span = slice(s, s + 3 + horizon)
            if not np.array_equal(item["pixels"].permute(0, 2, 3, 1).numpy(), pixels[span]):
                raise ValueError(f"episode {eid} window {s}: pixels differ from the npz frames")
            if not np.array_equal(item["action"].numpy(), wm[s:s + 2 + horizon]):
                raise ValueError(f"episode {eid} window {s}: actions differ from the collector's wm_action")
            if not np.array_equal(item["tilt_deg"].numpy(), tilt[span]) or not np.array_equal(item["unsafe"].numpy(), unsafe[span]):
                raise ValueError(f"episode {eid} window {s}: labels differ from the npz states")
            checked += 1
    finally:
        windows.close()
    return {"episodes": len(list(episode_ids)), "windows_checked": checked, "stride": stride, "horizon": horizon,
            "rule": "TiltWindows item == npz states (prefix a-2..a + rollout 1..32), wm actions and audited labels"}


def verify_cache(output, role, reference, bench_cache, holdout_cache, dataset, rows_by_id, spot_ids):
    """The plan's post-conversion checks. Raises on the first failure; returns the record for summary.json."""
    if any(importlib.util.find_spec(name) is None for name in AUDIT_MODULES):      # release: modules absent -> skip, record it
        print(f"post-conversion audit {AUDIT_SKIPPED}", flush=True)                 # release:
        return {"post_conversion_audit": AUDIT_SKIPPED}                             # release:
    from wm_training.eval_cache import check_eval_cache
    from wm_training.finetune_dataset import TiltWindows
    from wm_training.finetune_tilt import extra_cache_guard
    index_path = Path(output) / "index.json"
    index, index_sha = read_json(index_path), sha256(index_path)
    ref_index, ref_sha = read_json(Path(reference) / "index.json"), sha256(Path(reference) / "index.json")
    check_eval_cache(ref_index, index, ref_sha)
    primary = {r["source_npz_sha256"] for part in ("train", "val", "test") for r in ref_index["episodes"].get(part, [])}
    ours = [r["source_npz_sha256"] for r in index["episodes"]["val"]]
    if primary & set(ours) or len(set(ours)) != len(ours):
        raise ValueError("a cache episode shares its source hash with the reference cache or another episode")
    windows = TiltWindows(output, "val", 3, 10, 3)                      # opens and audits every row
    n_rows = len(windows.rows)
    per_episode = (STATES - 13) // 3 + 1
    if n_rows != len(index["episodes"]["val"]) or len(windows) != per_episode * n_rows or windows.skipped_short:
        raise ValueError(f"TiltWindows: {n_rows} rows, {len(windows)} windows (expected {per_episode} per episode)")
    labels = windows.label_counts()
    windows.close()
    anchored = TiltWindows(output, "val", 3, 10, 1, 1, anchor_starts={int(r["id"]): int(r["prefix_frames"]) - 2 for r in index["episodes"]["val"]})
    if len(anchored) != n_rows:
        raise ValueError("--extra-dev-cache style anchored windows do not give one window per episode")
    anchored.close()
    alignment = check_alignment(output, dataset, rows_by_id, spot_ids)
    bench_index = read_json(Path(bench_cache) / "index.json")
    guard = extra_cache_guard(index, index_sha, bench_index, sha256(Path(bench_cache) / "index.json"), {}, Path(holdout_cache).resolve(),
                              "convert_rrt_branches --holdout-cache", extra_root=Path(output), role=role)
    return {"check_eval_cache": True, "reference_index_sha256": ref_sha, "raw_overlap_with_reference": 0, "tilt_windows_stride3": {
                "rows": n_rows, "windows": per_episode * n_rows, "per_episode": per_episode, "labels": labels},
            "extra_dev_cache_windows": n_rows, "alignment": alignment,
            "extra_cache_guard": {"role": role, "passed": True, "upstream": guard["upstream"], "overlap": guard.get("overlap"),
                                  "name_check": guard["name_check"]}}


# ----- CLI -----------------------------------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", choices=ROLES, required=True, help="train (training rows) or dev (branch-dev validation rows); never confirm")
    p.add_argument("--dataset", type=Path, required=True, help="rrt_branches_v1 dataset (collect_rrt_branches.py output)")
    p.add_argument("--report", type=Path, required=True, help="the collector's report directory of the dataset (manifest binding)")
    p.add_argument("--output", type=Path, required=True, help="NEW cache directory (plan: data/wm_rrt_branches_001 / data/wm_rrt_dev_001)")
    p.add_argument("--reference-cache", type=Path, default=Path("data/wm_full_001"), help="training cache (contract + normalization copied)")
    p.add_argument("--bench-cache", type=Path, default=Path("data/wm_train_bench_001"), help="B's bench cache (bench ids; the guard's bench index)")
    p.add_argument("--holdout-cache", type=Path, default=Path("data/wm_bench_001"), help="real benchmark cache (hash guard; the guard's hold-out)")
    p.add_argument("--bench-split", type=Path, default=Path("data/danger_train_split_001/benchmark_split.json"),
                   help="split file of the bench sources (recorded as upstream split file)")
    p.add_argument("--classifier-run", type=Path, default=Path("runs/safety_classifier_001/run.json"), help="val dev / report halves")
    p.add_argument("--stage2-plan", type=Path, default=None, help="frozen stage-2 plan (required for the plan's datasets)")
    p.add_argument("--anchor-ids", default=None, help="comma list of anchor ids (checks on pilot data only; never for the plan's datasets)")
    p.add_argument("--check-only", action="store_true",
                   help="required for any dataset outside the stage-2 plan (the pilots): a check conversion, written outside the data "
                        "root (parent of --reference-cache), index marked check_only; never for the plan's datasets")
    p.add_argument("--workers", type=int, choices=range(1, 5), default=3, help="conversion processes (1 = in process)")
    return p


def build_index(a, manifest, manifest_sha, rows, ref_index, ref_sha, roots, split_used, plan_rule, counts):
    dataset = str(a.dataset.resolve())
    part = ROLE_PART[a.role]
    ids = [int(r["id"]) for r in rows]
    bench_split = {"bench_test": ids, "bench_dev": []} if a.role == "train" else {"bench_dev": ids, "bench_test": []}
    sources = {}
    for r in rows:
        sources[str(r["id"])] = {"path": dataset, "provenance": "rrt_branches", "source_part": part, "branch": r["branch"], "lead": None,
                                 "prefix_frames": PREFIX_FRAMES, "anchor_file": r["anchor_file"], "anchor_npz_sha256": r["anchor_npz_sha256"],
                                 "anchor_id": r["anchor_id"], "rollout": r["rollout"], "family": r["mode"],
                                 "family_set": r["family_set"], "stratum": r["stratum"], "set": r["set"], "goal_source": r["goal_source"],
                                 "label_goal_source": r["label_goal_source"], "source_cache": r["source"]["cache"],
                                 "source_episode_id": r["source"]["episode_id"], "source_frame": r["source"]["frame"],
                                 "upstream_source": {"episode_dir": r["source"]["episode_dir"], "npz_sha256": r["source"]["npz_sha256"]}}
    contract = {k: ref_index[k] for k in ("format", "image_size", "resize", "pixel_mean", "pixel_std", "source_action_columns", "action_names",
                                          "frameskip", "control_dt", "scene", "raw_label_semantics")}
    upstream = {"upstream_source_roots": roots, "upstream_split_file": split_used, "job_part": part}
    check_only = bool(plan_rule.get("check_only"))
    use = ("FORBIDDEN: check-only conversion of a dataset outside the stage-2 plan (plan.json conversion.outputs.never_converted); "
           "finetune_tilt does not read this flag - do not pass this cache to it") if check_only else \
        {"train": "finetune_tilt --extra-cache (training rows)", "dev": "finetune_tilt --extra-dev-cache (selection only)"}[a.role]
    return {**contract, "eval_cache": True, "eval_cache_revision": 2, "status": "complete", "check_only": check_only, "training_use": use,
            "source_root": dataset, "source_roots": [dataset], "source_manifest_sha256": {dataset: manifest_sha},
            "source_provenance": {dataset: "rrt_branches"}, "upstream_source_roots": roots,
            "upstream_split_files": [split_used], "upstream_by_dataset": {dataset: upstream},
            "source_parts": {"dev": len(ids) if part == "dev" else 0, "test": len(ids) if part == "test" else 0},
            "split_file": None, "split_sha256": None, "reference_cache": {"path": str(a.reference_cache.resolve()), "index_sha256": ref_sha},
            "seed": None, "split_ids": {"train": [], "val": ids, "test": []}, "selected_ids": {"train": [], "val": ids, "test": []},
            "split_method": "rrt_branches: one cache episode per law rollout; every episode in val; bench_split by role",
            "bench_split": bench_split, "episode_sources": sources, "labels_used_for_wm_training": False,
            "label_audit_scope": "stored flag consistency only (safety_classifier.audit_labels); class 0 recomputed against goal_xyz for "
                                 "G anchors with goal_source goal_image_achieved_com; every other anchor (T / F / M included, whatever "
                                 "their goal_source) against its hover goal, reproduced bit for bit",
            "label_goal_point": "diagnostics/hover_goal_xyz = the label goal point: the goal image's achieved green COM for G anchors that "
                                "carry one, else the hover goal (per row: label_goal_source)",
            "reserved_test_not_read": True, "test_used_for_training": False,
            "rrt_conversion": {"format": FORMAT, "role": a.role, "dataset": dataset, "dataset_role": manifest.get("role", "train"),
                               "stage2_plan": plan_rule["stage2_plan"], "plan_dataset": plan_rule["plan_dataset"], "check_only": check_only,
                               "partial": a.anchor_ids is not None, "anchors": len({r["anchor_id"] for r in rows}), "prefix_frames": PREFIX_FRAMES,
                               "states_per_episode": STATES, "id_rule": f"anchor_id * {ID_STRIDE} + rollout",
                               "source_npz_sha256_rule": "sha256(ascii '<anchor npz sha256>:<rollout>') - names no raw file; the upstream raw "
                                                         "trajectory.npz sha256 is upstream_npz_sha256 / episode_sources.upstream_source",
                               "source_metadata_sha256_rule": "sha256 of the dataset manifest row (JSON, sorted keys)", "starts": counts},
            "episodes": {"train": [], "val": rows, "test": []},
            "normalization": {**ref_index["normalization"], "copied_from_reference": True, "reference_index_sha256": ref_sha},
            "total_bytes": int(sum(r["file_bytes"] for r in rows))}


def write_report(output, s):
    v = s["verification"]
    if "post_conversion_audit" in v:                                                                                   # release: audit skipped
        checks = f"- 检查：转换后审计未运行（{v['post_conversion_audit']}；check_eval_cache、TiltWindows、对齐抽查、extra_cache_guard 需要训练模块）。"  # release:
    else:                                                                                                              # release:
        checks = (f"- 检查：check_eval_cache 通过；与参考缓存无原始 episode 重叠；TiltWindows（步长 3）{v['tilt_windows_stride3']['rows']} 行 × "
                  f"{v['tilt_windows_stride3']['per_episode']} 个窗口；--extra-dev-cache 式锚定窗口 {v['extra_dev_cache_windows']} 个；"
                  f"对齐抽查 {v['alignment']['episodes']} 个 episode 的 {v['alignment']['windows_checked']} 个窗口（像素、动作、标签与 npz 逐位相同）；"
                  f"finetune_tilt.extra_cache_guard(role={v['extra_cache_guard']['role']}) 通过（上游 {json.dumps(v['extra_cache_guard']['upstream']['upstream_source_roots'], ensure_ascii=False)}，"
                  f"行按来源部分 {json.dumps(v['extra_cache_guard']['upstream']['rows_by_source_part'], ensure_ascii=False)}）。")
    lines = [f"# RRT 分支 → 微调缓存（角色 {s['role']}）\n",
             f"- 输入 `{s['dataset']}`（manifest sha256 `{s['dataset_manifest_sha256']}`，数据集角色 {s['dataset_role']}）；输出 `{s['output']}`；"
             f"{'部分转换（--anchor-ids，只作检查）' if s['partial'] else '全部锚点'}。",
             f"- 锚点 {s['anchors']} 个 → 缓存 episode {s['episodes']} 个（每个动作律 rollout 一个；recorded 跳过）；每个 35 帧 = a−2、a−1、a + 32 个新状态，34 个动作；"
             f"prefix_frames 2；行 source_part = {s['source_part']}。",
             f"- 起点来源：{json.dumps(s['starts'], ensure_ascii=False)}；上游原始数据：{json.dumps(s['upstream_source_roots'], ensure_ascii=False)}。",
             f"- 标签：npz 的 goal_source {json.dumps(s['goal_sources'], ensure_ascii=False)}；标签用的目标点 {json.dumps(s['label_goal_sources'], ensure_ascii=False)}"
             f"（只有 G 层且带目标画面的锚点按目标画面质心重算：{s['relabelled_g_anchors']['anchors']} 个锚点，改变类别的帧 {s['relabelled_frames']} 个、"
             f"{s['relabelled_episodes']} 个 episode）；T/F/M 里带目标画面链接的 {s['linked_goal_image_non_g_anchors']['anchors']} 个锚点"
             f"（{json.dumps(s['linked_goal_image_non_g_anchors']['by_stratum'], ensure_ascii=False)}）按方案（plan.json 的“(G)”、PLAN_ZH.md）仍用悬停目标；"
             "所有用悬停目标的锚点重算都与记录逐位相同（不同即报错）。",
             checks,                                                                                                   # release: see above
             f"- 上游原始轨迹核对（从写好的 index 重读）：{v['upstream_raw_audit']['rows']} 行、{v['upstream_raw_audit']['distinct_upstream_episodes']} 条不同的原始轨迹，"
             f"全部在本角色的池子（{', '.join(v['upstream_raw_audit']['allowed_pools'])}）里，不在 hold-out（真基准）与其他池子里。"
             "（行的 source_npz_sha256 是按 rollout 合成的哈希，不对应任何原始文件，所以 finetune_tilt 的哈希重叠检查对这些行看不出上游；上游靠这里的核对。）",
             f"- 大小 {s['total_bytes'] / 1024**3:.3f} GiB（每帧 {s['bytes_per_state']:.0f} B）；用时 {s['elapsed_seconds']:.0f} s（{s['workers']} 个进程）。",
             f"- 用途：{s['training_use']}。",
             "- 确认集（random_branches_*、角色 confirm）永不转换；方案以外的数据集（试采）只能加 --check-only、转到数据根目录之外作检查，index 标 check_only。"]
    (Path(output) / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def atomic_progress(output, value):
    path = Path(output) / "progress.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(native(value), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main(argv=None):
    a = build_parser().parse_args(argv)
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    ids = None if a.anchor_ids is None else sorted({int(t) for t in a.anchor_ids.split(",") if t.strip()})
    manifest, manifest_sha = load_dataset(a.dataset, a.report)
    role = dataset_role(manifest, a.dataset)
    if role != a.role:
        raise ValueError(f"{a.dataset} was collected with role {role!r}; convert it with --role {role}")
    plan_rule = plan_dataset_rule(a.dataset.resolve(), output, a.role, a.stage2_plan, ids, a.check_only, a.reference_cache.resolve().parent)
    recorded_plan = (manifest.get("stage2_plan") or {}).get("sha256")
    if recorded_plan is not None and (plan_rule["stage2_plan"] is None or recorded_plan != plan_rule["stage2_plan"]["sha256"]):
        raise ValueError(f"{a.dataset} was collected from stage-2 plan {recorded_plan}; pass that frozen plan with --stage2-plan")
    for protected in (a.dataset, a.reference_cache, a.bench_cache, a.holdout_cache, a.report):
        protected = Path(protected).resolve()
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(f"output must be separate from its inputs: {protected}")
    ref_index, ref_sha = read_json(a.reference_cache / "index.json"), sha256(a.reference_cache / "index.json")
    if ref_index.get("status") != "complete" or ref_index.get("eval_cache"):
        raise ValueError("--reference-cache must be a complete TRAINING cache")
    bench_index, holdout_index = read_json(a.bench_cache / "index.json"), read_json(a.holdout_cache / "index.json")
    classifier, bench_split = read_json(a.classifier_run), read_json(a.bench_split)
    if bench_index.get("split_sha256") != sha256(a.bench_split):
        raise ValueError("--bench-split is not the split file of --bench-cache")
    norm = manifest["wm_action"]["normalization"]
    if list(norm["mean"]) != list(ref_index["normalization"]["mean"]) or list(norm["scale"]) != list(ref_index["normalization"]["scale"]):
        raise ValueError("the dataset's wm_action normalization differs from the reference cache")
    rows_all = sorted(manifest["anchors"], key=lambda r: int(r["id"]))
    if ids is not None:
        known = {int(r["id"]) for r in rows_all}
        if not ids or not set(ids) <= known:
            raise ValueError(f"unknown anchor ids {sorted(set(ids) - known)[:10]}")
        rows_all = [r for r in rows_all if int(r["id"]) in set(ids)]
    roots, starts = check_sources({**manifest, "anchors": rows_all}, a.role, ref_index, bench_index, holdout_index, classifier, bench_split)
    split_used = str(Path(a.bench_split).resolve())          # plan.json conversion.index: listed whether or not a bench start was converted
    episodes_planned = sum(len(r["rollouts"]) - 1 for r in rows_all)
    need = int(episodes_planned * STATES * BYTES_PER_STATE * 1.2) + DISK_RESERVE_BYTES
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < need:
        raise OSError(f"not enough free disk space ({need / 1024**3:.1f} GiB needed)")
    from wm_training.provenance import implementation_hashes
    root = Path(__file__).resolve().parent
    inputs = [a.dataset / "manifest.json", a.report / "summary.json", a.reference_cache / "index.json", a.bench_cache / "index.json",
              a.holdout_cache / "index.json", a.classifier_run, a.bench_split] + ([a.stage2_plan] if a.stage2_plan else [])
    output.mkdir(parents=True)
    (output / "episodes").mkdir()
    run = {"format": FORMAT, "role": a.role, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "input_sha256": {str(p): sha256(p) for p in inputs}, "plan_rule": plan_rule, "starts": starts, "upstream_source_roots": roots,
           "implementation_sha256": {**implementation_hashes(IMPLEMENTATION_ROOT, package_dir=root), **implementation_hashes(IMPLEMENTATION_WM)},
           "python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__,
           "anchors": [int(r["id"]) for r in rows_all], "episodes_planned": episodes_planned, "world_model_loaded": False,
           "test_episodes_opened": False, "confirmation_set_opened": False, "check_only": a.check_only}
    write_new_json(output / "run.json", run)
    started = time.perf_counter()
    try:
        jobs = [{"dataset": str(a.dataset.resolve()), "output": str(output), "row": r, "image_size": int(ref_index["image_size"]),
                 "mean": ref_index["normalization"]["mean"], "scale": ref_index["normalization"]["scale"], "role": a.role,
                 "check_only": a.check_only} for r in rows_all]
        rows, stats = [], []

        def record(result):
            rows.extend(result[0])
            stats.append(result[1])
            atomic_progress(output, {"status": "converting", "anchors_done": len(stats), "anchors": len(jobs), "episodes_done": len(rows),
                                     "elapsed_seconds": time.perf_counter() - started})
        if a.workers == 1:
            for job in jobs:
                record(convert_anchor(job))
        else:
            with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for result in pool.map(convert_anchor, jobs, chunksize=1):
                    record(result)
        rows.sort(key=lambda r: r["id"])
        if len({r["id"] for r in rows}) != len(rows) or len(rows) != episodes_planned:
            raise RuntimeError("episode ids are not unique or episodes are missing")
        index = build_index(a, manifest, manifest_sha, rows, ref_index, ref_sha, roots, split_used, plan_rule, starts)
        write_new_json(output / "index.json", index)
        audit = upstream_audit(read_json(output / "index.json"), a.role, ref_index, bench_index, holdout_index, classifier)
        rows_by_id = {int(r["id"]): r for r in rows_all}
        rng = np.random.default_rng(len(rows))
        spot = sorted({rows[0]["id"], rows[-1]["id"], *[rows[i]["id"] for i in rng.choice(len(rows), min(len(rows), ALIGNMENT_SPOT_EPISODES), replace=False)]})
        verification = verify_cache(output, a.role, a.reference_cache, a.bench_cache, a.holdout_cache, a.dataset.resolve(), rows_by_id, spot)
        verification["upstream_raw_audit"] = audit
        if a.stage2_plan is not None and sha256(a.stage2_plan) != plan_rule["stage2_plan"]["sha256"]:
            raise RuntimeError("the stage-2 plan changed during the conversion")
        elapsed = time.perf_counter() - started
        linked_non_g = [s for s in stats if s["goal_source"] == "goal_image_achieved_com" and s["label_goal_source"] == "hover_goal"]
        relabelled_g = [s for s in stats if s["label_goal_source"] == "goal_image_achieved_com"]
        summary = {"format": FORMAT, "status": "complete", "role": a.role, "dataset": str(a.dataset.resolve()), "dataset_manifest_sha256": manifest_sha,
                   "dataset_role": manifest.get("role", "train"), "output": str(output), "partial": ids is not None, "check_only": a.check_only,
                   "training_use": index["training_use"], "anchors": len(rows_all),
                   "episodes": len(rows), "source_part": ROLE_PART[a.role], "starts": starts, "upstream_source_roots": roots,
                   "goal_sources": dict(Counter(s["goal_source"] for s in stats)),
                   "label_goal_sources": dict(Counter(s["label_goal_source"] for s in stats)),
                   "linked_goal_image_non_g_anchors": {"anchors": len(linked_non_g), "by_stratum": dict(Counter(s["stratum"] for s in linked_non_g)),
                                                       "rule": "goal_source goal_image_achieved_com but stratum T / F / M: labelled against the hover goal "
                                                               "(recorded labels reproduced bit for bit), as plan.json '(G)' / PLAN_ZH.md say",
                                                       "goal_offset_m_max": max((s["label_goal_offset_m"] for s in linked_non_g), default=None)},
                   "relabelled_g_anchors": {"anchors": len(relabelled_g), "goal_offset_m_max": max((s["label_goal_offset_m"] for s in relabelled_g), default=None)},
                   "relabelled_frames": int(sum(s["relabelled_frames"] for s in stats)),
                   "relabelled_episodes": int(sum(s["relabelled_episodes"] for s in stats)),
                   "class_counts": {k: int(sum(r["class_counts"][k] for r in rows)) for k in ("0", "1", "2")},
                   "by_family": dict(Counter(r["mode"] for r in rows)), "by_stratum": dict(Counter(r["stratum"] for r in rows)),
                   "total_bytes": index["total_bytes"], "bytes_per_state": index["total_bytes"] / max(1, len(rows) * STATES),
                   "index_sha256": sha256(output / "index.json"), "verification": verification, "plan_rule": plan_rule,
                   "elapsed_seconds": elapsed, "workers": a.workers, "world_model_loaded": False}
        write_report(output, summary)
        atomic_progress(output, {"status": "complete", "anchors_done": len(stats), "anchors": len(jobs), "episodes_done": len(rows),
                                 "elapsed_seconds": elapsed})
        write_new_json(output / "summary.json", native(summary))
        print(json.dumps(native({k: summary[k] for k in ("role", "anchors", "episodes", "relabelled_frames", "total_bytes")}), ensure_ascii=False), flush=True)
        return summary
    except BaseException as error:
        if (output / "index.json").exists():
            os.replace(output / "index.json", output / "index.failed.json")       # a failed cache must not be usable
        write_new_json(output / "FAILED.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
