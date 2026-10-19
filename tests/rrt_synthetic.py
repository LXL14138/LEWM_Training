"""Synthetic rrt_branches_v1 datasets (numpy only) for the converter and confirmation-evaluator tests.

Every array has the collector's layout (collect_rrt_branches.py: R rollouts x 33 states, 32 actions, prefix frames a-10..a
with their per-state diagnostics); the labels follow the scene's rules (tilt from the guard quaternions, unsafe > 30 deg,
class 0 = grasped & |green - hover goal| <= 1.5 cm & speed <= 2 cm/s), so every consistency check of the tools can pass
or be made to fail on purpose. No real dataset, cache or benchmark episode is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

H, S = 32, 33
PREFIX = 10
NORM = {"names": ["dx", "dy", "dz", "dgripper"], "fit_split": "train", "mean": [0.01, -0.02, 0.03, 0.3],
        "std": [0.05, 0.05, 0.08, 0.9], "scale": [0.05, 0.05, 0.08, 0.9]}
SEGMENTS = {"hold": (["hold"], [32]), "edge_hold": (["edge", "hold"], [8, 24]), "planner": (["edge", "edge", "edge", "hold"], [8, 8, 8, 8]),
            "guard": (["edge", "edge", "edge", "hold"], [8, 8, 8, 8]), "push": (["edge", "edge", "edge", "hold"], [8, 8, 8, 8]),
            "push4": (["edge", "edge", "edge", "edge"], [8, 8, 8, 8]), "push_hold": (["edge", "hold"], [8, 24]),
            "push2_hold": (["edge", "edge", "hold"], [8, 8, 16])}
STEP_FIELDS = ("guard_tilt_deg", "unsafe", "class_label", "hover_error_m", "target_speed_mps", "hover_conditions_met", "grasped",
               "object_com_xyz", "object_quat_wxyz", "pinch_xyz", "guard_contact_force_n", "guard_contact_peak_last_step_n",
               "green_guard_contact_peak_last_step_n", "physics_state")
CONTACT_FIELDS = ("robot_env_contact", "robot_green_contact", "green_env_contact")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(actions5):
    a = np.asarray(actions5, np.float32)
    mean, scale = np.asarray(NORM["mean"], np.float32), np.asarray(NORM["scale"], np.float32)
    return ((a[..., [0, 1, 2, 4]] - mean) / scale).astype(np.float32)


def quats(tilt):
    """(n, 2) guard tilts -> (n, 3, 4) wxyz quaternions (green upright, guards rotated about x)."""
    n = len(tilt)
    q = np.zeros((n, 3, 4))
    q[:, :, 0] = 1.0
    for g in (0, 1):
        half = np.deg2rad(tilt[:, g]) / 2
        q[:, 1 + g, 0], q[:, 1 + g, 1] = np.cos(half), np.sin(half)
    return q


def tilt_from_quats(q):
    """The audit's tilt formula (safety_classifier.audit_labels) so stored tilt and quaternions agree to rounding."""
    g = q[:, 1:]
    return np.rad2deg(np.arccos(np.clip(1 - 2 * (g[..., 1] ** 2 + g[..., 2] ** 2), -1, 1)))


def states(tilt_worst, grasped, green, speed, hover_goal, guard=0):
    """Per-state STEP_FIELDS + contacts for a worst-tilt trace (applied to guard ``guard``)."""
    n = len(tilt_worst)
    tilt = np.zeros((n, 2))
    tilt[:, guard] = tilt_worst
    q = quats(tilt)
    tilt = tilt_from_quats(q)
    com = np.zeros((n, 3, 3))
    com[:, 0], com[:, 1], com[:, 2] = green, [0.425, -0.05, 0.05], [0.425, 0.05, 0.05]
    error = np.array([float(np.linalg.norm(c - hover_goal)) for c in com[:, 0]])
    cond = np.asarray(grasped, bool) & (error <= 0.015 + 1e-9) & (np.asarray(speed) <= 0.02 + 1e-9)
    unsafe = tilt.max(1) > 30 + 1e-7
    cls = np.where(unsafe, 1, np.where(cond, 0, 2)).astype(np.int8)
    out = {"guard_tilt_deg": tilt, "unsafe": unsafe, "class_label": cls, "hover_error_m": error, "target_speed_mps": np.asarray(speed, float),
           "hover_conditions_met": cond, "grasped": np.asarray(grasped, bool), "object_com_xyz": com, "object_quat_wxyz": q,
           "pinch_xyz": green + [0, 0, 0.03], "guard_contact_force_n": np.zeros((n, 2)), "guard_contact_peak_last_step_n": np.zeros((n, 2)),
           "green_guard_contact_peak_last_step_n": np.zeros((n, 2)), "physics_state": np.zeros((n, 16))}
    out.update({k: np.zeros(n, bool) for k in CONTACT_FIELDS})
    return out


def trace(kind, anchor_tilt, n=S):
    """Worst-tilt trace of one rollout: calm / rise (topples within 20 steps) / late (topples after step 12) / lean."""
    t = np.full(n, float(anchor_tilt))
    if kind == "rise":
        t[1:] = np.minimum(anchor_tilt + 4.0 * np.arange(1, n), 80.0)
    elif kind == "late":
        t[13:] = np.minimum(anchor_tilt + 5.0 * np.arange(1, n - 12), 80.0)
    elif kind == "lean":
        t[1:] = anchor_tilt + 12.0
    return t


def write_anchor(folder, spec, image_size, seed):
    """One anchor_XXXX.npz. spec: id, set, stratum, families, kinds (per rollout; kinds[0] = the recorded one), anchor_tilt,
    grasped, goal_source, goal_xyz, hover_goal_xyz, source_frame, cache_episode_id, cache, grip."""
    rng = np.random.default_rng(seed)
    fams = list(spec["families"])
    R = len(fams)
    hover = np.asarray(spec["hover_goal_xyz"], float)
    goal = np.asarray(spec.get("goal_xyz", hover), float)
    start = hover + np.array([0.012, 0.0, 0.0])                                    # 1.2 cm from the hover goal
    prefix_green = np.stack([start + [0.001 * (PREFIX - j), 0, 0] for j in range(PREFIX + 1)])
    anchor_tilt = float(spec["anchor_tilt"])
    prefix_tilt = np.full(PREFIX + 1, anchor_tilt)
    grasped = bool(spec.get("grasped", True))
    pre = states(prefix_tilt, np.full(PREFIX + 1, grasped), prefix_green, np.full(PREFIX + 1, 0.01), hover)
    per = []
    for r in range(R):
        green = np.stack([start + [0.0005 * k * (1 if r % 2 else -1), 0.0003 * k, 0] for k in range(S)])
        speed = np.full(S, 0.01 if r % 3 else 0.03)
        speed[0] = 0.01
        g = np.full(S, grasped)
        per.append(states(trace(spec["kinds"][r], anchor_tilt), g, green, speed, hover))
    arrays = {k: np.stack([p[k] for p in per]) for k in per[0]}
    for k in arrays:                                                              # rollout state 0 = the anchor state = prefix index 10
        arrays[k][:, 0] = pre[k][PREFIX]
    prefix_pixels = rng.integers(0, 255, (PREFIX + 1, image_size, image_size, 3), dtype=np.uint8)
    pixels = rng.integers(0, 255, (R, S, image_size, image_size, 3), dtype=np.uint8)
    pixels[:, 0] = prefix_pixels[PREFIX]
    grip = float(spec.get("grip", 1.0))
    actions = rng.uniform(-0.3, 0.3, (R, H, 5))
    actions[:, :, 3], actions[:, :, 4] = 0.0, grip
    prefix_actions = rng.uniform(-0.3, 0.3, (PREFIX, 5))
    prefix_actions[:, 3], prefix_actions[:, 4] = 0.0, grip
    seg_kind = np.full((R, 4), "", dtype="U4")
    seg_index = np.full((R, H), -1, np.int64)
    for r, f in enumerate(fams):
        if f == "recorded":
            continue
        kinds, lens = SEGMENTS[f]
        seg_kind[r, :len(kinds)] = kinds
        seg_index[r] = np.repeat(np.arange(len(kinds)), lens)
    arrays.update({f"prefix_{k}": v for k, v in pre.items()})
    arrays.update(pixels=pixels, prefix_pixels=prefix_pixels, actions=actions, wm_action=normalize(actions), prefix_actions=prefix_actions,
                  prefix_wm_action=normalize(prefix_actions), segment_kind=seg_kind, segment_index=seg_index, families=np.array(fams),
                  anchor_id=np.int64(spec["id"]), stratum=np.array(spec["stratum"]), set=np.array(spec["set"]),
                  group=np.array(f"{spec['set']}/{spec['stratum']}"), source_guard_tilt_deg=arrays["guard_tilt_deg"][0].copy(),
                  goal_xyz=goal, goal_source=np.array(spec.get("goal_source", "hover_goal")), goal_image_id=np.int64(spec.get("goal_image_id", -1)),
                  hover_goal_xyz=hover, source_frame=np.int64(spec["source_frame"]), cache_episode_id=np.int64(spec["cache_episode_id"]),
                  cache=np.array(spec.get("cache", "wm_full_001")), grip_command=np.float64(grip))
    path = Path(folder) / f"anchor_{int(spec['id']):04d}.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return path, arrays


def write_dataset(root, name, specs, image_size, role=None, stage2_plan=None, report_name=None, decision_points=None):
    """Dataset directory + collector report directory; returns (dataset, report, manifest)."""
    dataset, report = Path(root) / name, Path(root) / (report_name or f"{name}_report")
    dataset.mkdir(parents=True)
    report.mkdir(parents=True)
    rows = []
    for i, spec in enumerate(specs):
        path, arrays = write_anchor(dataset, spec, image_size, 1000 + i)
        plan_anchor = {"id": spec["id"], "set": spec["set"], "episode_id": spec["cache_episode_id"], "stratum": spec["stratum"],
                       "frame": spec["source_frame"], "seed": 1000 + i, "grip_command": float(spec.get("grip", 1.0)),
                       "family_set": spec.get("family_set", "A"), "families": list(spec["families"]),
                       "source": {"source": spec["set"], "episode_id": spec["cache_episode_id"], "cache_episode_id": spec["cache_episode_id"],
                                  "episode_dir": spec["episode_dir"], "npz_sha256": spec["npz_sha256"], "metadata_sha256": "m" * 64,
                                  "cache_h5": "unused", "frames": 400}}
        row = {"id": spec["id"], "set": spec["set"], "group": f"{spec['set']}/{spec['stratum']}", "stratum": spec["stratum"],
               "cache_episode_id": spec["cache_episode_id"], "frame": spec["source_frame"], "file": path.name, "sha256": sha(path),
               "plan_anchor": plan_anchor, "rollouts": [{"rollout": r, "family": f} for r, f in enumerate(spec["families"])],
               "sanity": {"prefix_replay_exact": True, "recorded_exact": True}}
        if decision_points is not None:
            row["decision_points"] = decision_points(arrays)
        rows.append(row)
    manifest = {"format": "rrt_branches_v1", "status": "complete", "prefix": PREFIX, "horizon": H,
                "wm_action": {"columns": [0, 1, 2, 4], "normalization": {k: NORM[k] for k in ("names", "mean", "scale")}}, "anchors": rows}
    if role is not None:
        manifest["role"] = role
    if stage2_plan is not None:
        manifest["stage2_plan"] = stage2_plan
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (report / "summary.json").write_text(json.dumps({"format": "rrt_branches_v1", "manifest_sha256": sha(dataset / "manifest.json")}), encoding="utf-8")
    return dataset, report, manifest
