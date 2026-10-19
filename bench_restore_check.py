"""SIMULATION ONLY: restore a recorded benchmark frame, then check determinism / point of no return.

Two read-only studies on the danger benchmark (WP4 preparation in HANDOFF.md section 5):

* ``determinism``: restore the simulator from a recorded mid-episode ``physics_state``
  (``mjSTATE_INTEGRATION`` = 16383), replay the recorded normalized actions across the
  first guard-tilt onset and MEASURE (never assert) how far qpos, guard tilt, object COM,
  the ``unsafe`` flag and re-rendered ``front_pixels`` images drift from the record.
* ``no_return``: restore k steps before the onset, stop the robot (anchored hold) or
  retreat 4 cm/s away from the tilting guard, and record whether the guard still exceeds
  30 degrees. Every rollout's action sequence and per-step tilts go to ``rollouts.npz`` so a
  world model can later be scored on exactly these counterfactual action sequences.

Restore recipe (HANDOFF.md section 6): ``env.reset()`` first (the TaskProgress tracker
refuses time going backwards), set the recorded hover goal while ``env.data.time == 0``,
``mujoco.mj_setState(model, data, physics_state, 16383)``, ``mj_forward``, ``env.post_step()``.
The hover timer and the ``episode_failed`` latch are not part of ``physics_state``; the
labels used here depend only on the current tilt.

Hold semantics (measured 2026-09-24, see BENCH_RESTORE.md): a normalized zero action is
NOT a physical hold. ``set_control`` re-anchors the target on the current pinch every
step, so the PD arm sags about 1.3 mm per control step (13 mm in 10 steps, 5-7 cm over a
3 s hold). ``hold_action`` therefore exists only as the open-loop zero action the spec
asked for; the rollouts use ``AnchoredHold``, a closed-loop PI hold around the pinch
position at the restore frame (max 2 mm error, converging to 0), which is the same
"track a fixed reference" law the recorded controllers use in their EVENT_HOLD phase.

Datasets are never modified; only the listed episode directories are opened. Output goes
to a NEW directory: ``summary.json`` only on success, ``FAILED.json`` on error.
"""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

import mujoco
import numpy as np
from PIL import Image

from exploration import ExplorationScene
from wm_training.danger_strata import body_up, native, read_json, sha256, write_new_json

FORMAT = "bench_restore_check_v1"
STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)          # 16383: time, qpos, qvel, act, warmstart, ctrl, ...
MODES = ("determinism", "no_return", "both")
VARIANTS = ("hold", "retreat")
GRIP_POLICIES = ("keep", "zero")
ORDERS = ("seeded", "listed")
DEFAULT_LEADS = tuple(range(1, 21))
DEFAULT_EVENTS = {"determinism": 20, "no_return": 60}
DETERMINISM_BEFORE, DETERMINISM_BEFORE_SHORT, DETERMINISM_AFTER = 10, 5, 5
HOLD_STEPS, RETREAT_STEPS, RETREAT_SPEED_MPS = 60, 20, 0.04     # 3 s hold; retreat 1 s at 4 cm/s then hold
PIXEL_LEVELS = 2                                                # "differs" = any channel off by more than this
AVOID_THRESHOLDS = (3, 5, 10, 20)
SEVERITIES = ("S1_flat", "S2_lean", "S3_just_over")
RENDER_KEYS = ("MUJOCO_GL", "GALLIUM_DRIVER", "MESA_D3D12_DEFAULT_ADAPTER_NAME", "LIBGL_ALWAYS_SOFTWARE", "LD_LIBRARY_PATH")
EPISODE_FIELDS = ("time", "qpos", "qvel", "physics_state", "actions", "guard_tilt_deg", "object_com_xyz",
                  "object_quat_wxyz", "pinch_xyz", "unsafe", "hover_goal_xyz")
ACTION_XYZ_LIMIT_M = 0.05                                       # |physical dx,dy,dz| bound of the scene's action space
HOLD_GAIN_P, HOLD_GAIN_I = 1.0, 1.0


# ----- split and episodes -------------------------------------------------------------------
def resolve_dataset_dir(recorded, split_path, data_root=None):
    """The split stores WSL paths; fall back to <data_root or the split's parent dir>/<basename>."""
    recorded = Path(recorded)
    candidates = [recorded]
    if data_root is not None:
        candidates.insert(0, Path(data_root) / recorded.name)
    candidates.append(Path(split_path).resolve().parent.parent / recorded.name)
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Dataset directory not found for {recorded}; tried {[str(c) for c in candidates]}")


def load_split(path, data_root=None):
    """Datasets (resolved), and every bench_test EVENT (episode with an onset) in split-file order."""
    path = Path(path)
    split = read_json(path)
    if split.get("format") != "danger_bench_split_v1":
        raise ValueError(f"{path}: not a danger_bench_split_v1 file")
    datasets = []
    for row in split["datasets"]:
        folder = resolve_dataset_dir(row["path"], path, data_root)
        manifest = read_json(folder / "manifest.json")
        if sha256(folder / "manifest.json") != row["manifest_sha256"]:
            raise ValueError(f"{folder}: manifest.json changed since the split was written")
        datasets.append({"index": int(row["index"]), "path": str(folder), "recorded_path": row["path"],
                         "manifest_sha256": row["manifest_sha256"],
                         "render_environment": manifest.get("execution", {}).get("render_environment"),
                         "versions": manifest.get("versions")})
    by_index = {d["index"]: d for d in datasets}
    test = {(int(e["dataset"]), int(e["episode_id"])) for part in split["strata"].values() for e in part["test"]}
    dev = {(int(e["dataset"]), int(e["episode_id"])) for part in split["strata"].values() for e in part["dev"]}
    if test & dev:
        raise ValueError("bench-dev and bench-test overlap")
    events = []
    for entry in split["episodes"]:
        key = (int(entry["dataset"]), int(entry["episode_id"]))
        if key not in test or entry.get("onset_frame") is None:
            continue
        events.append({"dataset": key[0], "episode_id": key[1], "path": entry["path"], "stratum": str(entry["stratum"]),
                       "onset_frame": int(entry["onset_frame"]), "frames": int(entry["frames"]),
                       "episode_dir": str(Path(by_index[key[0]]["path"]) / entry["path"])})
    if not events:
        raise ValueError("The split lists no bench_test events with an onset frame")
    return {"path": str(path.resolve()), "sha256": sha256(path), "seed": split.get("seed"), "datasets": datasets,
            "events": events, "test_episodes": len(test), "dev_episodes": len(dev)}


def default_strata(events):
    """The direction x severity cells (transient / negative strata are excluded unless asked for)."""
    return sorted({e["stratum"] for e in events
                   if "|" in e["stratum"] and e["stratum"].split("|", 1)[1] in SEVERITIES})


def select_events(events, n, seed, strata=None, order="seeded"):
    """Round-robin over strata (seeded shuffle inside each) so a small sample is spread over cells.

    ``strata`` are name prefixes; None means the 12 direction x severity cells. ``order="listed"``
    takes the first n matching events in split-file order instead (used by the tests).
    """
    if type(n) is not int or n <= 0 or order not in ORDERS:
        raise ValueError("n must be a positive integer and order one of " + ", ".join(ORDERS))
    if strata is None:
        names = default_strata(events)
        pool = [e for e in events if e["stratum"] in names]
    else:
        prefixes = [str(s) for s in strata if str(s)]
        if not prefixes:
            raise ValueError("Empty strata filter")
        pool = [e for e in events if any(e["stratum"].startswith(p) for p in prefixes)]
    if not pool:
        raise ValueError("No bench_test events match the requested strata")
    if order == "listed":
        return pool[:n]
    rng = np.random.default_rng(seed)
    groups = {}
    for e in pool:
        groups.setdefault(e["stratum"], []).append(e)
    names = sorted(groups)
    queues = {name: [groups[name][i] for i in rng.permutation(len(groups[name]))] for name in names}
    cycle = [names[i] for i in rng.permutation(len(names))]      # seeded stratum order: small samples still spread
    chosen = []
    while len(chosen) < n and any(queues.values()):
        for name in cycle:
            if queues[name] and len(chosen) < n:
                chosen.append(queues[name].pop(0))
    return chosen


def load_episode(episode_dir, expected=None):
    """metadata + the restore/replay arrays of one recorded episode, strictly checked."""
    episode_dir = Path(episode_dir)
    meta = read_json(episode_dir / "metadata.json")
    if meta.get("status") != "complete":
        raise ValueError(f"{episode_dir}: episode is not complete")
    if int(meta.get("physics_state_spec", -1)) != STATE_SPEC:
        raise ValueError(f"{episode_dir}: physics_state_spec {meta.get('physics_state_spec')} != {STATE_SPEC}")
    with np.load(episode_dir / "trajectory.npz", allow_pickle=False) as npz:
        missing = [k for k in EPISODE_FIELDS if k not in npz.files]
        if missing:
            raise ValueError(f"{episode_dir}: trajectory.npz lacks {missing}")
        arrays = {k: np.asarray(npz[k]) for k in EPISODE_FIELDS}
    n = len(arrays["time"])
    shapes = {"time": (n,), "physics_state": (n, arrays["physics_state"].shape[-1]), "actions": (n - 1, 5),
              "guard_tilt_deg": (n, 2), "object_com_xyz": (n, 3, 3), "object_quat_wxyz": (n, 3, 4),
              "pinch_xyz": (n, 3), "unsafe": (n,), "hover_goal_xyz": (n, 3),
              "qpos": (n, arrays["qpos"].shape[-1]), "qvel": (n, arrays["qvel"].shape[-1])}
    if n < 2 or int(meta["num_states"]) != n or arrays["physics_state"].ndim != 2:
        raise ValueError(f"{episode_dir}: state count disagrees with metadata")
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{episode_dir}: {key} has shape {arrays[key].shape}, expected {shape}")
    for key in EPISODE_FIELDS:
        if key == "unsafe":
            if arrays[key].dtype != np.dtype(bool):
                raise ValueError(f"{episode_dir}: unsafe must be boolean")
        elif arrays[key].dtype.kind != "f" or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{episode_dir}: {key} must be finite float")
    dt = float(meta["control_dt"])
    if not np.allclose(np.diff(arrays["time"]), dt, atol=1e-7, rtol=0):
        raise ValueError(f"{episode_dir}: timestamps disagree with control_dt")
    if np.any(np.abs(arrays["actions"]) > 1) or np.any(arrays["actions"][:, 3] != 0):
        raise ValueError(f"{episode_dir}: recorded actions violate the normalized bounds or the fixed yaw")
    goal = np.asarray(meta["hover_goal_xyz"], dtype=float)
    if goal.shape != (3,) or not np.allclose(arrays["hover_goal_xyz"], goal, atol=1e-12, rtol=0):
        raise ValueError(f"{episode_dir}: hover goal in metadata and trajectory disagree")
    onsets = np.flatnonzero(arrays["unsafe"])
    onset = int(onsets[0]) if len(onsets) else None
    if expected is not None:
        if int(expected["frames"]) != n or int(expected["onset_frame"]) != onset:
            raise ValueError(f"{episode_dir}: frames/onset ({n}/{onset}) disagree with the split entry "
                             f"({expected['frames']}/{expected['onset_frame']})")
    return {"dir": str(episode_dir), "meta": meta, "control_dt": dt, "hover_goal": goal, "frames": n,
            "onset_frame": onset, "camera": meta["image"]["camera"], **arrays}


# ----- restore --------------------------------------------------------------------------------
def restore_state(env, physics_state, hover_goal=None):
    """reset -> hover goal at time 0 -> mj_setState(16383) -> mj_forward -> post_step. Returns env."""
    physics_state = np.asarray(physics_state, dtype=np.float64)
    if physics_state.ndim != 1 or not np.isfinite(physics_state).all():
        raise ValueError("physics_state must be a finite 1-D vector")
    env.reset(seed=0)
    expected = mujoco.mj_stateSize(env.model, STATE_SPEC)
    if physics_state.shape != (expected,):
        raise ValueError(f"physics_state has {physics_state.shape[0]} values; this model needs {expected}")
    if hover_goal is not None:
        if not hasattr(env, "set_hover_goal"):
            raise TypeError("A hover goal needs an ExplorationScene (set_hover_goal)")
        env.set_hover_goal(hover_goal)          # only legal while env.data.time == 0
    mujoco.mj_setState(env.model, env.data, physics_state, STATE_SPEC)
    mujoco.mj_forward(env.model, env.data)
    env.post_step()
    return env


def guard_tilts(env):
    return np.array([g["tilt_deg"] for g in env.safety_report()["guards"]], dtype=float)


def pinch_xyz(env):
    return env.data.site_xpos[env._pinch_site_id].copy()


def restore(env, episode_dir, frame, loaded=None):
    """Put ``env`` into recorded frame ``frame`` of ``episode_dir``; the env is then ready to step.

    ``loaded`` (from load_episode) avoids re-reading the npz for every lead. The restored
    qpos/qvel/time/tilt/unsafe must equal the recorded values of that frame exactly (they
    were captured in the same instant as physics_state); anything else is a model mismatch.
    """
    loaded = load_episode(episode_dir) if loaded is None else loaded
    frame = int(frame)
    if not 0 <= frame < loaded["frames"]:
        raise ValueError(f"frame {frame} outside 0..{loaded['frames'] - 1}")
    if abs(float(env.control_timestep()) - loaded["control_dt"]) > 1e-12:
        raise ValueError("control_dt of the scene and of the recording disagree")
    restore_state(env, loaded["physics_state"][frame], loaded["hover_goal"])
    if not np.array_equal(env.data.qpos, loaded["qpos"][frame]) or not np.array_equal(env.data.qvel, loaded["qvel"][frame]):
        raise RuntimeError(f"{loaded['dir']} frame {frame}: restored qpos/qvel differ from the recorded arrays")
    if abs(float(env.data.time) - float(loaded["time"][frame])) > 1e-9:
        raise RuntimeError(f"{loaded['dir']} frame {frame}: restored time differs from the record")
    safety = env.safety_report()
    if (not np.allclose(guard_tilts(env), loaded["guard_tilt_deg"][frame], atol=1e-9, rtol=0)
            or bool(safety["unsafe"]) != bool(loaded["unsafe"][frame])):
        raise RuntimeError(f"{loaded['dir']} frame {frame}: restored guard tilt / unsafe differ from the record")
    return env


# ----- actions --------------------------------------------------------------------------------
def hold_action(env):
    """Normalized action with zero XYZ motion and zero gripper change (the spec's definition).

    WARNING: this is an OPEN-LOOP zero. set_control re-anchors the target on the current pinch,
    so under gravity the arm sags about 1.3 mm per control step (measured). Use AnchoredHold to
    actually keep the pinch in place.
    """
    action = np.asarray(env.normalize_action(np.zeros(5)), dtype=float)
    if action.shape != (5,) or not np.isfinite(action).all() or action[3] != 0 or np.any(np.abs(action) > 1e-12):
        raise RuntimeError(f"normalize_action(0) is not the zero action: {action}")
    return action


def normalized_xyz_action(env, delta_xyz, grip):
    """Normalize a physical [dx, dy, dz] (clipped to the action range) with yaw 0 and gripper ``grip``."""
    delta = np.clip(np.asarray(delta_xyz, dtype=float), -ACTION_XYZ_LIMIT_M, ACTION_XYZ_LIMIT_M)
    if delta.shape != (3,) or not np.isfinite(delta).all() or not np.isfinite(grip) or abs(grip) > 1:
        raise ValueError("delta_xyz must be three finite metres and grip a finite value in [-1, 1]")
    action = np.asarray(env.normalize_action(np.array([*delta, 0.0, 0.0])), dtype=float)
    action[3] = 0.0
    action[4] = float(grip)
    if not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise RuntimeError("Invalid normalized action")
    return action


class AnchoredHold:
    """Closed-loop hold / slow retreat: PI tracking of a reference in pinch coordinates.

    action_k = P * e_k + I * sum_{j<=k} e_j with e_k = reference_k - pinch_k (metres, clipped to
    the +-5 cm action range; the integral is clipped too). With P = I = 1 the pinch stays within
    2 mm of a fixed anchor and converges to it (measured). ``velocity_xy`` moves the reference
    horizontally for the first ``move_steps`` steps (retreat), then the reference stays fixed.
    ``grip`` is the constant normalized gripper command (+1 close, -1 open, 0 = no change).
    """

    def __init__(self, anchor_xyz, grip=0.0, velocity_xy=None, move_steps=0, dt=0.05,
                 gain_p=HOLD_GAIN_P, gain_i=HOLD_GAIN_I):
        self.anchor = np.asarray(anchor_xyz, dtype=float).copy()
        if self.anchor.shape != (3,) or not np.isfinite(self.anchor).all():
            raise ValueError("anchor_xyz must be three finite coordinates")
        if type(move_steps) is not int or move_steps < 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("move_steps must be a non-negative integer and dt positive")
        self.velocity = np.zeros(2) if velocity_xy is None else np.asarray(velocity_xy, dtype=float)
        if self.velocity.shape != (2,) or not np.isfinite(self.velocity).all():
            raise ValueError("velocity_xy must be two finite m/s values")
        if np.linalg.norm(self.velocity) * dt > ACTION_XYZ_LIMIT_M:
            raise ValueError("retreat speed exceeds the per-step action range")
        self.grip, self.move_steps, self.dt = float(grip), move_steps, float(dt)
        self.gain_p, self.gain_i = float(gain_p), float(gain_i)
        self.step_index, self.integral = 0, np.zeros(3)

    def reference(self, step_index):
        moved = min(step_index + 1, self.move_steps)
        ref = self.anchor.copy()
        ref[:2] += self.velocity * self.dt * moved
        return ref

    def action(self, env):
        error = self.reference(self.step_index) - pinch_xyz(env)
        self.integral = np.clip(self.integral + error, -ACTION_XYZ_LIMIT_M, ACTION_XYZ_LIMIT_M)
        self.step_index += 1
        return normalized_xyz_action(env, self.gain_p * error + self.gain_i * self.integral, self.grip)


def retreat_direction(pinch, guard_com, guard_quat_wxyz=None):
    """Horizontal unit vector pointing from the guard to the pinch (moving along it retreats).

    If the pinch is (almost) above the guard, retreat against the guard's lean; if the guard is
    upright too, retreat toward the robot base (-X, away from the camera).
    """
    away = np.asarray(pinch, dtype=float)[:2] - np.asarray(guard_com, dtype=float)[:2]
    if np.linalg.norm(away) < 0.005 and guard_quat_wxyz is not None:
        away = -body_up(guard_quat_wxyz)[:2]
    if np.linalg.norm(away) < 1e-6:
        away = np.array([-1.0, 0.0])
    return away / np.linalg.norm(away)


def rollout(env, controller, steps):
    """Step ``controller.action(env)`` ``steps`` times; record actions, guard tilts, unsafe and pinch."""
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    dt = float(env.control_timestep())
    tilt, unsafe, pinch, actions = [guard_tilts(env)], [bool(env.safety_report()["unsafe"])], [pinch_xyz(env)], []
    for _ in range(steps):
        before = float(env.data.time)
        action = controller.action(env)
        result = env.step(action)
        if len(result) != 5 or abs(float(env.data.time) - before - dt) > 1e-9:
            raise RuntimeError("env.step did not advance exactly one control step")
        executed = env.last_transition["action"] if getattr(env, "last_transition", None) else action
        actions.append(np.asarray(executed, dtype=float).copy())
        tilt.append(guard_tilts(env))
        unsafe.append(bool(env.safety_report()["unsafe"]))
        pinch.append(pinch_xyz(env))
    tilt, unsafe = np.asarray(tilt), np.asarray(unsafe, dtype=bool)
    if not np.isfinite(tilt).all():
        raise RuntimeError("Non-finite guard tilt during rollout")
    hits = np.flatnonzero(unsafe)
    return {"actions": np.asarray(actions).reshape(steps, 5), "tilt": tilt, "unsafe": unsafe, "pinch_xyz": np.asarray(pinch),
            "max_tilt_deg": float(tilt.max()), "reached_unsafe": bool(unsafe.any()),
            "first_unsafe_step": int(hits[0]) if len(hits) else None, "final_tilt_deg": tilt[-1].tolist(),
            "pinch_drift_max_mm": float(1000 * np.linalg.norm(np.asarray(pinch) - pinch[0], axis=1).max())}


# ----- determinism ----------------------------------------------------------------------------
def compare_pixels(rendered, stored_png):
    stored = np.asarray(Image.open(stored_png).convert("RGB"))
    if rendered.shape != stored.shape or rendered.dtype != np.uint8:
        raise ValueError(f"Rendered image {rendered.shape} {rendered.dtype} does not match {stored_png} {stored.shape}")
    diff = np.abs(rendered.astype(np.int16) - stored.astype(np.int16))
    return {"fraction_gt2": float((diff.max(-1) > PIXEL_LEVELS).mean()), "max_abs_diff": int(diff.max()),
            "identical": bool(diff.max() == 0)}


def determinism_episode(env, loaded, event, render=False, image_dir=None):
    """Restore before the onset, replay recorded actions past it, measure drift from the record."""
    onset, n = loaded["onset_frame"], loaded["frames"]
    before = DETERMINISM_BEFORE if onset >= DETERMINISM_BEFORE else DETERMINISM_BEFORE_SHORT
    start, end = max(onset - before, 0), min(onset + DETERMINISM_AFTER, n - 1)
    render_at = sorted({start, onset, end})
    restore(env, loaded["dir"], start, loaded)
    pixels = []

    def snapshot(frame):
        if render and frame in render_at:
            image = env.render(camera=loaded["camera"])
            row = {"frame": frame, **compare_pixels(image, Path(loaded["dir"]) / "images" / f"{frame:06d}.png")}
            if image_dir is not None:
                name = f"ds{event['dataset']}_ep{event['episode_id']:04d}_f{frame:06d}_rerender.png"
                Image.fromarray(image).save(Path(image_dir) / name, compress_level=1)
                row["rerender_png"] = name
            pixels.append(row)

    snapshot(start)
    qpos_diff, tilt_diff, com_diff, unsafe_equal = [], [], [], []
    for frame in range(start, end):
        env.step(loaded["actions"][frame])
        k = frame + 1
        qpos_diff.append(float(np.abs(env.data.qpos - loaded["qpos"][k]).max()))
        tilt_diff.append(float(np.abs(guard_tilts(env) - loaded["guard_tilt_deg"][k]).max()))
        com = np.array([env.data.body(f"object_{i}").xipos for i in range(3)])
        com_diff.append(float(np.abs(com - loaded["object_com_xyz"][k]).max()))
        unsafe_equal.append(bool(env.safety_report()["unsafe"]) == bool(loaded["unsafe"][k]))
        snapshot(k)
    diverged = [start + 1 + i for i, d in enumerate(qpos_diff) if d > 0]
    return {"mode": "determinism", "dataset": event["dataset"], "episode_id": event["episode_id"], "path": event["path"],
            "stratum": event["stratum"], "onset_frame": onset, "restore_frame": start, "end_frame": end, "steps": end - start,
            "max_qpos_diff": max(qpos_diff), "max_tilt_diff_deg": max(tilt_diff), "max_com_diff_m": max(com_diff),
            "unsafe_mismatches": int(sum(not v for v in unsafe_equal)),
            "first_divergent_frame": diverged[0] if diverged else None,
            "exact": bool(not diverged and max(tilt_diff) == 0 and max(com_diff) == 0 and all(unsafe_equal)),
            "per_step": {"frame": list(range(start + 1, end + 1)), "qpos_diff": qpos_diff, "tilt_diff_deg": tilt_diff,
                         "com_diff_m": com_diff, "unsafe_equal": unsafe_equal},
            "pixels": pixels if render else None}


def aggregate_determinism(records):
    pixels = [p for r in records for p in (r["pixels"] or [])]
    return {"episodes": len(records), "steps_compared": int(sum(r["steps"] for r in records)),
            "exact_episodes": int(sum(r["exact"] for r in records)),
            "max_qpos_diff": max(r["max_qpos_diff"] for r in records),
            "max_tilt_diff_deg": max(r["max_tilt_diff_deg"] for r in records),
            "max_com_diff_m": max(r["max_com_diff_m"] for r in records),
            "unsafe_mismatches": int(sum(r["unsafe_mismatches"] for r in records)),
            "episodes_with_divergence": int(sum(r["first_divergent_frame"] is not None for r in records)),
            "pixels": None if not pixels else {
                "frames_compared": len(pixels), "identical_frames": int(sum(p["identical"] for p in pixels)),
                "max_fraction_gt2": max(p["fraction_gt2"] for p in pixels), "mean_fraction_gt2": float(np.mean([p["fraction_gt2"] for p in pixels])),
                "max_abs_diff": max(p["max_abs_diff"] for p in pixels)}}


# ----- no return ------------------------------------------------------------------------------
def no_return_event(env, loaded, event, leads, hold_steps=HOLD_STEPS, retreat_steps=RETREAT_STEPS,
                    retreat_speed=RETREAT_SPEED_MPS, grip="keep", variants=VARIANTS):
    """For each lead k: restore at onset-k and roll out hold (and retreat). Returns (record, rollout rows)."""
    onset = loaded["onset_frame"]
    worst = int(np.argmax(loaded["guard_tilt_deg"][onset]))          # 0/1 -> guard body object_{worst+1}
    dt = loaded["control_dt"]
    rows, per_lead = [], []
    for lead in leads:
        frame = onset - lead
        if frame < 0:
            per_lead.append({"lead": lead, "restore_frame": None, "skipped": "onset shorter than lead"})
            continue
        grip_command = float(loaded["actions"][frame][4]) if grip == "keep" else 0.0
        entry = {"lead": lead, "restore_frame": frame, "restore_tilt_deg": loaded["guard_tilt_deg"][frame].tolist(),
                 "grip_command": grip_command}
        for variant in variants:
            restore(env, loaded["dir"], frame, loaded)
            anchor = pinch_xyz(env)
            if variant == "hold":
                controller = AnchoredHold(anchor, grip_command, dt=dt)
                direction = None
            else:
                direction = retreat_direction(anchor, loaded["object_com_xyz"][frame][1 + worst],
                                              loaded["object_quat_wxyz"][frame][1 + worst])
                controller = AnchoredHold(anchor, grip_command, direction * retreat_speed, min(retreat_steps, hold_steps), dt=dt)
            result = rollout(env, controller, hold_steps)
            entry[variant] = {k: result[k] for k in ("max_tilt_deg", "reached_unsafe", "first_unsafe_step", "final_tilt_deg", "pinch_drift_max_mm")}
            if direction is not None:
                entry[variant]["direction_xy"] = direction.tolist()
            rows.append({"dataset": event["dataset"], "episode_id": event["episode_id"], "stratum": event["stratum"],
                         "onset_frame": onset, "restore_frame": frame, "lead": lead, "variant": variant,
                         "actions": result["actions"], "tilt": result["tilt"], "unsafe": result["unsafe"], "pinch_xyz": result["pinch_xyz"]})
        per_lead.append(entry)
    record = {"mode": "no_return", "dataset": event["dataset"], "episode_id": event["episode_id"], "path": event["path"],
              "stratum": event["stratum"], "onset_frame": onset, "worst_guard_index": worst + 1,
              "onset_tilt_deg": loaded["guard_tilt_deg"][onset].tolist(), "grip_policy": grip, "leads": per_lead,
              "min_lead": {}, "avoid_leads": {}, "monotone": {}}
    for variant in variants:
        avoid = [e["lead"] for e in per_lead if e.get(variant) is not None and not e[variant]["reached_unsafe"]]
        tried = [e["lead"] for e in per_lead if e.get(variant) is not None]
        record["avoid_leads"][variant] = avoid
        record["min_lead"][variant] = min(avoid) if avoid else None
        record["monotone"][variant] = bool(not avoid or all(k in avoid for k in tried if k >= min(avoid)))
    return record, rows


def aggregate_no_return(records, leads, variants=VARIANTS):
    """Distribution of the minimal avoiding lead per stratum and overall, per variant."""
    def block(rows, variant):
        mins = [r["min_lead"][variant] for r in rows]
        found = [m for m in mins if m is not None]
        counts = Counter(mins)
        return {"events": len(rows), "min_lead_counts": {**{str(k): int(counts.get(k, 0)) for k in leads}, "none": int(counts.get(None, 0))},
                "median_min_lead": float(np.median(found)) if found else None,
                "unavoidable_up_to_max_lead": int(counts.get(None, 0)),
                "avoidable_fraction_le": {str(t): float(np.mean([m is not None and m <= t for m in mins])) for t in AVOID_THRESHOLDS},
                "non_monotone_events": int(sum(not r["monotone"][variant] for r in rows))}
    strata = sorted({r["stratum"] for r in records})
    return {"events": len(records), "leads": list(leads),
            "overall": {v: block(records, v) for v in variants},
            "by_stratum": {s: {v: block([r for r in records if r["stratum"] == s], v) for v in variants} for s in strata}}


def save_rollouts(path, rows, hold_steps):
    if not rows:
        raise ValueError("No rollouts to save")
    arrays = {"dataset": np.array([r["dataset"] for r in rows], dtype=np.int64),
              "episode_id": np.array([r["episode_id"] for r in rows], dtype=np.int64),
              "stratum": np.array([r["stratum"] for r in rows], dtype=str),
              "onset_frame": np.array([r["onset_frame"] for r in rows], dtype=np.int64),
              "restore_frame": np.array([r["restore_frame"] for r in rows], dtype=np.int64),
              "lead": np.array([r["lead"] for r in rows], dtype=np.int64),
              "variant": np.array([r["variant"] for r in rows], dtype=str),
              "steps": np.full(len(rows), hold_steps, dtype=np.int64),
              "actions": np.stack([r["actions"] for r in rows]), "tilt": np.stack([r["tilt"] for r in rows]),
              "unsafe": np.stack([r["unsafe"] for r in rows]), "pinch_xyz": np.stack([r["pinch_xyz"] for r in rows])}
    if arrays["actions"].shape != (len(rows), hold_steps, 5) or arrays["tilt"].shape != (len(rows), hold_steps + 1, 2):
        raise RuntimeError("Rollout arrays have inconsistent shapes")
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


# ----- report ---------------------------------------------------------------------------------
def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}" if value != 0 else "0"
    return str(value)


def _cell(text):
    """Stratum names contain '|', which would split a Markdown table cell."""
    return str(text).replace("|", "\\|")


def write_report(output, summary, records):
    lines = ["# 恢复中间帧检查：确定性与不可挽回点\n",
             f"- split：`{summary['split']['path']}`（sha256 `{summary['split']['sha256'][:12]}…`）；seed {summary['args']['seed']}；模式 {summary['args']['mode']}。",
             f"- 恢复方法：`env.reset()` → 在 time=0 设置记录的悬停目标 → `mj_setState(16383)` → `mj_forward` → `post_step`。恢复帧本身的 qpos/qvel/time/倾角/unsafe 与记录逐位相等（不等直接报错）。",
             f"- 渲染环境：{json.dumps(summary['render_environment'], ensure_ascii=False)}；与各数据集 manifest 一致：{summary['render_environment_matches_manifest']}。",
             f"- 保持动作：{summary['hold_law']['description']}；夹爪命令策略 `{summary['args']['grip']}`。\n"]
    det = [r for r in records if r["mode"] == "determinism"]
    if det:
        a = summary["determinism"]
        lines += ["## 1. 确定性（恢复 onset−10 帧，重放记录动作到 onset+5）\n",
                  f"- {a['episodes']} 条轨迹，共 {a['steps_compared']} 步；完全一致（所有差为 0、unsafe 全对）的轨迹：{a['exact_episodes']}/{a['episodes']}。",
                  f"- 最大差：qpos {_fmt(a['max_qpos_diff'])}，红柱倾角 {_fmt(a['max_tilt_diff_deg'])}°，物体 COM {_fmt(a['max_com_diff_m'])} m；unsafe 不一致步数 {a['unsafe_mismatches']}。"]
        if a["pixels"]:
            p = a["pixels"]
            lines.append(f"- 像素：比较 {p['frames_compared']} 帧，逐像素相同 {p['identical_frames']} 帧；差 >{PIXEL_LEVELS} 级的像素比例最大 {_fmt(p['max_fraction_gt2'])}（平均 {_fmt(p['mean_fraction_gt2'])}），最大绝对差 {p['max_abs_diff']}。")
        else:
            lines.append("- 未渲染（没有 `--render`）。")
        lines += ["", "| 数据集 | 轨迹 | 层 | 恢复帧 | 终帧 | 步数 | qpos 最大差 | 倾角最大差(°) | COM 最大差(m) | unsafe 不一致 | 首个分歧帧 | 像素（帧：差>2 比例 / 最大差） |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in det:
            pix = "—" if r["pixels"] is None else "；".join(f"{p['frame']}：{_fmt(p['fraction_gt2'])} / {p['max_abs_diff']}" for p in r["pixels"])
            lines.append(f"| {r['dataset']} | {r['episode_id']:04d} | {_cell(r['stratum'])} | {r['restore_frame']} | {r['end_frame']} | {r['steps']} | "
                         f"{_fmt(r['max_qpos_diff'])} | {_fmt(r['max_tilt_diff_deg'])} | {_fmt(r['max_com_diff_m'])} | {r['unsafe_mismatches']} | "
                         f"{_fmt(r['first_divergent_frame'])} | {pix} |")
        lines.append("")
    nr = [r for r in records if r["mode"] == "no_return"]
    if nr:
        a = summary["no_return"]
        args = summary["args"]
        lines += ["## 2. 不可挽回点（恢复 onset−k 帧，改成停下 / 撤离）\n",
                  f"- {a['events']} 个事件；k ∈ {{{', '.join(str(k) for k in a['leads'])}}}；每次 rollout {args['hold_steps']} 步（{args['hold_steps'] * 0.05:g} s）。"
                  f"hold = 闭环保持恢复帧的夹爪位置；retreat = 前 {min(args['retreat_steps'], args['hold_steps'])} 步以 {100 * args['retreat_speed']:g} cm/s 水平远离倾倒红柱，再保持。",
                  "- \"最小 k\" = 最小的提前步数，使该变体在整段 rollout 内红柱倾角不超过 30°；\"无\" = 直到最大 k 都躲不掉。", ""]
        for variant in VARIANTS:
            if variant not in a["overall"]:
                continue
            lines += [f"### {variant}\n",
                      "| 层 | 事件 | 最小 k 分布（k:次数） | 无 | 中位数 | ≤3 | ≤5 | ≤10 | ≤20 | 非单调 |", "|---|---|---|---|---|---|---|---|---|---|"]
            for name, blocks in [*sorted(a["by_stratum"].items()), ("**全部**", a["overall"])]:
                b = blocks[variant]
                dist = "，".join(f"{k}:{v}" for k, v in b["min_lead_counts"].items() if k != "none" and v)
                frac = b["avoidable_fraction_le"]
                lines.append(f"| {_cell(name)} | {b['events']} | {dist or '—'} | {b['min_lead_counts']['none']} | {_fmt(b['median_min_lead'])} | "
                             f"{frac['3']:.2f} | {frac['5']:.2f} | {frac['10']:.2f} | {frac['20']:.2f} | {b['non_monotone_events']} |")
            lines.append("")
        lines += ["### 每个事件\n", "| 数据集 | 轨迹 | 层 | onset | 倾倒红柱 | hold 最小 k | retreat 最小 k | hold 各 k 最大倾角(°) |", "|---|---|---|---|---|---|---|---|"]
        for r in nr:
            tilts = " ".join(f"{e['lead']}:{e['hold']['max_tilt_deg']:.0f}" for e in r["leads"] if e.get("hold"))
            lines.append(f"| {r['dataset']} | {r['episode_id']:04d} | {_cell(r['stratum'])} | {r['onset_frame']} | {r['worst_guard_index']} | "
                         f"{_fmt(r['min_lead'].get('hold'))} | {_fmt(r['min_lead'].get('retreat'))} | {tilts} |")
        lines.append("")
    lines += ["## 说明\n",
              "- 这里只测量，不做断言；确定性数字只代表当前代码、MuJoCo 版本和渲染后端下的复现情况。",
              "- `rollouts.npz` 保存了每条 rollout 实际执行的归一化动作序列和逐步倾角，供 world model 做同动作序列的反事实对比。",
              f"- 用时 {summary['elapsed_seconds']:.1f} s。"]
    (Path(output) / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----- CLI ------------------------------------------------------------------------------------
def parse_leads(text):
    """'1,2,...,20', '1-20', '1..20' or explicit lists -> sorted unique positive ints."""
    leads, previous = [], None
    tokens = [t.strip() for t in str(text).split(",") if t.strip()]
    for i, token in enumerate(tokens):
        if token in ("...", ".."):
            if previous is None or i + 1 >= len(tokens):
                raise ValueError("'...' needs a number on both sides")
            leads.extend(range(previous + 1, int(tokens[i + 1])))
            continue
        for sep in ("-", ".."):
            if sep in token:
                low, high = (int(v) for v in token.split(sep, 1))
                leads.extend(range(low, high + 1))
                previous = high
                break
        else:
            leads.append(int(token))
            previous = leads[-1]
    leads = sorted(set(leads))
    if not leads or leads[0] <= 0:
        raise ValueError("leads must be positive integers")
    return leads


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, required=True, help="benchmark_split.json of the danger benchmark")
    p.add_argument("--output", type=Path, required=True, help="NEW directory")
    p.add_argument("--mode", choices=MODES, default="both")
    p.add_argument("--events", type=int, default=None, help=f"bench_test events per mode (default {DEFAULT_EVENTS})")
    p.add_argument("--leads", default=",".join(str(k) for k in DEFAULT_LEADS), help="no_return leads, e.g. 1,2,...,20 or 1-20")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="determinism only: re-render front_pixels at 3 frames and compare with the PNGs")
    p.add_argument("--hold-steps", type=int, default=HOLD_STEPS, help="rollout length in control steps (default 60 = 3 s)")
    p.add_argument("--retreat-steps", type=int, default=RETREAT_STEPS, help="moving steps of the retreat variant (default 20 = 1 s)")
    p.add_argument("--retreat-speed", type=float, default=RETREAT_SPEED_MPS, help="m/s, horizontal, away from the tilting guard")
    p.add_argument("--grip", choices=GRIP_POLICIES, default="keep", help="gripper command during rollouts: keep the recorded one or 0")
    p.add_argument("--no-retreat", action="store_true", help="skip the retreat variant")
    p.add_argument("--strata", default=None, help="comma-separated stratum prefixes (default: the 12 direction|severity cells)")
    p.add_argument("--order", choices=ORDERS, default="seeded", help="seeded round-robin over strata, or the first events as listed")
    p.add_argument("--data-root", type=Path, default=None, help="directory holding the dataset folders if the split's paths do not resolve")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    leads = parse_leads(a.leads)
    if a.events is not None and a.events <= 0:
        raise ValueError("--events must be positive")
    if a.hold_steps <= 0 or a.retreat_steps < 0:
        raise ValueError("--hold-steps must be positive and --retreat-steps non-negative")
    if not np.isfinite(a.retreat_speed) or a.retreat_speed <= 0 or a.retreat_speed * 0.05 > ACTION_XYZ_LIMIT_M:
        raise ValueError("--retreat-speed must be positive and at most 1 m/s")
    if a.seed < 0:
        raise ValueError("--seed must be non-negative")
    strata = None if a.strata is None else [s.strip() for s in a.strata.split(",") if s.strip()]
    split = load_split(a.split, a.data_root)
    for d in split["datasets"]:
        if output.is_relative_to(Path(d["path"])):
            raise ValueError("Output must be outside the datasets")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    env, records, rollouts = None, [], []
    variants = ("hold",) if a.no_retreat else VARIANTS
    render_env = {k: os.environ.get(k) for k in RENDER_KEYS}
    try:
        modes = ("determinism", "no_return") if a.mode == "both" else (a.mode,)
        selection = {m: select_events(split["events"], a.events or DEFAULT_EVENTS[m], a.seed, strata, a.order) for m in modes}
        env = ExplorationScene()
        image_dir = None
        if a.render and "determinism" in modes:
            image_dir = output / "images"
            image_dir.mkdir()
        with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
            for mode in modes:
                events = selection[mode]
                for i, event in enumerate(events):
                    t0 = time.perf_counter()
                    loaded = load_episode(event["episode_dir"], event)
                    if mode == "determinism":
                        record = determinism_episode(env, loaded, event, a.render, image_dir)
                        note = (f"restore {record['restore_frame']}->{record['end_frame']} max dqpos {_fmt(record['max_qpos_diff'])} "
                                f"dtilt {_fmt(record['max_tilt_diff_deg'])} dcom {_fmt(record['max_com_diff_m'])} unsafe_mismatch {record['unsafe_mismatches']}")
                        if record["pixels"]:
                            note += " pixels " + " ".join(f"{p['frame']}:{_fmt(p['fraction_gt2'])}/{p['max_abs_diff']}" for p in record["pixels"])
                    else:
                        record, rows = no_return_event(env, loaded, event, leads, a.hold_steps, a.retreat_steps, a.retreat_speed, a.grip, variants)
                        rollouts.extend(rows)
                        note = "min lead " + " ".join(f"{v}={_fmt(record['min_lead'][v])}" for v in variants)
                    records.append(record)
                    stream.write(json.dumps(native(record), ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    print(f"[{mode} {i + 1}/{len(events)}] ds{event['dataset']} ep{event['episode_id']:04d} {event['stratum']} onset {event['onset_frame']}: "
                          f"{note} | {time.perf_counter() - t0:.1f}s, elapsed {time.perf_counter() - started:.0f}s", flush=True)
        summary = {"format": FORMAT, "args": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "leads": leads,
                                              "variants": list(variants), "strata": strata},
                   "split": {k: split[k] for k in ("path", "sha256", "seed", "datasets", "test_episodes", "dev_episodes")},
                   "events_available": len(split["events"]),
                   "selected": {m: [{k: e[k] for k in ("dataset", "episode_id", "stratum", "onset_frame")} for e in selection[m]] for m in modes},
                   "render_environment": render_env,
                   "render_environment_matches_manifest": {d["path"]: d["render_environment"] == render_env if d["render_environment"] else None
                                                           for d in split["datasets"]},
                   "hold_law": {"type": "anchored_pi", "gain_p": HOLD_GAIN_P, "gain_i": HOLD_GAIN_I,
                                "description": "hold = 闭环 PI 保持恢复帧的夹爪位置（零动作会以约 1.3 mm/步下沉，不是保持）"},
                   "state_spec": STATE_SPEC, "pixel_levels": PIXEL_LEVELS,
                   "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                                "ogbench": version("ogbench")},
                   "source_sha256": {"bench_restore_check.py": sha256(Path(__file__)),
                                     "three_block_scene.py": sha256(Path(__file__).resolve().parent / "three_block_scene.py"),
                                     "exploration.py": sha256(Path(__file__).resolve().parent / "exploration.py")},
                   "datasets_modified": False, "images_written_inside_datasets": False}
        if "determinism" in modes:
            summary["determinism"] = aggregate_determinism([r for r in records if r["mode"] == "determinism"])
        if "no_return" in modes:
            summary["no_return"] = aggregate_no_return([r for r in records if r["mode"] == "no_return"], leads, variants)
            save_rollouts(output / "rollouts.npz", rollouts, a.hold_steps)
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_report(output, summary, records)
        write_new_json(output / "summary.json", summary)
        print(f"Complete: {output / 'summary.json'} ({len(records)} records, {len(rollouts)} rollouts)", flush=True)
        return summary
    except BaseException as error:
        write_new_json(output / "FAILED.json", {"error": f"{type(error).__name__}: {error}", "records_written": len(records)})
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
