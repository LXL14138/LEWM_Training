"""SIMULATION ONLY: counterfactual BRANCH episodes with rendered images from recorded TRAINING fall events.

Proposal E4-a (handoff/danger_benchmark_20260923/WM_IMPROVEMENT_PROPOSAL_20260925.md section 5 E4,
WM_IMPROVEMENT_RESPONSE_FABLE.md step 3). For every selected fall event of the TRAINING material
(the split file's datasets, e.g. data/danger_train_split_001 -> danger_train_001/002; danger benchmark
datasets are refused) and every lead k:

  restore the recorded ``physics_state`` ``prefix`` frames before restore_frame = onset - k
  (bench_restore_check recipe: reset -> hover goal -> mj_setState(16383) -> mj_forward -> post_step),
  re-base the simulator clock to 0, replay the recorded actions up to the restore frame (the recorded
  pre-onset history, expected bit-exact; the measured difference is recorded), then run ONE action law
  for --hold-steps control steps: continue / hold / retreat / push_back (see counterfactual_plan).

Every rollout is a NEW raw episode directory in the exploration on-disk format written by the SAME
recorder as the datasets (ExplorationRecorder: images/*.png of the front camera at every state,
trajectory.npz with every field, metadata.json with status complete), with phase labels
CF_CONTINUE / CF_HOLD / CF_RETREAT / CF_PUSHBACK, interaction_revision 4 and a ``counterfactual``
record; the manifest carries a ``counterfactual_branches`` record (split, selection, leads, branches,
job digest). Datasets are read-only; the output directory must be new; on interruption the manifest
is reconciled from the episode metadata (complete episodes stay usable, like collect_exploration_trajs).
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import time

import mujoco
import numpy as np

from bench_restore_check import (
    ACTION_XYZ_LIMIT_M, HOLD_GAIN_I, HOLD_GAIN_P, RENDER_KEYS, STATE_SPEC, AnchoredHold, guard_tilts, load_episode,
    parse_leads, pinch_xyz, resolve_dataset_dir, retreat_direction, select_events,
)
from collect_exploration_trajs import CLASS_NAMES, DATASET_KIND, ExplorationRecorder, _episode_row
from collect_grasp_trajs import episode_metadata, source_fingerprints, write_json
from collection_workers import CHILD_THREAD_LIMITS, run_workers
from counterfactual_plan import (
    BRANCH_MODE, BRANCH_PHASE, BRANCHES, CF_CONTROLLER_TYPE, CF_FORMAT, CF_INTERACTION_REVISION, DEFAULT_HOLD_STEPS,
    DEFAULT_LEADS, DEFAULT_MOVE_SPEED_MPS, DEFAULT_MOVE_STEPS, DEFAULT_PREFIX, HOLD_PHASE, MIN_PREFIX, PARTS, PLAN_KEYS,
    REPLAY_TOLERANCE, jobs_digest, parse_branches, plan_jobs, push_back_direction, validate_branch_record,
)
from exploration import ExplorationScene, GOAL_HIGH, GOAL_LOW
from wm_training.danger_strata import body_up, native, read_json, sha256

SPLIT_FORMAT = "danger_bench_split_v1"
STOP_REASON = "cf_hold_steps_completed"
ORDERS = ("seeded", "listed")
BYTES_PER_FRAME_ESTIMATE = 45_000          # measured: ~38 KB per 640x480 PNG of this scene + npz share
DISK_RESERVE_BYTES = 20 * 1024**3
MANIFEST_WRITE_INTERVAL_S = 30.0
SCENE_KEYS = ("schema_version", "dataset_kind", "image", "control_dt", "physics_dt", "action_order", "action_low",
              "action_high", "fixed_yaw_deg", "blocks", "unsafe_tilt_deg", "displacement_check_enabled", "safety_scope")
HOLD_LAW = {"type": "anchored_pi", "gain_p": HOLD_GAIN_P, "gain_i": HOLD_GAIN_I,
            "description": "closed-loop PI hold of the pinch at the branch point (bench_restore_check.AnchoredHold); "
                           "retreat/push_back move the reference horizontally for move_steps then keep it fixed"}
SOURCE_FILES = ("collect_counterfactual_branches.py", "counterfactual_plan.py", "bench_restore_check.py", "exploration.py",
                "exploration_validation.py", "collect_exploration_trajs.py", "collect_grasp_trajs.py", "collection_workers.py",
                "three_block_scene.py", "scene_config.py", "task_progress.py")


# ----- split and events -----------------------------------------------------------------------
def load_split_events(path, part, data_root=None):
    """Datasets (resolved, manifest sha256 checked, benchmark datasets refused) and every fall event of ``part``."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    path = Path(path)
    split = read_json(path)
    if split.get("format") != SPLIT_FORMAT:
        raise ValueError(f"{path}: not a {SPLIT_FORMAT} file")
    datasets = []
    for row in split["datasets"]:
        folder = resolve_dataset_dir(row["path"], path, data_root)
        if folder.name.startswith("danger_bench") or "bench" in Path(row["path"]).name.split("_")[:2]:
            raise ValueError(f"{folder}: danger benchmark datasets are evaluation material, never a source of training data")
        manifest = read_json(folder / "manifest.json")
        if sha256(folder / "manifest.json") != row["manifest_sha256"]:
            raise ValueError(f"{folder}: manifest.json changed since the split was written")
        if manifest.get("interaction_revision", 1) < 3 or not manifest.get("benchmark_schedule"):
            raise ValueError(f"{folder}: expected a targeted (interaction_revision 3) training collection")
        datasets.append({"index": int(row["index"]), "path": str(folder), "recorded_path": row["path"],
                         "manifest_sha256": row["manifest_sha256"], "status": manifest.get("status"),
                         "render_environment": manifest.get("execution", {}).get("render_environment"),
                         "versions": manifest.get("versions")})
    by_index = {d["index"]: d for d in datasets}
    membership = {}
    for name in ("test", "dev"):
        for entries in split["strata"].values():
            for e in entries[name]:
                key = (int(e["dataset"]), int(e["episode_id"]))
                if key in membership:
                    raise ValueError(f"Episode {key} appears twice in the split")
                membership[key] = name
    events = []
    for entry in split["episodes"]:
        key = (int(entry["dataset"]), int(entry["episode_id"]))
        which = membership.get(key)
        if which is None:
            raise ValueError(f"Episode {key} is in no part of the split")
        if entry.get("onset_frame") is None or (part != "all" and which != part):
            continue
        events.append({"dataset": key[0], "episode_id": key[1], "path": entry["path"], "stratum": str(entry["stratum"]),
                       "part": which, "onset_frame": int(entry["onset_frame"]), "frames": int(entry["frames"]),
                       "episode_dir": str(Path(by_index[key[0]]["path"]) / entry["path"])})
    if not events:
        raise ValueError(f"The split lists no fall events in part {part}")
    counts = {name: sum(1 for v in membership.values() if v == name) for name in ("test", "dev")}
    return {"path": str(path.resolve()), "sha256": sha256(path), "seed": split.get("seed"), "datasets": datasets,
            "events": events, "part": part, "test_episodes": counts["test"], "dev_episodes": counts["dev"]}


def choose_events(events, n, seed, strata, order, all_events):
    if all_events:
        return sorted(events, key=lambda e: (e["dataset"], e["episode_id"]))
    return select_events(events, n, seed, strata, order)


# ----- restore and controllers ------------------------------------------------------------------
def load_source(episode_dir):
    """bench_restore_check.load_episode plus the recorded phase labels (for the replayed prefix)."""
    loaded = load_episode(episode_dir)
    if loaded["onset_frame"] is None:
        raise ValueError(f"{episode_dir}: no unsafe frame, not a fall event")
    with np.load(Path(episode_dir) / "trajectory.npz", allow_pickle=False) as npz:
        phases = np.asarray(npz["action_phase"]).astype(str)
    if phases.shape != (loaded["frames"] - 1,):
        raise ValueError(f"{episode_dir}: action_phase has the wrong length")
    loaded["action_phase"] = phases
    return loaded


def restore_rebased(env, loaded, frame):
    """bench_restore_check.restore, but the branch episode's clock starts at 0.

    reset -> hover goal (time 0) -> mj_setState(16383) -> data.time = 0 -> mj_forward -> post_step. MuJoCo's
    dynamics do not depend on the absolute time, so the restored state evolves exactly as recorded; the
    validators require episodes to start at time 0. qpos/qvel must equal the record bit for bit, tilt and
    unsafe within 1e-9 (anything else is a model/data mismatch, not a measurement).
    """
    frame = int(frame)
    if not 0 <= frame < loaded["frames"]:
        raise ValueError(f"frame {frame} outside 0..{loaded['frames'] - 1}")
    if abs(float(env.control_timestep()) - loaded["control_dt"]) > 1e-12:
        raise ValueError("control_dt of the scene and of the recording disagree")
    state = np.asarray(loaded["physics_state"][frame], dtype=np.float64)
    env.reset(seed=0)
    expected = mujoco.mj_stateSize(env.model, STATE_SPEC)
    if state.shape != (expected,) or not np.isfinite(state).all():
        raise ValueError(f"physics_state has {state.shape} values; this model needs ({expected},)")
    env.set_hover_goal(loaded["hover_goal"])                 # only legal while env.data.time == 0
    mujoco.mj_setState(env.model, env.data, state, STATE_SPEC)
    env.data.time = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.post_step()
    if not np.array_equal(env.data.qpos, loaded["qpos"][frame]) or not np.array_equal(env.data.qvel, loaded["qvel"][frame]):
        raise RuntimeError(f"{loaded['dir']} frame {frame}: restored qpos/qvel differ from the recorded arrays")
    if float(env.data.time) != 0.0:
        raise RuntimeError("Clock re-base failed")
    safety = env.safety_report()
    if (not np.allclose(guard_tilts(env), loaded["guard_tilt_deg"][frame], atol=1e-9, rtol=0)
            or bool(safety["unsafe"]) != bool(loaded["unsafe"][frame])):
        raise RuntimeError(f"{loaded['dir']} frame {frame}: restored guard tilt / unsafe differ from the record")
    return env


class ReplayThenHold:
    """continue: the recorded actions from the restore frame on (as far as the recording goes), then AnchoredHold."""

    def __init__(self, loaded, restore_frame, dt):
        self.actions, self.frame, self.dt = loaded["actions"], int(restore_frame), float(dt)
        self.replayed, self.hold = 0, None

    def action(self, env):
        if self.frame < len(self.actions):
            action = np.asarray(self.actions[self.frame], dtype=float).copy()
            self.frame += 1
            self.replayed += 1
            return action, BRANCH_PHASE["continue"]
        if self.hold is None:
            self.hold = AnchoredHold(pinch_xyz(env), float(self.actions[-1][4]), dt=self.dt)
        return self.hold.action(env), HOLD_PHASE


class LabelledHold:
    """AnchoredHold with the branch phase for the first ``active_steps`` steps and CF_HOLD afterwards."""

    def __init__(self, controller, active_steps, active_phase):
        self.controller, self.active_steps, self.active_phase, self.step = controller, int(active_steps), active_phase, 0

    def action(self, env):
        phase = self.active_phase if self.step < self.active_steps else HOLD_PHASE
        self.step += 1
        return self.controller.action(env), phase


def build_controller(env, job, loaded):
    """Controller for the branch at the branch point (env is at restore_frame); returns (controller, bookkeeping)."""
    branch, restore_frame, onset = job["branch"], job["restore_frame"], loaded["onset_frame"]
    dt, hold_steps, move_steps = loaded["control_dt"], job["hold_steps"], job["move_steps"]
    worst = int(np.argmax(loaded["guard_tilt_deg"][onset]))            # 0/1 -> body object_{worst+1}
    grip = float(loaded["actions"][restore_frame][4])
    anchor = pinch_xyz(env)
    info = {"active_steps": hold_steps, "grip_command": grip, "direction_xy": None, "lean_source": None,
            "worst_guard_index": worst + 1}
    if branch == "continue":
        controller = ReplayThenHold(loaded, restore_frame, dt)
        info["active_steps"] = int(min(hold_steps, loaded["frames"] - 1 - restore_frame))
    elif branch == "hold":
        controller = LabelledHold(AnchoredHold(anchor, grip, dt=dt), hold_steps, HOLD_PHASE)
    else:
        com, quat = loaded["object_com_xyz"][restore_frame][1 + worst], loaded["object_quat_wxyz"][restore_frame][1 + worst]
        if branch == "retreat":
            direction = retreat_direction(anchor, com, quat)
            away = anchor[:2] - com[:2]
            source = "guard_com" if np.linalg.norm(away) >= 0.005 else "restore" if np.linalg.norm(body_up(quat)[:2]) >= 1e-6 else "base_minus_x"
        else:
            direction, source = push_back_direction(body_up(quat)[:2], body_up(loaded["object_quat_wxyz"][onset][1 + worst])[:2])
        hold = AnchoredHold(anchor, grip, direction * job["move_speed_mps"], move_steps, dt=dt)
        controller = LabelledHold(hold, move_steps, BRANCH_PHASE[branch])
        info.update(active_steps=move_steps, direction_xy=[float(v) for v in direction], lean_source=source)
    return controller, info


# ----- metadata -----------------------------------------------------------------------------------
def scene_keys(metadata):
    return {key: metadata[key] for key in SCENE_KEYS}


def branch_plan(job, loaded, dataset):
    return {"source_dataset": int(job["dataset"]), "source_dataset_path": dataset["path"], "source_episode_id": int(job["episode_id"]),
            "source_path": job["source_path"], "source_manifest_sha256": dataset["manifest_sha256"],
            "source_npz_sha256": sha256(Path(job["episode_dir"]) / "trajectory.npz"), "source_stratum": job["stratum"],
            "source_part": job["part"], "source_frames": int(loaded["frames"]), "onset_frame": int(loaded["onset_frame"]),
            "lead": int(job["lead"]), "restore_frame": int(job["restore_frame"]), "prefix_frames": int(job["prefix"]),
            "branch": job["branch"], "phase": BRANCH_PHASE[job["branch"]], "hold_steps": int(job["hold_steps"]),
            "move_steps": int(job["move_steps"]), "move_speed_mps": float(job["move_speed_mps"]), "grip_policy": "keep"}


def branch_metadata(env, job, loaded, plan, camera):
    source = loaded["meta"]
    start_sample = {"kind": "restored_recorded_frame", "source_episode_dir": job["episode_dir"],
                    "restored_frame": plan["restore_frame"] - plan["prefix_frames"], "requested_start_xyz": None,
                    "actual_start_xyz": pinch_xyz(env).tolist(), "attempts": 0, "rejected": [], "validity": None,
                    "seeds_inherited_from_source": True}
    max_steps = plan["prefix_frames"] + plan["hold_steps"]
    metadata = episode_metadata(env, start_sample, int(source.get("action_seed", 0)), camera, 0.0, 0.0, max_steps)
    metadata.update(schema_version=3, dataset_kind=DATASET_KIND,
                    start_seed=source.get("start_seed"), action_seed=source.get("action_seed"),
                    execution={"pid": os.getpid()},
                    controller={"type": CF_CONTROLLER_TYPE, "mode": BRANCH_MODE[job["branch"]],
                                "interaction_revision": CF_INTERACTION_REVISION, "counterfactual": plan,
                                "max_steps": max_steps, "max_attempts": 0, "goal_sampling_attempts": 0, "hold_law": HOLD_LAW},
                    hover_goal_xyz=env.hover_goal.tolist(), hover_tolerance_m=env.hover_tolerance_m,
                    hover_speed_limit_mps=env.hover_speed_limit_mps, hover_required_seconds=env.hover_required_seconds,
                    class_names=CLASS_NAMES,
                    visibility_note="block_in_frame checks box corners; does not detect occlusion",
                    guard_contact_note="robot/green-to-red normal-force peak across the preceding control step")
    metadata["label_notes"].update(
        class_label="current unsafe takes priority; class 0 is instantaneous hover conditions, not one-second completion",
        hover_success="goal-conditioned one-second substep hold with no historical guard hazard",
        task_success="LEGACY lift-height success only; use hover_success for the new task",
        goal_conditions_met="LEGACY grip and lift height; use hover_conditions_met for the new task",
        time="re-based: frame 0 of this episode is recorded frame restore_frame - prefix_frames of the source episode",
        branch_latch="episode_failed / hover timers restart at this episode's first frame (all prefix frames precede the source onset)")
    cam = env.model.camera(camera).id
    metadata["image"].update(camera_position_world=env.data.cam_xpos[cam].tolist(),
                             camera_rotation_world=env.data.cam_xmat[cam].tolist(),
                             fovy_deg=float(env.model.cam_fovy[cam]))
    if scene_keys(metadata) != scene_keys(source):
        raise RuntimeError("Scene/camera/action settings of the branch differ from the source episode")
    return metadata


# ----- one branch episode ---------------------------------------------------------------------------
def generate_branch(env, job, loaded, dataset, camera="front_pixels", stop=None):
    """Restore, replay the prefix, run the branch, record; returns the manifest row. Same physics serial or parallel."""
    started = time.perf_counter()
    if loaded["onset_frame"] != job["onset_frame"] or loaded["frames"] != job["frames"]:
        raise ValueError(f"{job['episode_dir']}: onset/frames disagree with the split entry")
    prefix, restore_frame, hold_steps = job["prefix"], job["restore_frame"], job["hold_steps"]
    if not MIN_PREFIX <= prefix <= restore_frame:
        raise ValueError("prefix must be between MIN_PREFIX and the restore frame")
    start = restore_frame - prefix
    restore_rebased(env, loaded, start)
    plan = branch_plan(job, loaded, dataset)
    metadata = branch_metadata(env, job, loaded, plan, camera)
    recorder = None
    try:
        recorder = ExplorationRecorder(Path(job["output_dir"]) / f"episode_{job['id']:04d}", env, metadata, camera)
        replay_diff = 0.0

        def step_recorded(action, phase, frame):
            nonlocal replay_diff
            env.last_transition = None
            env.step(np.asarray(action, dtype=float))
            if env.last_transition is None:
                raise RuntimeError("Missing executed transition")
            recorder.append(env.last_transition, phase)
            replay_diff = max(replay_diff, float(np.abs(env.data.qpos - loaded["qpos"][frame + 1]).max()),
                              float(np.abs(env.data.qvel - loaded["qvel"][frame + 1]).max()))

        for j in range(prefix):
            if stop is not None and stop.is_set():
                raise InterruptedError("Collection cancelled by parent")
            frame = start + j
            step_recorded(loaded["actions"][frame], loaded["action_phase"][frame], frame)
        controller, info = build_controller(env, job, loaded)
        anchor = pinch_xyz(env)
        tilt, unsafe, pinch = [guard_tilts(env)], [bool(env.safety_report()["unsafe"])], [anchor]
        for j in range(hold_steps):
            if stop is not None and stop.is_set():
                raise InterruptedError("Collection cancelled by parent")
            action, phase = controller.action(env)
            if job["branch"] == "continue" and phase == BRANCH_PHASE["continue"]:
                step_recorded(action, phase, restore_frame + j)         # still replaying: compare with the record
            else:
                env.last_transition = None
                env.step(np.asarray(action, dtype=float))
                recorder.append(env.last_transition, phase)
            tilt.append(guard_tilts(env))
            unsafe.append(bool(env.safety_report()["unsafe"]))
            pinch.append(pinch_xyz(env))
        tilt, unsafe, pinch = np.asarray(tilt), np.asarray(unsafe, dtype=bool), np.asarray(pinch)
        if not np.isfinite(tilt).all():
            raise RuntimeError("Non-finite guard tilt during the branch")
        hits = np.flatnonzero(unsafe)
        record = {**plan, **info, "replay_max_qpos_diff": replay_diff, "replay_exact": bool(replay_diff <= REPLAY_TOLERANCE),
                  "max_tilt_deg": float(tilt.max()), "reached_unsafe": bool(unsafe.any()),
                  "first_unsafe_step": int(hits[0]) if len(hits) else None, "final_tilt_deg": tilt[-1].tolist(),
                  "pinch_drift_max_mm": float(1000 * np.linalg.norm(pinch - anchor, axis=1).max()),
                  "time_rebased_to_zero": True}
        validate_branch_record(record, plan)
        states = recorder.states
        outcome = {"phase": "DONE", "stop_reason": STOP_REASON, "mode": BRANCH_MODE[job["branch"]],
                   "control_steps": prefix + hold_steps, "attempts": 0,
                   "ever_lifted": bool(any(s["grasped"] and s["target_height_m"] >= 0.05 + 0.015 for s in states)),
                   "ever_hover_success": bool(any(s["hover_success"] for s in states)),
                   "events": [{"name": "branch_point", "time_s": prefix * loaded["control_dt"], "step": prefix, "branch": job["branch"]}]
                             + ([{"name": "guard_hazard_detected", "time_s": (prefix + int(hits[0])) * loaded["control_dt"],
                                  "step": prefix + int(hits[0])}] if len(hits) else []),
                   "segments": [{"phase": "recorded_prefix", "start_step": 0, "steps": prefix},
                                {"phase": BRANCH_PHASE[job["branch"]], "start_step": prefix, "steps": info["active_steps"]},
                                {"phase": HOLD_PHASE, "start_step": prefix + info["active_steps"], "steps": hold_steps - info["active_steps"]}],
                   "interaction_revision": CF_INTERACTION_REVISION, "interactions": [], "goal_sampling_attempts": 0,
                   "counterfactual": record, "training_data_collected": False, "world_model_loaded": False,
                   "uses_simulator_truth": True, "safety": env.safety_report()}
        recorder.metadata["execution"]["wall_seconds"] = time.perf_counter() - started
        result = recorder.finish(native(outcome))
        return branch_row(job["id"], result)
    except (Exception, KeyboardInterrupt) as error:
        if recorder is not None and recorder.metadata["status"] != "complete":
            recorder.mark_error(error)
        raise


def branch_row(index, metadata):
    record = metadata["outcome"]["counterfactual"]
    row = _episode_row(index, metadata)
    row.update({k: record[k] for k in ("branch", "lead", "restore_frame", "prefix_frames", "source_dataset", "source_episode_id",
                                       "source_stratum", "source_part", "reached_unsafe", "first_unsafe_step", "max_tilt_deg",
                                       "replay_exact", "worst_guard_index")})
    return row


# ----- workers ---------------------------------------------------------------------------------------
class SourceCache:
    """One loaded source episode per worker (consecutive jobs share it)."""

    def __init__(self):
        self.key, self.value = None, None

    def get(self, episode_dir):
        if self.key != episode_dir:
            self.key, self.value = episode_dir, load_source(episode_dir)
        return self.value


def _branch_worker(pipe, stop):
    signal.signal(signal.SIGINT, signal.SIG_IGN)      # only the parent handles Ctrl+C
    env, cache = None, SourceCache()
    try:
        while True:
            job = pipe.recv()
            if job is None or stop.is_set():
                return
            try:
                if env is None:
                    env = ExplorationScene()
                row = generate_branch(env, job, cache.get(job["episode_dir"]), job["dataset_info"], job["camera"], stop)
            except (Exception, KeyboardInterrupt) as error:
                pipe.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
                return
            pipe.send({"status": "complete", "row": row})
    finally:
        if env is not None:
            env.close()
        pipe.close()


def _reconcile_interrupted(output_dir, manifest, error):
    complete = {row["id"]: row for row in manifest["episodes"]}
    incomplete = []
    for index in range(manifest["requested_episodes"]):
        episode = output_dir / f"episode_{index:04d}"
        if not episode.exists():
            continue
        path = episode / "metadata.json"
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("status") == "complete":
                complete[index] = branch_row(index, metadata)
            else:
                if metadata.get("status") != "error":
                    metadata.update(status="error", error=f"Collection interrupted: {error}")
                    write_json(path, metadata)
                incomplete.append({"id": index, "path": episode.name, "status": "error"})
        except (OSError, ValueError, KeyError, TypeError) as read_error:
            incomplete.append({"id": index, "path": episode.name, "status": "error", "error": f"Cannot reconcile metadata: {read_error}"})
    manifest["episodes"] = sorted(complete.values(), key=lambda row: row["id"])
    manifest["incomplete_episodes"] = incomplete


# ----- dataset ---------------------------------------------------------------------------------------
def estimate_bytes(jobs):
    return int(sum(job["prefix"] + job["hold_steps"] + 1 for job in jobs) * BYTES_PER_FRAME_ESTIMATE)


def summarize(rows, wall_seconds, workers):
    by_branch = {}
    for b in BRANCHES:
        sel = [r for r in rows if r["branch"] == b]
        if sel:
            by_branch[b] = {"episodes": len(sel), "reached_unsafe": int(sum(r["reached_unsafe"] for r in sel)),
                            "max_tilt_deg_median": float(np.median([r["max_tilt_deg"] for r in sel]))}
    by_lead = {}
    for lead in sorted({r["lead"] for r in rows}):
        sel = [r for r in rows if r["lead"] == lead]
        by_lead[str(lead)] = {b: {"episodes": len([r for r in sel if r["branch"] == b]),
                                  "reached_unsafe": int(sum(r["reached_unsafe"] for r in sel if r["branch"] == b))}
                              for b in BRANCHES if any(r["branch"] == b for r in sel)}
    states = sum(r["num_states"] for r in rows)
    return {"episodes": len(rows), "total_actions": sum(r["num_actions"] for r in rows), "total_states": states,
            "source_events": len({(r["source_dataset"], r["source_episode_id"]) for r in rows}),
            "inexact_replays": int(sum(not r["replay_exact"] for r in rows)),
            "class_counts": {str(k): sum(r["class_counts"][str(k)] for r in rows) for k in range(3)},
            "guard_contact_steps": sum(r["guard_contact_steps"] for r in rows),
            "out_of_frame_states": sum(r["out_of_frame_states"] for r in rows),
            "by_branch": by_branch, "by_lead": by_lead, "wall_seconds": wall_seconds, "workers": workers,
            "states_per_wall_second": states / wall_seconds if wall_seconds else None,
            "actions_per_wall_second": sum(r["num_actions"] for r in rows) / wall_seconds if wall_seconds else None,
            "episode_wall_seconds_median": float(np.median([r["wall_seconds"] for r in rows if r.get("wall_seconds")])) if rows else None,
            "scope": "counterfactual branches of TRAINING fall events; coverage material for the WM fine-tune, no benchmark episode used"}


def collect_branches(output_dir, split_path, *, part="test", events=None, all_events=False, leads=DEFAULT_LEADS,
                     branches=BRANCHES, hold_steps=DEFAULT_HOLD_STEPS, prefix=DEFAULT_PREFIX, move_steps=DEFAULT_MOVE_STEPS,
                     move_speed=DEFAULT_MOVE_SPEED_MPS, seed=0, workers=1, camera="front_pixels", strata=None,
                     order="seeded", data_root=None, plan_only=False):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer from 1 to 4")
    if type(seed) is not int or seed < 0 or order not in ORDERS or camera not in ("front_pixels", "side_pixels"):
        raise ValueError("seed must be non-negative, order seeded|listed, camera front_pixels|side_pixels")
    if (events is None) == (not all_events):
        raise ValueError("Give exactly one of events=N or all_events=True")
    if not np.isfinite(move_speed) or move_speed <= 0 or move_speed * 0.05 > ACTION_XYZ_LIMIT_M:
        raise ValueError("move_speed must be positive and at most 1 m/s")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Use a NEW output directory: {output_dir}")
    split = load_split_events(split_path, part, data_root)
    for d in split["datasets"]:
        if output_dir.is_relative_to(Path(d["path"])):
            raise ValueError("Output must be outside the datasets")
    chosen = choose_events(split["events"], events, seed, strata, order, all_events)
    jobs, skipped = plan_jobs(chosen, leads, branches, hold_steps, prefix, move_steps, move_speed)
    by_index = {d["index"]: d for d in split["datasets"]}
    for job in jobs:
        job.update(output_dir=str(output_dir), camera=camera,
                   dataset_info={"path": by_index[job["dataset"]]["path"], "manifest_sha256": by_index[job["dataset"]]["manifest_sha256"]})
    digest = jobs_digest(jobs)
    frames = sum(job["prefix"] + job["hold_steps"] + 1 for job in jobs)
    estimate = estimate_bytes(jobs)
    plan_report = {"events_available": len(split["events"]), "events_selected": len(chosen), "jobs": len(jobs),
                   "skipped_event_leads": len(skipped), "frames": frames, "estimated_bytes": estimate,
                   "estimated_gib": estimate / 1024**3, "leads": [int(k) for k in leads], "branches": list(branches),
                   "hold_steps": hold_steps, "prefix": prefix, "jobs_sha256": digest}
    print(json.dumps(plan_report, ensure_ascii=False), flush=True)
    if plan_only:
        return plan_report
    parent = output_dir.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < estimate + DISK_RESERVE_BYTES:
        raise OSError(f"Not enough free space: need about {estimate / 1024**3:.1f} GiB + 20 GiB reserve, {free / 1024**3:.1f} GiB free")
    root = Path(__file__).resolve().parent
    fingerprints = source_fingerprints()
    for name in SOURCE_FILES:
        fingerprints[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    effective_workers = min(workers, len(jobs))
    render_env = {key: os.environ.get(key) for key in RENDER_KEYS}
    record = {"format": CF_FORMAT, "sha256": digest, "jobs_file": "branch_jobs.json",
              "split": {"path": split["path"], "sha256": split["sha256"], "seed": split["seed"], "part": part,
                        "test_episodes": split["test_episodes"], "dev_episodes": split["dev_episodes"]},
              "source_datasets": [{k: d[k] for k in ("index", "path", "recorded_path", "manifest_sha256", "status")} for d in split["datasets"]],
              "render_environment_matches_source": {d["path"]: (d["render_environment"] == render_env) if d["render_environment"] else None
                                                    for d in split["datasets"]},
              "selection": {"mode": "all_events" if all_events else order, "events_requested": None if all_events else events,
                            "seed": seed, "strata_filter": strata, "events_available": len(split["events"]),
                            "events": [{k: e[k] for k in ("dataset", "episode_id", "stratum", "part", "onset_frame", "frames")} for e in chosen],
                            "skipped_event_leads": skipped, "rule": f"an (event, lead) pair is skipped when onset - lead < {MIN_PREFIX}"},
              "leads": [int(k) for k in leads], "branches": list(branches), "hold_steps": hold_steps, "prefix_frames": prefix,
              "move_steps": move_steps, "move_speed_mps": move_speed, "grip_policy": "keep", "hold_law": HOLD_LAW,
              "direction_rules": {"retreat": "bench_restore_check.retreat_direction (away from the worst guard's COM at the restore frame)",
                                  "push_back": "against the horizontal part of the worst guard's up-vector at the restore frame, "
                                               "or at the onset when the guard is upright there (counterfactual_plan.push_back_direction)"},
              "time_rebased_to_zero": True, "replay_tolerance": REPLAY_TOLERANCE, "state_spec": STATE_SPEC,
              "note": "branch episodes start mid-episode (recorded prefix + one action law); which stratum a branch reaches is "
                      "measured afterwards (counterfactual_split.py / wm_training.danger_strata); no benchmark episode is used"}
    manifest = {"schema_version": 3, "dataset_kind": DATASET_KIND, "status": "collecting",
                "interaction_revision": CF_INTERACTION_REVISION, "benchmark_schedule": None, "counterfactual_branches": record,
                "requested_episodes": len(jobs), "seed": seed, "episodes": [],
                "fixed_object_layout": True, "objects_teleported_during_episode": False,
                "goal_bounds_xyz": [GOAL_LOW.tolist(), GOAL_HIGH.tolist()], "class_names": CLASS_NAMES,
                "max_seconds": (prefix + hold_steps) * 0.05,
                "execution": {"workers_requested": workers, "workers_used": effective_workers,
                              "start_method": "spawn" if effective_workers > 1 else "in_process",
                              "child_thread_limits": CHILD_THREAD_LIMITS if effective_workers > 1 else {},
                              "render_environment": render_env},
                "source_sha256": fingerprints,
                "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                             **{k: version(k) for k in ("ogbench", "stable-worldmodel")}}}
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "branch_jobs.json").open("x", encoding="utf-8") as stream:
        json.dump({"format": CF_FORMAT, "sha256": digest,
                   "jobs": [{k: v for k, v in job.items() if k not in ("output_dir", "dataset_info", "camera")} for job in jobs]},
                  stream, ensure_ascii=False, indent=1, allow_nan=False)
        stream.write("\n")
    write_json(output_dir / "manifest.json", manifest)
    started = time.perf_counter()
    state = {"last_write": started, "frames": 0}

    def write_progress(final=False):
        elapsed = time.perf_counter() - started
        done = len(manifest["episodes"])
        rate = state["frames"] / elapsed if elapsed > 0 else 0.0
        eta = (frames - state["frames"]) / rate if rate > 0 else None
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "progress.json", {"status": "complete" if final else "collecting", "done": done, "total": len(jobs),
                                                  "frames_done": state["frames"], "total_frames": frames, "elapsed_seconds": elapsed,
                                                  "states_per_second": rate, "eta_seconds": eta})
        state["last_write"] = time.perf_counter()

    def record_result(row):
        manifest["episodes"].append(row)
        manifest["episodes"].sort(key=lambda r: r["id"])
        state["frames"] += row["num_states"]
        elapsed = time.perf_counter() - started
        rate = state["frames"] / elapsed if elapsed > 0 else 0.0
        eta = (frames - state["frames"]) / rate if rate > 0 else float("nan")
        print(f"[{len(manifest['episodes'])}/{len(jobs)}] episode_{row['id']:04d} | ds{row['source_dataset']} ep{row['source_episode_id']:04d} "
              f"k={row['lead']} {row['branch']} | unsafe={row['reached_unsafe']} max_tilt={row['max_tilt_deg']:.0f} "
              f"replay_exact={row['replay_exact']} | {row['num_states']} states in {row['wall_seconds']:.1f}s | "
              f"{rate:.1f} states/s elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
        if time.perf_counter() - state["last_write"] >= MANIFEST_WRITE_INTERVAL_S:
            write_progress()

    env = None
    try:
        print(f"Generating {len(jobs)} branch episodes ({frames} states, ~{estimate / 1024**3:.1f} GiB) from {len(chosen)} events "
              f"with {effective_workers} worker(s). Progress is completion order; episode ids are fixed by branch_jobs.json.", flush=True)
        if effective_workers == 1:
            env, cache = ExplorationScene(), SourceCache()
            for job in jobs:
                record_result(generate_branch(env, job, cache.get(job["episode_dir"]), job["dataset_info"], camera))
        else:
            run_workers(jobs, _branch_worker, effective_workers, record_result)
        for d in split["datasets"]:
            if sha256(Path(d["path"]) / "manifest.json") != d["manifest_sha256"]:
                raise RuntimeError("A source manifest changed during generation")
        manifest.update(status="complete", summary=summarize(manifest["episodes"], time.perf_counter() - started, effective_workers))
        write_progress(final=True)
        return manifest
    except (Exception, KeyboardInterrupt) as error:
        _reconcile_interrupted(output_dir, manifest, error)
        manifest.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        manifest["execution"]["wall_seconds"] = time.perf_counter() - started
        write_json(output_dir / "manifest.json", manifest)
        raise
    finally:
        if env is not None:
            env.close()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, required=True, help="benchmark_split.json of the TRAINING material (danger_train_split_*)")
    p.add_argument("--output", type=Path, required=True, help="NEW dataset directory")
    p.add_argument("--part", choices=PARTS, default="test", help="test = the fine-tune training half (default), dev, or all")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--events", type=int, help="number of events (seeded round-robin over the direction x severity cells)")
    group.add_argument("--all-events", action="store_true", help="every fall event of the part, in (dataset, episode) order")
    p.add_argument("--leads", default=",".join(str(k) for k in DEFAULT_LEADS), help="restore leads k (frames before the onset)")
    p.add_argument("--branches", default=",".join(BRANCHES), help=f"comma-separated subset of {BRANCHES}")
    p.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS, help="branch length in control steps (default 200 = 10 s)")
    p.add_argument("--prefix", type=int, default=DEFAULT_PREFIX, help=f"recorded frames replayed before the restore frame (>= {MIN_PREFIX})")
    p.add_argument("--move-steps", type=int, default=DEFAULT_MOVE_STEPS, help="retreat/push_back moving steps before holding")
    p.add_argument("--move-speed", type=float, default=DEFAULT_MOVE_SPEED_MPS, help="retreat/push_back horizontal speed, m/s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    p.add_argument("--camera", choices=("front_pixels", "side_pixels"), default="front_pixels")
    p.add_argument("--strata", default=None, help="comma-separated stratum prefixes for --events (default: the 12 direction x severity cells)")
    p.add_argument("--order", choices=ORDERS, default="seeded")
    p.add_argument("--data-root", type=Path, default=None, help="directory holding the dataset folders if the split's paths do not resolve")
    p.add_argument("--plan-only", action="store_true", help="print the job count / byte estimate and write nothing")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.events is not None and a.events <= 0:
        raise ValueError("--events must be positive")
    strata = None if a.strata is None else [s.strip() for s in a.strata.split(",") if s.strip()]
    return collect_branches(a.output, a.split, part=a.part, events=a.events, all_events=a.all_events, leads=parse_leads(a.leads),
                            branches=parse_branches(a.branches), hold_steps=a.hold_steps, prefix=a.prefix, move_steps=a.move_steps,
                            move_speed=a.move_speed, seed=a.seed, workers=a.workers, camera=a.camera, strata=strata, order=a.order,
                            data_root=a.data_root, plan_only=a.plan_only)


if __name__ == "__main__":
    result = main()
    if "summary" in result:
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False), flush=True)
