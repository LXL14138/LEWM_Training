"""Collect guided-random SIMULATION trajectories, including unsuccessful ones.

No GUI, no training, no real robot. Four balanced attempt modes by default.
Use inspect_grasp_dataset.py for file/label validation and recorded replay.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import signal
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np

from benchmark_exploration import BenchmarkController
from benchmark_plan import BENCH_INTERACTION_REVISION, read_schedule
from collect_grasp_trajs import EpisodeRecorder, episode_metadata, source_fingerprints, write_json
from exploration import ExplorationController, ExplorationScene, MODES, GOAL_LOW, GOAL_HIGH, INTERACTION_REVISION
from random_starts import sample_valid_start, SUPPORTED_START_LOW, SUPPORTED_START_HIGH
from scene_config import ARM_START_XYZ, IMAGE_HEIGHT, IMAGE_WIDTH
from collection_workers import CHILD_THREAD_LIMITS, run_workers

DATASET_KIND = "guided_exploration_v1"
CLASS_NAMES = {"0": "safe and instantaneous hover conditions met",
               "1": "currently unsafe", "2": "safe and hover conditions not met"}


def blocks_in_frame(env, camera):
    """Whole-box frustum check, not an occlusion/visibility guarantee."""
    cam = env.model.camera(camera).id
    pos = env.data.cam_xpos[cam]
    rot = env.data.cam_xmat[cam].reshape(3, 3)
    focal = 0.5 * IMAGE_HEIGHT / np.tan(np.deg2rad(env.model.cam_fovy[cam] / 2))
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    visible = []
    for i in range(3):
        geom = env.model.geom(f"object_{i}").id
        corners = (signs * env.model.geom_size[geom]) @ env.data.geom_xmat[geom].reshape(3, 3).T
        xyz = (corners + env.data.geom_xpos[geom] - pos) @ rot
        depth = -xyz[:, 2]
        if np.any(depth <= 0):
            visible.append(False)
            continue
        pixels = np.column_stack((IMAGE_WIDTH / 2 + focal * xyz[:, 0] / depth,
                                  IMAGE_HEIGHT / 2 - focal * xyz[:, 1] / depth))
        visible.append(bool(np.all(pixels >= 0) and np.all(pixels < [IMAGE_WIDTH, IMAGE_HEIGHT])))
    return np.array(visible, dtype=bool)


class ExplorationRecorder(EpisodeRecorder):
    schema_version = 3
    terminal_phases = ("DONE",)

    def capture_state(self):
        state = self.env.exploration_state()
        state["block_in_frame"] = blocks_in_frame(self.env, self.camera)
        return state

    def finish(self, outcome):
        labels = np.array([s["class_label"] for s in self.states])
        summary = {
            "class_counts": {str(k): int((labels == k).sum()) for k in range(3)},
            "grasped_frames": sum(bool(s["grasped"]) for s in self.states),
            "hover_success_frames": sum(bool(s["hover_success"]) for s in self.states),
            "guard_contact_steps": sum(bool(np.any(s["guard_contact_peak_last_step_n"] > 1e-3))
                                       for s in self.states[1:]),
            "out_of_frame_states": sum(not bool(np.all(s["block_in_frame"])) for s in self.states),
            "green_guard_contact_steps": sum(bool(np.any(s["green_guard_contact_peak_last_step_n"] > 1e-3))
                                               for s in self.states[1:]),
        }
        self.metadata["exploration_summary"] = summary
        return super().finish(outcome)

    def mark_error(self, error):
        # Keep observed complete transitions for diagnosis, never mark them as
        # training-ready. The error metadata makes the validator reject them.
        count = min(len(self.states), len(self.actions) + 1)
        if count:
            partial = {k: np.asarray([s[k] for s in self.states[:count]]) for k in self.states[0]}
            partial["actions"] = np.asarray(self.actions[:count - 1], dtype=float).reshape(-1, 5)
            partial["action_phase"] = np.asarray(self.phases[:count - 1], dtype="U16")
            path = self.path / "partial_trajectory.npz"
            if not path.exists():
                with path.open("xb") as handle:
                    np.savez_compressed(handle, **partial)
        super().mark_error(error)


def _episode_row(index, metadata):
    outcome = metadata["outcome"]
    return {"id": index, "path": f"episode_{index:04d}", "status": "complete", "phase": "DONE",
            "mode": outcome["mode"], "num_actions": metadata["num_actions"],
            "num_states": metadata["num_states"], "stop_reason": outcome["stop_reason"],
            "ever_lifted": outcome["ever_lifted"], "ever_hover_success": outcome["ever_hover_success"],
            "worker_pid": metadata.get("execution", {}).get("pid"),
            "wall_seconds": metadata.get("execution", {}).get("wall_seconds"),
            **metadata["exploration_summary"]}


def _collect_episode(env, job, stop=None):
    """Shared single-episode path: identical physics in serial and parallel."""
    recorder = None
    started = time.perf_counter()
    camera = job["camera"]
    try:
        if stop is not None and stop.is_set():
            raise InterruptedError("Collection cancelled before episode start")
        dt = env.control_timestep()
        max_steps = max(1, int(math.floor(job["max_seconds"] / dt + 1e-9)))
        start = sample_valid_start(env, np.random.default_rng(job["start_seed"]),
                                   low=job["start_low"], high=job["start_high"])
        intent = job.get("intent")
        rng = np.random.default_rng(job["action_seed"])
        if intent is None:
            controller = ExplorationController(env, rng, job["mode"], max_steps=max_steps, max_attempts=job["max_attempts"])
            controller_meta = {"type": "guided_random_simulator_truth", "mode": job["mode"],
                               "interaction_revision": INTERACTION_REVISION}
        else:
            # Scheduled benchmark attempt; the original four modes never take this path.
            controller = BenchmarkController(env, rng, job["mode"], intent, max_steps=max_steps, max_attempts=job["max_attempts"])
            controller_meta = {"type": "danger_benchmark_simulator_truth", "mode": job["mode"],
                               "interaction_revision": controller.interaction_revision, "benchmark_intent": intent}
        metadata = episode_metadata(env, start, job["action_seed"], camera, 0.07, 6.0, max_steps)
        metadata.update(schema_version=3, dataset_kind=DATASET_KIND,
                        start_seed=job["start_seed"], action_seed=job["action_seed"],
                        execution={"pid": os.getpid()},
                        controller={**controller_meta,
                                    "max_steps": max_steps, "max_attempts": job["max_attempts"],
                                    "goal_sampling_attempts": controller.goal_sampling_attempts},
                        hover_goal_xyz=env.hover_goal.tolist(), hover_tolerance_m=env.hover_tolerance_m,
                        hover_speed_limit_mps=env.hover_speed_limit_mps,
                        hover_required_seconds=env.hover_required_seconds,
                        class_names=CLASS_NAMES,
                        visibility_note="block_in_frame checks box corners; does not detect occlusion",
                        guard_contact_note="robot/green-to-red normal-force peak across the preceding control step")
        metadata["label_notes"].update(
            class_label="current unsafe takes priority; class 0 is instantaneous hover conditions, not one-second completion",
            hover_success="goal-conditioned one-second substep hold with no historical guard hazard",
            task_success="LEGACY lift-height success only; use hover_success for the new task",
            goal_conditions_met="LEGACY grip and lift height; use hover_conditions_met for the new task")
        cam = env.model.camera(camera).id
        metadata["image"].update(camera_position_world=env.data.cam_xpos[cam].tolist(),
                                 camera_rotation_world=env.data.cam_xmat[cam].tolist(),
                                 fovy_deg=float(env.model.cam_fovy[cam]))
        recorder = ExplorationRecorder(Path(job["output_dir"]) / f"episode_{job['id']:04d}",
                                       env, metadata, camera)
        while not controller.finished:
            if stop is not None and stop.is_set():
                raise InterruptedError("Collection cancelled by parent")
            before = float(env.data.time)
            env.last_transition = None
            phase = controller.step()
            if phase is not None:
                if env.last_transition is None:
                    raise RuntimeError("Missing executed transition")
                recorder.append(env.last_transition, phase)
            elif float(env.data.time) != before or not controller.finished:
                raise RuntimeError("Unrecorded physics or stalled exploration")
        recorder.metadata["execution"]["wall_seconds"] = time.perf_counter() - started
        result = recorder.finish(controller.report())
        return _episode_row(job["id"], result)
    except (Exception, KeyboardInterrupt) as error:
        if recorder is not None and recorder.metadata["status"] != "complete":
            recorder.mark_error(error)
        raise


def _exploration_worker(pipe, stop):
    # Only the parent handles Ctrl+C. Each worker has its own renderer/physics.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = None
    try:
        while True:
            job = pipe.recv()
            if job is None or stop.is_set():
                return
            try:
                if env is None:
                    env = ExplorationScene()
                row = _collect_episode(env, job, stop)
            except (Exception, KeyboardInterrupt) as error:
                pipe.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
                return
            pipe.send({"status": "complete", "row": row})
    finally:
        if env is not None:
            env.close()
        pipe.close()


def _reconcile_interrupted(output_dir, manifest, error):
    """Called only after all our workers stop; preserve completed episodes."""
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
                complete[index] = _episode_row(index, metadata)
            else:
                if metadata.get("status") != "error":
                    metadata.update(status="error", error=f"Collection interrupted: {error}")
                    write_json(path, metadata)
                incomplete.append({"id": index, "path": episode.name, "status": "error"})
        except (OSError, ValueError, KeyError, TypeError) as read_error:
            incomplete.append({"id": index, "path": episode.name, "status": "error",
                               "error": f"Cannot reconcile metadata: {read_error}"})
    manifest["episodes"] = sorted(complete.values(), key=lambda row: row["id"])
    manifest["incomplete_episodes"] = incomplete


def collect_exploration(output_dir, n_trajectories=None, seed=20260920, mode="mixed",
                        max_seconds=30.0, max_attempts=2, camera="front_pixels",
                        start_low=None, start_high=None, workers=1, schedule=None):
    plan = None
    if schedule is not None:
        # A benchmark schedule fixes the attempt count and every episode's mode/intent.
        plan = read_schedule(schedule)
        if mode != "mixed":
            raise ValueError("A schedule decides every episode's mode; do not combine it with --mode")
        if n_trajectories is None:
            n_trajectories = len(plan["entries"])
        elif n_trajectories != len(plan["entries"]):
            raise ValueError("n_trajectories must equal the number of schedule entries")
    elif n_trajectories is None:
        n_trajectories = 40
    if type(n_trajectories) is not int or n_trajectories <= 0:
        raise ValueError("n_trajectories must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer from 1 to 4")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be an integer from 1 to 5")
    if isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or not 0.05 <= max_seconds <= 120:
        raise ValueError("max_seconds must be finite, from 0.05 to 120")
    if mode not in (*MODES, "mixed") or camera not in ("front_pixels", "side_pixels"):
        raise ValueError("Unsupported mode or camera")
    center = np.asarray(ARM_START_XYZ, dtype=float)
    low = np.asarray(center + [-0.015, -0.01, 0] if start_low is None else start_low, dtype=float)
    high = np.asarray(center + [0.015, 0.01, 0.02] if start_high is None else start_high, dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low > high):
        raise ValueError("start bounds must be finite ordered three-vectors")
    if np.any(low < SUPPORTED_START_LOW) or np.any(high > SUPPORTED_START_HIGH):
        raise ValueError("start bounds exceed the supported initialization workspace")
    output_dir = Path(output_dir).resolve()
    root = Path(__file__).resolve().parent
    fingerprints = source_fingerprints()
    for name in ("collect_exploration_trajs.py", "exploration.py", "inspect_grasp_dataset.py",
                 "exploration_validation.py", "collection_workers.py",
                 "benchmark_exploration.py", "benchmark_plan.py"):
        fingerprints[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    children = np.random.SeedSequence(seed).spawn(n_trajectories + 1)
    if plan is None:
        modes = list(np.resize(np.asarray(MODES), n_trajectories)) if mode == "mixed" else [mode] * n_trajectories
        np.random.default_rng(children[-1]).shuffle(modes)
    else:
        modes = [entry["mode"] for entry in plan["entries"]]
    # Sample everything by episode id BEFORE dispatch. Completion order and
    # worker count must not change the sequence of RNG draws for any episode.
    jobs = []
    for index, child in enumerate(children[:-1]):
        start_seed, action_seed = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in child.spawn(2)]
        jobs.append({"id": index, "output_dir": str(output_dir), "mode": str(modes[index]),
                     "start_seed": start_seed, "action_seed": action_seed,
                     "start_low": low.tolist(), "start_high": high.tolist(), "camera": camera,
                     "max_seconds": max_seconds, "max_attempts": max_attempts,
                     "intent": None if plan is None else plan["entries"][index]["intent"]})
    effective_workers = min(workers, n_trajectories)
    manifest = {"schema_version": 3, "dataset_kind": DATASET_KIND, "status": "collecting",
                "interaction_revision": INTERACTION_REVISION if plan is None else BENCH_INTERACTION_REVISION,
                "benchmark_schedule": None if plan is None else {
                    "path": str(Path(schedule).resolve()), "sha256": plan["sha256"], "seed": plan["seed"],
                    "counts": plan["counts"], "attempts": len(plan["entries"]),
                    "note": "cells are attempt intents; measure the reached stratum with wm_training.danger_strata"},
                "requested_episodes": n_trajectories, "seed": seed, "episodes": [],
                "mode_schedule": modes, "start_low": low.tolist(), "start_high": high.tolist(),
                "fixed_object_layout": True, "objects_teleported_during_episode": False,
                "goal_bounds_xyz": [GOAL_LOW.tolist(), GOAL_HIGH.tolist()],
                "class_names": CLASS_NAMES, "max_seconds": max_seconds,
                "execution": {"workers_requested": workers, "workers_used": effective_workers,
                              "start_method": "spawn" if effective_workers > 1 else "in_process",
                              "child_thread_limits": CHILD_THREAD_LIMITS if effective_workers > 1 else {},
                              "render_environment": {key: os.environ.get(key) for key in
                                  ("MUJOCO_GL", "GALLIUM_DRIVER", "MESA_D3D12_DEFAULT_ADAPTER_NAME",
                                   "LIBGL_ALWAYS_SOFTWARE", "LD_LIBRARY_PATH")}},
                "source_sha256": fingerprints,
                "versions": {k: version(k) for k in ("numpy", "mujoco", "ogbench", "stable-worldmodel")}}
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", manifest)
    started = time.perf_counter()

    def record_result(row):
        manifest["episodes"].append(row)
        manifest["episodes"].sort(key=lambda r: r["id"])
        write_json(output_dir / "manifest.json", manifest)
        elapsed = time.perf_counter() - started
        print(f"[{len(manifest['episodes'])}/{n_trajectories}] episode_{row['id']:04d} | "
              f"{row['mode']} | {row['stop_reason']} | actions={row['num_actions']} "
              f"lifted={row['ever_lifted']} labels={row['class_counts']} "
              f"contact_steps={row['guard_contact_steps']} | elapsed={elapsed:.1f}s", flush=True)

    env = None
    try:
        print(f"Collecting {n_trajectories} episodes with {effective_workers} worker(s). "
              "Progress is completion order; episode ids remain fixed.", flush=True)
        if effective_workers == 1:
            env = ExplorationScene()
            for job in jobs:
                record_result(_collect_episode(env, job))
        else:
            run_workers(jobs, _exploration_worker, effective_workers, record_result)
        rows = manifest["episodes"]
        summary = {"episodes": len(rows), "total_actions": sum(r["num_actions"] for r in rows),
                   "total_states": sum(r["num_states"] for r in rows),
                   "lifted_episodes": sum(r["ever_lifted"] for r in rows),
                   "hover_success_episodes": sum(r["ever_hover_success"] for r in rows),
                   "class_counts": {str(k): sum(r["class_counts"][str(k)] for r in rows) for k in range(3)},
                   "guard_contact_steps": sum(r["guard_contact_steps"] for r in rows),
                   "out_of_frame_states": sum(r["out_of_frame_states"] for r in rows),
                   "wall_seconds": time.perf_counter() - started,
                   "scope": "coverage diagnostic only, not evidence of training sufficiency"}
        summary["actions_per_wall_second"] = summary["total_actions"] / summary["wall_seconds"]
        manifest.update(status="complete", summary=summary)
        write_json(output_dir / "manifest.json", manifest)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW directory; existing data never overwritten")
    parser.add_argument("--n-trajectories", type=int, default=None,
                        help="default 40; with --schedule it must be omitted or equal the schedule length")
    parser.add_argument("--schedule", type=Path, default=None,
                        help="benchmark_plan schedule json: fixes attempt count, modes and intents (interaction_revision 3)")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--mode", choices=("mixed", *MODES), default="mixed")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1,
                        help="Independent collection processes (1-4); default 1. GPU settings are inherited.")
    parser.add_argument("--camera", choices=("front_pixels", "side_pixels"), default="front_pixels")
    parser.add_argument("--start-low", nargs=3, type=float)
    parser.add_argument("--start-high", nargs=3, type=float)
    args = parser.parse_args()
    result = collect_exploration(**vars(args))
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved exploration dataset: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
