"""Collect a small random-arm-start pilot: raw PNGs plus aligned numeric NPZ.

Fixed blocks, camera and yaw. No learned model, no randomized scene layouts,
no synthetic mid-trajectory teleports, and no automatic training split.
Each dataset directory is new; complete failed attempts are kept like successes.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import mujoco
import numpy as np
from PIL import Image

from auto_grasp import AutoGraspController
from random_starts import sample_valid_start, validate_start
from scene_config import IMAGE_HEIGHT, IMAGE_WIDTH, GRASP_MIN_NORMAL_FORCE_N
from three_block_scene import ThreeBlockScene


SCHEMA_VERSION = 2
DEFAULT_LOW = [0.28, -0.02, 0.24]
DEFAULT_HIGH = [0.32, 0.02, 0.28]
BOOL_FIELDS = (
    "unsafe", "episode_failed", "grasped", "left_contact", "right_contact",
    "goal_conditions_met", "task_success", "height_ok",
)
INTEGRATION_STATE = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path, content):
    """Atomically update only a manifest/metadata file in this newly owned run."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, ensure_ascii=False, default=json_value, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class CollectedScene(ThreeBlockScene):
    """Remember exactly the action whose normal physics step returned.

    Recording happens outside the controller's exception handler. An attempted
    action that never completed physics cannot quietly become a valid sample.
    """

    last_transition = None

    def step(self, action):
        self.last_transition = None
        actual_action = np.asarray(action, dtype=float).copy()
        result = super().step(actual_action)
        if len(result) != 5:
            raise RuntimeError("Unexpected reset in the middle of a trajectory")
        self.last_transition = {
            "action": actual_action,
            "physical_action": self.unnormalize_action(actual_action).copy(),
            "terminated": bool(result[2]), "truncated": bool(result[3]),
        }
        return result


def capture_state(env):
    task, safety = env.task_report(), env.safety_report()
    physics_state = np.empty(mujoco.mj_stateSize(env.model, INTEGRATION_STATE))
    mujoco.mj_getState(env.model, env.data, physics_state, INTEGRATION_STATE)
    state = {
        "time": float(env.data.time), "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(), "ctrl": env.data.ctrl.copy(),
        "physics_state": physics_state,
        "object_com_xyz": np.array([env.data.body(f"object_{i}").xipos.copy() for i in range(3)]),
        "object_quat_wxyz": np.array([env.data.body(f"object_{i}").xquat.copy() for i in range(3)]),
        "pinch_xyz": env.data.site_xpos[env._pinch_site_id].copy(),
        "pinch_rotation": env.data.site_xmat[env._pinch_site_id].reshape(3, 3).copy(),
        "guard_tilt_deg": np.array([g["tilt_deg"] for g in safety["guards"]]),
        "guard_displacement_xy_m": np.array([g["displacement_xy_m"] for g in safety["guards"]]),
        "finger_normal_force_n": np.array([task["left_normal_force_n"], task["right_normal_force_n"]]),
        "target_height_m": task["target_height_m"], "hold_seconds": task["hold_seconds"],
        "task_status": task["status"], "unsafe": safety["unsafe"],
    }
    state.update({key: task[key] for key in BOOL_FIELDS if key != "unsafe"})
    for key, value in state.items():
        if key != "task_status" and not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite state field: {key}")
    return state


class EpisodeRecorder:
    """One initial state and exactly one successor for each completed action."""

    schema_version = SCHEMA_VERSION
    terminal_phases = ("SUCCESS", "FAILED")

    def __init__(self, path, env, metadata, camera="front_pixels"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.images = self.path / "images"
        self.images.mkdir()
        self.env = env
        self.camera = camera
        self.metadata = {**metadata, "schema_version": self.schema_version, "status": "recording"}
        self.states, self.actions, self.physical_actions, self.phases = [], [], [], []
        self.terminated, self.truncated = [], []
        write_json(self.path / "metadata.json", self.metadata)
        self._append_state()

    def _append_state(self):
        state = self.capture_state()
        frame = self.env.render(camera=self.camera)
        if frame.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or frame.dtype != np.uint8:
            raise RuntimeError(f"Unexpected raw RGB image: {frame.shape} {frame.dtype}")
        # Direct renderer image: never use the pygame window/HUD or label text.
        Image.fromarray(frame).save(self.images / f"{len(self.states):06d}.png", compress_level=1)
        self.states.append(state)

    def capture_state(self):
        return capture_state(self.env)

    def append(self, transition, action_phase):
        action = np.asarray(transition["action"], dtype=float)
        physical = np.asarray(transition["physical_action"], dtype=float)
        if action.shape != (5,) or physical.shape != (5,) or not np.isfinite(action).all() or not np.isfinite(physical).all():
            raise RuntimeError("Invalid recorded action")
        if np.any(np.abs(action) > 1) or action[3] != 0 or physical[3] != 0:
            raise RuntimeError("Action violates normalized bounds or fixed yaw")
        expected = self.env.control_timestep()
        if not np.isclose(float(self.env.data.time) - self.states[-1]["time"], expected, atol=1e-8, rtol=0):
            raise RuntimeError("Action/state clock misalignment; refusing invalid trajectory")
        self._append_state()
        self.actions.append(action.copy())
        self.physical_actions.append(physical.copy())
        self.phases.append(str(action_phase))
        self.terminated.append(transition["terminated"])
        self.truncated.append(transition["truncated"])

    def finish(self, outcome):
        if len(self.states) != len(self.actions) + 1:
            raise RuntimeError("An episode must have T actions and T+1 states")
        if outcome["phase"] not in self.terminal_phases:
            raise RuntimeError("Cannot finalize a running attempt as complete")
        arrays = {key: np.asarray([state[key] for state in self.states]) for key in self.states[0]}
        arrays.update(
            actions=np.asarray(self.actions, dtype=float).reshape(-1, 5),
            physical_actions=np.asarray(self.physical_actions, dtype=float).reshape(-1, 5),
            action_phase=np.asarray(self.phases, dtype="U16"),
            terminated=np.asarray(self.terminated, dtype=bool),
            truncated=np.asarray(self.truncated, dtype=bool),
        )
        for key in BOOL_FIELDS:
            arrays[key] = arrays[key].astype(bool)
        with (self.path / "trajectory.npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        self.metadata.update(
            status="complete", num_actions=len(self.actions), num_states=len(self.states),
            outcome={**outcome, "training_data_collected": True},
            counts={key: int(arrays[key].sum()) for key in
                    ("unsafe", "goal_conditions_met", "task_success", "episode_failed")},
        )
        write_json(self.path / "metadata.json", self.metadata)
        return self.metadata

    def mark_error(self, error):
        self.metadata.update(status="error", error=str(error),
                             num_actions=len(self.actions), num_states=len(self.states))
        write_json(self.path / "metadata.json", self.metadata)


def source_fingerprints():
    root = Path(__file__).resolve().parent
    names = ("collect_grasp_trajs.py", "random_starts.py", "auto_grasp.py",
             "three_block_scene.py", "scene_config.py", "task_progress.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def episode_metadata(env, start_sample, episode_seed, camera, speed, timeout, max_steps):
    task, safety = env.task_report(), env.safety_report()
    return {
        "schema_version": SCHEMA_VERSION,
        "control_dt": env.control_timestep(), "physics_dt": env.physics_timestep(),
        "image": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "camera": camera, "overlay": False,
                  "shadows_enabled": env.shadows_enabled},
        "action_low": env.action_low.tolist(), "action_high": env.action_high.tolist(),
        "action_order": ["dx", "dy", "dz", "yaw_zero_only", "dgripper"],
        "action_units": "normalized [-1,1]; physical XYZ is metres, yaw radians, gripper unitless",
        "state_alignment": "state[i] -- actions[i] --> state[i+1]",
        "quaternion_order": "wxyz", "physics_state_spec": INTEGRATION_STATE,
        "fixed_yaw_deg": env.arm_fixed_yaw_deg,
        "unsafe_tilt_deg": env.unsafe_tilt_deg, "unsafe_displacement_m": env.unsafe_displacement_m,
        "displacement_check_enabled": env.displacement_check_enabled,
        "safety_scope": safety["safety_scope"],
        "target_height_threshold_m": task["height_threshold_m"],
        "required_hold_seconds": task["required_hold_seconds"], "min_normal_force_n": GRASP_MIN_NORMAL_FORCE_N,
        "guard_initial_com_xy_m": [g["initial_com_xy_m"] for g in safety["guards"]],
        "initial_block_positions_xyz": [env.data.body(f"object_{i}").xipos.tolist() for i in range(3)],
        "blocks": env.blocks, "start_sample": start_sample,
        "sampling_seed": episode_seed, "environment_reset_seed": 0,
        "controller": {"type": "scripted_simulator_truth", "speed_mps": speed,
                       "phase_timeout_s": timeout, "max_control_steps": max_steps},
        "label_notes": {
            "unsafe": "current guard tilt; reset-relative COM shift only if displacement_check_enabled; not episode_failed",
            "goal_conditions_met": "instantaneous dual contact and COM height; not 1-second completion",
            "task_success": "history-dependent hold duration and no previous guard hazard",
            "episode_failed": "history latch; may be true even if current unsafe is false",
        },
    }


def collect_dataset(output_dir, n_trajectories=20, seed=20260913,
                    start_low=DEFAULT_LOW, start_high=DEFAULT_HIGH,
                    speed_mps=0.06, phase_timeout_s=15.0, max_steps=2000,
                    max_start_attempts=50, camera="front_pixels",
                    grasp_offset_x_m=0.0, grasp_offset_x_range=None, close_timeout_s=None):
    if not isinstance(n_trajectories, int) or n_trajectories <= 0:
        raise ValueError("n_trajectories must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(max_start_attempts, int) or max_start_attempts <= 0:
        raise ValueError("max_start_attempts must be a positive integer")
    if not math.isfinite(speed_mps) or not 0 < speed_mps <= 0.1:
        raise ValueError("speed_mps must be in (0, 0.1]")
    if not math.isfinite(phase_timeout_s) or phase_timeout_s <= 0:
        raise ValueError("phase_timeout_s must be finite and positive")
    if not math.isfinite(grasp_offset_x_m) or abs(grasp_offset_x_m) > 0.06:
        raise ValueError("grasp_offset_x_m must be finite and within +/-0.06 m")
    offset_range = None
    if grasp_offset_x_range is not None:
        offset_range = np.asarray(grasp_offset_x_range, dtype=float)
        if (offset_range.shape != (2,) or not np.isfinite(offset_range).all() or
                offset_range[0] >= offset_range[1] or np.any(np.abs(offset_range) > 0.06)):
            raise ValueError("grasp_offset_x_range requires MIN < MAX, both finite and within +/-0.06 m")
        if grasp_offset_x_m != 0:
            raise ValueError("Use a fixed grasp offset OR an offset range, not both")
    if close_timeout_s is not None and (not math.isfinite(close_timeout_s) or close_timeout_s <= 0):
        raise ValueError("close_timeout_s must be finite and positive, or None")
    low, high = np.asarray(start_low, dtype=float), np.asarray(start_high, dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low >= high):
        raise ValueError("start bounds must be finite ordered three-vectors with low < high")
    if np.any(low < [0.25, -0.35, 0.20]) or np.any(high > [0.60, 0.35, 0.35]):
        raise ValueError("start bounds exceed the scene's supported initialization workspace")
    if camera not in ("front_pixels", "side_pixels"):
        raise ValueError("Unsupported camera")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "collecting",
        "requested_episodes": n_trajectories, "seed": seed,
        "start_low": low.tolist(), "start_high": high.tolist(),
        "fixed_object_layout": True, "no_action_noise": True,
        "grasp_perturbation": {
            "type": "uniform_world_x_aim_offset" if offset_range is not None else "fixed_world_x_aim_offset",
            "fixed_offset_x_m": None if offset_range is not None else float(grasp_offset_x_m),
            "range_x_m": None if offset_range is None else offset_range.tolist(),
            "close_timeout_s": close_timeout_s,
            "objects_teleported": False, "labels_overridden": False,
        },
        "source_sha256": source_fingerprints(),
        "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                     "numpy": np.__version__, "pillow": version("pillow"),
                     "ogbench": version("ogbench"), "stable-worldmodel": version("stable-worldmodel")},
        "episodes": [],
    }
    write_json(output_dir / "manifest.json", manifest)
    env = CollectedScene()
    recorder = None
    try:
        seeds = np.random.SeedSequence(seed).spawn(n_trajectories)
        for index, child_seed in enumerate(seeds):
            recorder = None
            episode_seed = int(child_seed.generate_state(1, dtype=np.uint64)[0])
            offset_seed = None
            offset_x = float(grasp_offset_x_m)
            if offset_range is not None:
                # Independent RNG: enabling aim errors must not change the
                # sampled robot starts for the same dataset seed.
                offset_seed = int(child_seed.spawn(1)[0].generate_state(1, dtype=np.uint64)[0])
                offset_x = float(np.random.default_rng(offset_seed).uniform(*offset_range))
            start_sample = sample_valid_start(
                env, np.random.default_rng(episode_seed), max_attempts=max_start_attempts,
                low=low, high=high,
            )
            controller = AutoGraspController(env, speed_mps, phase_timeout_s,
                                             offset_x, close_timeout_s)
            # Controller reset restores the same configured candidate. Verify
            # the actual recorded reset too, not just an earlier sampling probe.
            recorded_validity = validate_start(env, start_sample["requested_start_xyz"])
            if not recorded_validity["valid"]:
                raise RuntimeError(f"Recorded reset failed validity check: {recorded_validity}")
            start_sample["recorded_reset_validity"] = recorded_validity
            name = f"episode_{index:04d}"
            metadata = episode_metadata(env, start_sample, episode_seed, camera, speed_mps, phase_timeout_s, max_steps)
            metadata["controller"].update(grasp_offset_x_m=offset_x, close_timeout_s=close_timeout_s)
            metadata["grasp_perturbation"] = {
                **manifest["grasp_perturbation"], "actual_offset_x_m": offset_x,
                "sampling_seed": offset_seed,
                "original_target_xyz_m": controller.original_target_xyz.tolist(),
                "grasp_target_xyz_m": controller.grasp_target_xyz.tolist(),
            }
            recorder = EpisodeRecorder(
                output_dir / name, env, metadata, camera,
            )
            while not controller.finished:
                if controller.control_steps >= max_steps:
                    controller._transition("FAILED", "Collection control-step limit reached.")
                    break
                phase_before = controller.phase
                time_before = float(env.data.time)
                env.last_transition = None
                controller.step()
                if env.last_transition is not None:
                    recorder.append(env.last_transition, phase_before)
                elif float(env.data.time) != time_before:
                    raise RuntimeError("Physics advanced without a completed recorded action; episode quarantined")
                elif not controller.finished:
                    raise RuntimeError("Controller made no progress without reaching an outcome")
            if (controller.failure_reason or "").startswith("Robot control error:"):
                raise RuntimeError(controller.failure_reason)
            metadata = recorder.finish(controller.report())
            row = {
                "id": index, "path": name, "status": "complete", "phase": controller.phase,
                "num_actions": metadata["num_actions"], "num_states": metadata["num_states"],
                "sampling_seed": episode_seed, "requested_start_xyz": start_sample["requested_start_xyz"],
                "grasp_offset_x_m": offset_x,
                "counts": metadata["counts"], "failure_reason": controller.failure_reason,
            }
            manifest["episodes"].append(row)
            write_json(output_dir / "manifest.json", manifest)
            print(f"[{index + 1}/{n_trajectories}] {controller.phase} | "
                  f"actions={row['num_actions']} frames={row['num_states']} | "
                  f"start={np.round(env.arm_start, 4).tolist()} | X aim offset={100 * offset_x:+.2f} cm", flush=True)
        episodes = manifest["episodes"]
        manifest.update(
            status="complete",
            summary={
                "episodes": len(episodes), "successful_episodes": sum(e["phase"] == "SUCCESS" for e in episodes),
                "failed_episodes": sum(e["phase"] == "FAILED" for e in episodes),
                "total_actions": sum(e["num_actions"] for e in episodes),
                "total_states": sum(e["num_states"] for e in episodes),
                **{key + "_states": sum(e["counts"][key] for e in episodes) for key in
                   ("unsafe", "goal_conditions_met", "task_success", "episode_failed")},
                "scope": "small random-arm-start pilot; fixed blocks; not a generalization guarantee or training-ready coverage",
            },
        )
        write_json(output_dir / "manifest.json", manifest)
        return manifest
    except (Exception, KeyboardInterrupt) as exc:
        if recorder is not None and recorder.metadata["status"] != "complete":
            recorder.mark_error(exc)
        manifest.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
        write_json(output_dir / "manifest.json", manifest)
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New dataset directory; never overwrite")
    parser.add_argument("--n-trajectories", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--start-low", nargs=3, type=float, default=DEFAULT_LOW, metavar=("X", "Y", "Z"))
    parser.add_argument("--start-high", nargs=3, type=float, default=DEFAULT_HIGH, metavar=("X", "Y", "Z"))
    parser.add_argument("--speed", type=float, default=0.06)
    parser.add_argument("--phase-timeout", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-start-attempts", type=int, default=50)
    parser.add_argument("--camera", choices=("front_pixels", "side_pixels"), default="front_pixels")
    offsets = parser.add_mutually_exclusive_group()
    offsets.add_argument("--grasp-offset-x", type=float, default=0.0, help="Fixed world-X aim offset in metres; no object movement")
    offsets.add_argument("--grasp-offset-x-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="Sample one fixed X aim offset per episode, in metres")
    parser.add_argument("--close-timeout", type=float, help="Optional shorter CLOSE wait; normal phase timeout remains an upper bound")
    args = parser.parse_args()
    manifest = collect_dataset(
        args.output_dir, args.n_trajectories, args.seed, args.start_low, args.start_high,
        args.speed, args.phase_timeout, args.max_steps, args.max_start_attempts, args.camera,
        args.grasp_offset_x, args.grasp_offset_x_range, args.close_timeout,
    )
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    print(f"Saved dataset: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
