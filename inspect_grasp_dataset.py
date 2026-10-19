"""Inspect/replay grasp (schema 1/2) and exploration (schema 3) episodes read-only.

Validation checks the saved format and label consistency, not visual ground truth,
model quality, or whether the data is sufficient for training. Physics is NOT run
during replay. In particular 20-Hz observations cannot reconstruct every 2-ms
contact interruption: the simulator's saved hold timer and failure latch retain
that substep history. Never load arrays using pickle. Explicit --report and
--snapshot-dir options export NEW files/directories; existing output is preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image


BOOL_FIELDS = (
    "unsafe", "episode_failed", "grasped", "left_contact", "right_contact",
    "goal_conditions_met", "task_success", "height_ok",
)
STATE_TAILS = {
    "time": (), "object_com_xyz": (3, 3), "object_quat_wxyz": (3, 4),
    "pinch_xyz": (3,), "hold_seconds": (), "target_height_m": (),
    "guard_tilt_deg": (2,), "guard_displacement_xy_m": (2,),
    "finger_normal_force_n": (2,),
}
KNOWN_PHASES = {"SETTLE", "APPROACH", "DESCEND", "CLOSE", "LIFT", "HOLD", "SUCCESS", "FAILED"}


def _json(path):
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _count(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _numeric(array, name):
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")


def _vector(value, shape, name):
    array = np.asarray(value)
    _numeric(array, name)
    if array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    return array.astype(float)


def _same(actual, expected, name):
    if not np.array_equal(actual, expected):
        raise ValueError(f"{name} does not match the saved physical measurements/thresholds")


def _close(actual, expected, name, atol=1e-8):
    if not np.allclose(actual, expected, atol=atol, rtol=1e-7):
        raise ValueError(f"{name} is inconsistent")


def _check_episode(path):
    metadata = _json(path / "metadata.json")
    schema_version = metadata.get("schema_version")
    if type(schema_version) is not int or schema_version not in (1, 2, 3):
        raise ValueError("unsupported or missing episode schema_version (expected 1, 2 or 3)")
    exploration = schema_version == 3
    if metadata.get("status") != "complete":
        raise ValueError("episode status is not complete; partial data is not a validated episode")
    action_count = _count(metadata.get("num_actions"), "num_actions")
    state_count = _count(metadata.get("num_states"), "num_states", minimum=1)
    if state_count != action_count + 1:
        raise ValueError("num_states must equal num_actions + 1 (T+1 images/states for T actions)")
    dt = _positive(metadata.get("control_dt"), "control_dt")
    physics_dt = _positive(metadata.get("physics_dt"), "physics_dt")
    if physics_dt > dt or not np.isclose(dt / physics_dt, round(dt / physics_dt), atol=1e-8):
        raise ValueError("control_dt must be a positive integer multiple of physics_dt")
    image_info = metadata.get("image")
    if not isinstance(image_info, dict) or image_info.get("overlay") is not False:
        raise ValueError("image metadata must explicitly declare overlay=false")
    width = _count(image_info.get("width"), "image.width", minimum=1)
    height = _count(image_info.get("height"), "image.height", minimum=1)
    if image_info.get("camera") not in ("front_pixels", "side_pixels"):
        raise ValueError("image.camera must identify the saved scene camera")
    if schema_version >= 2 and type(image_info.get("shadows_enabled")) is not bool:
        raise ValueError("image.shadows_enabled must be a boolean for schema 2")
    low = _vector(metadata.get("action_low"), (5,), "action_low")
    high = _vector(metadata.get("action_high"), (5,), "action_high")
    if np.any(low > high) or np.any(low > 0) or np.any(high < 0):
        raise ValueError("action_low/high must be ordered physical ranges containing zero")
    if not np.isfinite(metadata.get("fixed_yaw_deg", float("nan"))):
        raise ValueError("fixed_yaw_deg must be finite")
    tilt_limit = _positive(metadata.get("unsafe_tilt_deg"), "unsafe_tilt_deg")
    shift_limit = _positive(metadata.get("unsafe_displacement_m"), "unsafe_displacement_m")
    # Old schema-1 recordings always used tilt OR displacement. Never reinterpret
    # their labels using today's scene_config defaults.
    shift_enabled = True if schema_version == 1 else metadata.get("displacement_check_enabled")
    if type(shift_enabled) is not bool:
        raise ValueError("displacement_check_enabled must be a boolean for schema 2")
    if schema_version == 1 and metadata.get("displacement_check_enabled", True) is not True:
        raise ValueError("schema 1 requires displacement safety; disabled checks require schema 2")
    expected_scope = "guard_tilt_or_displacement" if shift_enabled else "guard_tilt_only"
    if "safety_scope" in metadata and metadata["safety_scope"] != expected_scope:
        raise ValueError("safety_scope contradicts displacement_check_enabled")
    height_limit = _positive(metadata.get("target_height_threshold_m"), "target_height_threshold_m")
    hold_required = _positive(metadata.get("required_hold_seconds"), "required_hold_seconds")
    force_limit = _positive(metadata.get("min_normal_force_n"), "min_normal_force_n")
    initial_guard_xy = _vector(metadata.get("guard_initial_com_xy_m"), (2, 2), "guard_initial_com_xy_m")
    _vector(metadata.get("initial_block_positions_xyz"), (3, 3), "initial_block_positions_xyz")
    if not isinstance(metadata.get("start_sample"), dict):
        raise ValueError("start_sample must record the start sampling metadata")
    outcome = metadata.get("outcome")
    allowed_outcomes = ("DONE",) if exploration else ("SUCCESS", "FAILED")
    if not isinstance(outcome, dict) or outcome.get("phase") not in allowed_outcomes:
        raise ValueError(f"complete episode outcome.phase must be one of {allowed_outcomes}")

    # Reading every archive member also rejects any hidden pickled/object array.
    with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for name, array in arrays.items():
        if array.dtype.kind in "fiu":
            _numeric(array, name)
        elif array.dtype.kind not in "bU":
            raise ValueError(f"{name} has an unsupported dtype (only real, boolean and Unicode arrays are permitted)")
    required = set(STATE_TAILS) | set(BOOL_FIELDS) | {"qpos", "qvel", "ctrl", "actions", "physical_actions", "action_phase"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"missing trajectory arrays: {sorted(missing)}")
    for name, tail in STATE_TAILS.items():
        array = arrays[name]
        if array.shape != (state_count,) + tail:
            raise ValueError(f"{name} shape must be {(state_count,) + tail}, got {array.shape}")
        _numeric(array, name)
    if arrays["time"].dtype != np.dtype("float64"):
        raise ValueError("time must use float64")
    for name in ("qpos", "qvel", "ctrl"):
        array = arrays[name]
        if array.ndim != 2 or array.shape[0] != state_count or array.shape[1] == 0:
            raise ValueError(f"{name} must have shape (num_states, nonzero simulator dimension)")
        _numeric(array, name)
    if "physics_state" in arrays:
        physics_state = arrays["physics_state"]
        if physics_state.ndim != 2 or physics_state.shape[0] != state_count or physics_state.shape[1] == 0:
            raise ValueError("physics_state must have shape (num_states, nonzero state dimension)")
        _numeric(physics_state, "physics_state")
    if "pinch_rotation" in arrays:
        rotation = arrays["pinch_rotation"]
        if rotation.shape != (state_count, 3, 3):
            raise ValueError("pinch_rotation must have shape (num_states, 3, 3)")
        _numeric(rotation, "pinch_rotation")
        _close(rotation @ rotation.transpose(0, 2, 1), np.eye(3), "pinch_rotation orthonormality", atol=1e-5)
        _close(np.linalg.det(rotation), 1.0, "pinch_rotation determinant", atol=1e-5)
    for name in ("terminated", "truncated"):
        if name in arrays and (arrays[name].shape != (action_count,) or arrays[name].dtype != np.dtype(bool)):
            raise ValueError(f"{name} must be a boolean transition array of shape (num_actions,)")
    for name in BOOL_FIELDS:
        if arrays[name].shape != (state_count,) or arrays[name].dtype != np.dtype(bool):
            raise ValueError(f"{name} must be a boolean array of shape (num_states,)")
    for name in ("actions", "physical_actions"):
        if arrays[name].shape != (action_count, 5):
            raise ValueError(f"{name} shape must be (num_actions, 5)")
        _numeric(arrays[name], name)
    phases = arrays["action_phase"]
    if phases.shape != (action_count,) or phases.dtype.kind != "U":
        raise ValueError("action_phase must be a Unicode string array of shape (num_actions,)")
    if exploration:
        from exploration_validation import EXPLORATION_PHASES
        allowed_phases = EXPLORATION_PHASES
    else:
        allowed_phases = KNOWN_PHASES - {"SUCCESS", "FAILED"}
    if any(phase not in allowed_phases for phase in phases):
        raise ValueError("action_phase contains an unknown or frozen terminal phase")
    times = arrays["time"]
    if abs(times[0]) > 1e-8:
        raise ValueError("episode must start at simulator time zero, not concatenate episodes")
    _close(np.diff(times), np.full(action_count, dt), "time intervals", atol=1e-8)
    if np.any(np.abs(arrays["actions"]) > 1 + 1e-8):
        raise ValueError("normalized actions must stay within [-1, 1]")
    _close(arrays["actions"][:, 3], 0.0, "normalized yaw action", atol=1e-12)
    _close(arrays["physical_actions"][:, 3], 0.0, "physical yaw action", atol=1e-12)
    expected_physical = low + (arrays["actions"] + 1.0) * (high - low) / 2
    _close(arrays["physical_actions"], expected_physical, "physical action scaling")
    quat = arrays["object_quat_wxyz"]
    _close(np.linalg.norm(quat, axis=-1), 1.0, "object quaternion unit norm", atol=1e-5)
    # Local z-axis dot world z for wxyz quaternions, identical to R[2,2].
    expected_tilt = np.rad2deg(np.arccos(np.clip(1 - 2 * (quat[:, 1:, 1] ** 2 + quat[:, 1:, 2] ** 2), -1, 1)))
    _close(arrays["guard_tilt_deg"], expected_tilt, "guard tilt from quaternion", atol=1e-5)
    expected_shift = np.linalg.norm(arrays["object_com_xyz"][:, 1:, :2] - initial_guard_xy, axis=-1)
    _close(arrays["guard_displacement_xy_m"], expected_shift, "guard COM displacement")
    if np.any(arrays["guard_displacement_xy_m"] < 0) or np.any(arrays["guard_tilt_deg"] < 0):
        raise ValueError("guard displacement/tilt measurements cannot be negative")
    unsafe = np.any((arrays["guard_tilt_deg"] > tilt_limit + 1e-7) |
                    (shift_enabled & (arrays["guard_displacement_xy_m"] > shift_limit + 1e-9)), axis=1)
    _same(arrays["unsafe"], unsafe, "unsafe")
    failed = arrays["episode_failed"]
    if np.any(unsafe & ~failed) or np.any(failed[:-1] & ~failed[1:]):
        raise ValueError("episode_failed must latch any hazard and cannot clear inside an episode")
    forces = arrays["finger_normal_force_n"]
    if np.any(forces < 0):
        raise ValueError("finger normal forces cannot be negative")
    _same(arrays["left_contact"], forces[:, 0] > force_limit, "left_contact")
    _same(arrays["right_contact"], forces[:, 1] > force_limit, "right_contact")
    grasped = arrays["left_contact"] & arrays["right_contact"]
    _same(arrays["grasped"], grasped, "grasped")
    _close(arrays["target_height_m"], arrays["object_com_xyz"][:, 0, 2], "target COM height")
    height_ok = arrays["target_height_m"] >= height_limit - 1e-9
    _same(arrays["height_ok"], height_ok, "height_ok")
    goal = grasped & height_ok
    _same(arrays["goal_conditions_met"], goal, "goal_conditions_met")
    held = arrays["hold_seconds"]
    if np.any(held < -1e-9) or abs(held[0]) > 1e-9:
        raise ValueError("hold_seconds must be nonnegative and start at zero")
    if np.any(np.abs(held[~goal | failed]) > 1e-9):
        raise ValueError("hold timer must reset while goal conditions are absent or episode has failed")
    if np.any(held[1:] - held[:-1] > dt + 1e-8):
        raise ValueError("hold timer cannot increase faster than simulation time")
    if np.any(held > times + 1e-8):
        raise ValueError("hold timer cannot exceed elapsed episode time")
    # A timer drop while both saved frames satisfy goal may reflect an unseen
    # 2-ms contact break. Permit a reset but not an arbitrary partially lost time.
    if action_count:
        increased = np.isclose(held[1:], held[:-1] + dt, atol=1e-8, rtol=1e-7)
        reset_within_interval = held[1:] <= dt + 1e-8
        if np.any(~increased & ~reset_within_interval):
            raise ValueError("hold timer must advance continuously or restart within the last control interval")
    expected_success = goal & ~failed & (held >= hold_required - 1e-9)
    _same(arrays["task_success"], expected_success, "task_success")
    if "task_status" in arrays:
        status = arrays["task_status"]
        if status.shape != (state_count,) or status.dtype.kind != "U":
            raise ValueError("task_status must be a Unicode array of shape (num_states,)")
        expected_status = np.where(failed, "FAILED", np.where(expected_success, "SUCCESS", np.where(goal, "HOLDING", "IN_PROGRESS")))
        _same(status, expected_status, "task_status")
    if outcome["phase"] == "SUCCESS" and not bool(arrays["task_success"][-1]):
        raise ValueError("SUCCESS outcome requires a successful final task state")

    extra_report = {}
    if exploration:
        from exploration_validation import validate_exploration
        extra_report = validate_exploration(metadata, arrays, dt)

    image_dir = path / "images"
    expected_names = {f"{index:06d}.png" for index in range(state_count)}
    actual_names = {item.name for item in image_dir.iterdir() if item.is_file()}
    if actual_names != expected_names:
        missing_frames = sorted(expected_names - actual_names)
        extra_frames = sorted(actual_names - expected_names)
        raise ValueError(f"image frame set mismatch: missing={missing_frames[:5]}, extra={extra_frames[:5]}")
    for index in range(state_count):
        frame_path = image_dir / f"{index:06d}.png"
        with Image.open(frame_path) as frame:
            if frame.format != "PNG" or frame.size != (width, height) or frame.mode != "RGB":
                raise ValueError(f"{frame_path.name} must be a {width}x{height} RGB PNG")
            frame.load()  # Decode the entire PNG, not just its header.
    warnings = ["PNG/state temporal alignment and absence of overlays are recorded metadata, not visually proven by this check."]
    if np.any(failed & ~np.maximum.accumulate(unsafe)):
        warnings.append("A latched failure without a saved unsafe frame may originate in an intervening 2-ms physics substep.")
    return {
        "schema_version": schema_version,
        "num_actions": action_count, "num_states": state_count,
        "duration_s": float(times[-1] - times[0]), "outcome": outcome["phase"],
        "unsafe_states": int(np.count_nonzero(unsafe)),
        "final_episode_failed": bool(failed[-1]),
        "final_task_success": bool(arrays["task_success"][-1]),
        "image_size": [width, height], "warnings": warnings, **extra_report,
    }


def validate_episode(path) -> dict:
    """Inspect one episode without simulation, writes, or unpickling objects."""
    path = Path(path).resolve()
    result = {"path": str(path), "valid": False, "errors": [], "warnings": []}
    try:
        result.update(_check_episode(path))
        result["valid"] = True
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
        result["errors"].append(str(error))
    return result


def _manifest_episode_path(root, entry):
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError("each manifest episode must have a relative path")
    relative = Path(entry["path"])
    if relative.is_absolute() or len(relative.parts) != 1 or relative.parts[0] in (".", ".."):
        raise ValueError("manifest episode path must be one child directory, not an external path")
    resolved = (root / relative).resolve()
    if resolved.parent != root:
        raise ValueError("manifest episode path escapes the dataset directory")
    return resolved


def validate_dataset(root) -> dict:
    """Validate episodes separately; never concatenate arrays across resets."""
    root = Path(root).resolve()
    result = {"path": str(root), "valid": False, "errors": [], "warnings": [], "episodes": []}
    try:
        manifest = _json(root / "manifest.json")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] not in (1, 2, 3):
            raise ValueError("unsupported or missing dataset schema_version (expected 1, 2 or 3)")
        requested = _count(manifest.get("requested_episodes"), "requested_episodes", minimum=1)
        result["status"] = manifest.get("status")
        if manifest.get("status") != "complete":
            result["errors"].append("dataset is not complete; partial/interrupted episodes must not be silently used")
        entries = manifest.get("episodes")
        if not isinstance(entries, list):
            raise ValueError("manifest episodes must be a list")
        if len(entries) != requested:
            result["errors"].append(f"expected {requested} episodes, manifest contains {len(entries)}")
        seen_paths, seen_ids = set(), set()
        for entry in entries:
            path = _manifest_episode_path(root, entry)
            if "id" not in entry or str(entry["id"]) in seen_ids or path.name in seen_paths:
                raise ValueError("manifest has missing/duplicate episode ids or duplicate episode paths")
            seen_paths.add(path.name)
            seen_ids.add(str(entry["id"]))
            report = validate_episode(path)
            result["episodes"].append(report)
            if entry.get("status") != "complete":
                result["errors"].append(f"{path.name}: manifest episode status is not complete")
            if not report["valid"]:
                result["errors"].append(f"{path.name}: {'; '.join(report['errors'])}")
            else:
                if report["schema_version"] != manifest["schema_version"]:
                    result["errors"].append(f"{path.name}: manifest and episode schema versions disagree")
                for field in ("num_actions", "num_states"):
                    if entry.get(field) != report[field]:
                        result["errors"].append(f"{path.name}: manifest {field} disagrees with episode")
                if entry.get("phase") != report["outcome"]:
                    result["errors"].append(f"{path.name}: manifest phase disagrees with episode outcome")
        extra = {path.name for path in root.glob("episode_*") if path.is_dir()} - seen_paths
        if extra:
            result["errors"].append(f"episode directories absent from manifest: {sorted(extra)}")
        result["valid"] = not result["errors"]
        valid_episodes = [episode for episode in result["episodes"] if episode["valid"]]
        result["num_episodes"] = len(entries)
        result["total_actions"] = sum(episode["num_actions"] for episode in valid_episodes)
        result["total_states"] = sum(episode["num_states"] for episode in valid_episodes)
        if manifest["schema_version"] == 3:
            if manifest.get("dataset_kind") != "guided_exploration_v1":
                result["errors"].append("schema 3 manifest needs guided_exploration_v1")
            result["class_counts"] = {str(k): sum(e.get("class_counts", {}).get(str(k), 0)
                                                 for e in valid_episodes) for k in range(3)}
            result["hover_success_states"] = sum(e.get("hover_success_states", 0) for e in valid_episodes)
            result["hover_success_episode_count"] = sum(e.get("hover_success_states", 0) > 0
                                                       for e in valid_episodes)
            result["warnings"].append("Schema 3 DONE means recording ended, not task success; final_task_success is the legacy lift-only rule.")
            result["valid"] = not result["errors"]
        else:
            result["success_count"] = sum(episode["outcome"] == "SUCCESS" for episode in valid_episodes)
        result["warnings"].append("This is a file/label consistency check, not evidence of sufficient world-model training coverage.")
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
        result["errors"].append(str(error))
    return result


def export_snapshots(episode_path, output_dir):
    """Copy untouched first/middle/last PNGs into a NEW directory for visual QA."""
    episode_path, output_dir = Path(episode_path), Path(output_dir)
    metadata = _json(episode_path / "metadata.json")
    output_dir.mkdir(parents=True, exist_ok=False)
    indices = {"first": 0, "middle": metadata["num_states"] // 2, "last": metadata["num_states"] - 1}
    outputs = {}
    for name, index in indices.items():
        destination = output_dir / f"{name}.png"
        shutil.copyfile(episode_path / "images" / f"{index:06d}.png", destination)
        outputs[name] = str(destination.resolve())
    return outputs


def replay_episode(episode_path):
    """Display recorded frames with controls; no controller or physics is run."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    episode_path = Path(episode_path)
    metadata = _json(episode_path / "metadata.json")
    with np.load(episode_path / "trajectory.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    width, height = metadata["image"]["width"], metadata["image"]["height"]
    count, dt = metadata["num_states"], metadata["control_dt"]
    pygame.display.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode((width, height + 110))
        pygame.display.set_caption(f"Recorded grasp replay - {episode_path.name}")
        font, clock = pygame.font.Font(None, 22), pygame.time.Clock()
        index, loaded_index, surface = 0, -1, None
        paused, running, accumulated = False, True, 0.0
        while running:
            elapsed = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWMINIMIZED):
                    paused = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        accumulated = 0.0
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        paused, accumulated = True, 0.0
                        index = max(0, min(count - 1, index + (-1 if event.key == pygame.K_LEFT else 1)))
                    elif event.key == pygame.K_r:
                        index, accumulated, paused = 0, 0.0, False
            if not paused:
                accumulated += elapsed
                while accumulated >= dt and index < count - 1:
                    index += 1
                    accumulated -= dt
                if index == count - 1:
                    paused = True
            if index != loaded_index:
                surface = pygame.image.load(str(episode_path / "images" / f"{index:06d}.png")).convert()
                loaded_index = index
            screen.fill((25, 28, 35))
            screen.blit(surface, (0, 0))
            action_text = "Initial state (no preceding action)" if index == 0 else (
                f"Previous action [{index-1}] {arrays['action_phase'][index-1]}: "
                f"{np.array2string(arrays['actions'][index-1], precision=2, suppress_small=True)}"
            )
            lines = [
                f"RECORDED ONLY | Frame {index}/{count-1} | t={arrays['time'][index]:.2f} s | {'PAUSED' if paused else 'PLAYING'}",
                action_text,
                f"Unsafe={int(arrays['unsafe'][index])} Failed={int(arrays['episode_failed'][index])} "
                f"Success={int(arrays['task_success'][index])} Hold={arrays['hold_seconds'][index]:.2f} s",
                "Space: pause/play | Left/Right: one frame | R: restart | Esc: exit",
            ]
            if metadata.get("schema_version") == 3:
                lines[2] = (f"Class={arrays['class_label'][index]} Unsafe={int(arrays['unsafe'][index])} "
                            f"Hover={int(arrays['hover_success'][index])} "
                            f"Goal error={100 * arrays['hover_error_m'][index]:.1f}cm "
                            f"Hold={arrays['hover_hold_seconds'][index]:.2f}s")
            for row, line in enumerate(lines):
                screen.blit(font.render(line, True, (230, 234, 240)), (10, height + 10 + row * 24))
            pygame.display.flip()
    finally:
        pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, help="Dataset directory containing manifest.json")
    parser.add_argument("--episode", type=Path, help="Check a single episode directory instead")
    parser.add_argument("--replay", type=int, metavar="INDEX", help="After validation, replay an episode (zero-based manifest index)")
    parser.add_argument("--snapshot-dir", type=Path, help="NEW output directory for original first/middle/last PNGs")
    parser.add_argument("--report", type=Path, help="NEW JSON report file; also written for invalid/incomplete data, never overwrites")
    args = parser.parse_args(argv)
    if (args.dataset is None) == (args.episode is None):
        parser.error("provide either the dataset directory or --episode, not both")
    report = validate_episode(args.episode) if args.episode else validate_dataset(args.dataset)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report is not None:
        try:
            # Exclusive creation avoids races and preserves any existing report.
            # Do not create parents or alter source episode files implicitly.
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
        except OSError as error:
            parser.error(f"report export failed (existing files are never overwritten): {error}")
    if not report["valid"]:
        return 1
    if args.replay is not None or args.snapshot_dir is not None:
        selected = 0 if args.replay is None else args.replay
        if args.episode:
            if selected != 0:
                parser.error("--episode supports only --replay 0")
            episode_path = args.episode.resolve()
        else:
            manifest = _json(args.dataset / "manifest.json")
            if not 0 <= selected < len(manifest["episodes"]):
                parser.error("replay index is outside the manifest episode list")
            episode_path = _manifest_episode_path(args.dataset.resolve(), manifest["episodes"][selected])
        if args.snapshot_dir is not None:
            try:
                print(json.dumps({"snapshots": export_snapshots(episode_path, args.snapshot_dir)}, indent=2))
            except OSError as error:
                parser.error(f"snapshot export failed (existing output is never overwritten): {error}")
        if args.replay is not None:
            replay_episode(episode_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
