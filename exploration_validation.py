"""Pure saved-array checks for guided exploration (no simulator imports)."""

import numpy as np

from benchmark_plan import BENCH_INTERACTION_REVISION, BENCH_MODES, BENCH_PHASES, validate_intent
from counterfactual_plan import CF_INTERACTION_REVISION, CF_PHASES, validate_counterfactual

EXPLORATION_PHASES = {"SETTLE", "FREE", "APPROACH", "DESCEND", "CLOSE", "LIFT", "RETRACT",
                      "CARRY", "HOVER", "GUARD_APPROACH", "GUARD_LOWER", "GUARD_PUSH",
                      "CARRY_GUARD", "RELEASE", "EVENT_HOLD"}
EXPLORATION_PHASES.update({"GUARD_SEEK", "BRUSH_LIFT", "BRUSH_APPROACH", "BRUSH_LOWER", "BRUSH_SWEEP"})
EXPLORATION_PHASES.update(BENCH_PHASES)   # interaction_revision 3: danger benchmark attempts
EXPLORATION_PHASES.update(CF_PHASES)      # interaction_revision 4: counterfactual branches from restored frames
BENCHMARK_RECORD_KEYS = {"recipe", "cell_intent", "guard_index", "lean", "after_event", "park_green", "stop", "params",
                         "prologue", "hazard_step", "hazard_phase", "hazard_time_s", "hazard_tilt_deg",
                         "stop_rule_reached", "stop_rule_value", "end_reason", "tip_angle_deg", "intent"}


def validate_exploration(metadata, arrays, dt):
    def same(a, b, name):
        if not np.array_equal(a, b):
            raise ValueError(f"{name} does not match recorded measurements")

    def close(a, b, name):
        if not np.allclose(a, b, atol=1e-8, rtol=1e-7):
            raise ValueError(f"{name} is inconsistent")

    def positive(key):
        v = metadata.get(key)
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or v <= 0:
            raise ValueError(f"{key} must be finite and positive")
        return v

    n = metadata["num_states"]
    if metadata.get("dataset_kind") != "guided_exploration_v1":
        raise ValueError("schema 3 requires guided_exploration_v1")
    goal = np.asarray(metadata.get("hover_goal_xyz"), dtype=float)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("hover_goal_xyz must contain three finite coordinates")
    tol, speed_limit, hold_required = [positive(k) for k in
        ("hover_tolerance_m", "hover_speed_limit_mps", "hover_required_seconds")]
    numeric = {"hover_goal_xyz": (3,), "target_velocity_xyz_mps": (3,), "target_speed_mps": (),
               "hover_error_m": (), "hover_hold_seconds": (), "guard_contact_force_n": (2,),
               "guard_contact_peak_last_step_n": (2,)}
    boolean = {"hover_in_goal": (), "hover_slow": (), "hover_conditions_met": (),
               "hover_success": (), "block_in_frame": (3,)}
    for name, tail in {**numeric, **boolean}.items():
        value = arrays.get(name)
        if value is None or value.shape != (n,) + tail:
            raise ValueError(f"{name} has an invalid shape")
        if name in boolean:
            if value.dtype != np.dtype(bool):
                raise ValueError(f"{name} must be boolean")
        elif value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite numeric")
    close(arrays["hover_goal_xyz"], np.tile(goal, (n, 1)), "fixed hover goal")
    error = np.linalg.norm(arrays["object_com_xyz"][:, 0] - goal, axis=1)
    speed = np.linalg.norm(arrays["target_velocity_xyz_mps"], axis=1)
    close(arrays["hover_error_m"], error, "hover_error_m")
    close(arrays["target_speed_mps"], speed, "target_speed_mps")
    inside, slow = error <= tol + 1e-9, speed <= speed_limit + 1e-9
    conditions = inside & slow & arrays["grasped"]
    same(arrays["hover_in_goal"], inside, "hover_in_goal")
    same(arrays["hover_slow"], slow, "hover_slow")
    same(arrays["hover_conditions_met"], conditions, "hover_conditions_met")
    expected_class = np.where(arrays["unsafe"], 1, np.where(conditions, 0, 2))
    labels = arrays.get("class_label")
    if labels is None or labels.shape != (n,) or labels.dtype.kind not in "iu":
        raise ValueError("class_label must be an integer state vector")
    same(labels, expected_class, "class_label")
    held, failed = arrays["hover_hold_seconds"], arrays["episode_failed"]
    if abs(held[0]) > 1e-9 or np.any(held < -1e-9) or np.any(held > arrays["time"] + 1e-8):
        raise ValueError("invalid hover hold timer")
    if np.any(np.abs(held[~conditions | failed]) > 1e-9) or np.any(np.diff(held) > dt + 1e-8):
        raise ValueError("hover hold timer must reset outside conditions or after hazard")
    if n > 1:
        increased = np.isclose(held[1:], held[:-1] + dt, atol=1e-8, rtol=1e-7)
        if np.any(~increased & (held[1:] > dt + 1e-8)):
            raise ValueError("hover hold timer must advance continuously or restart")
    same(arrays["hover_success"], conditions & ~failed & (held >= hold_required - 1e-9), "hover_success")
    forces, peaks = arrays["guard_contact_force_n"], arrays["guard_contact_peak_last_step_n"]
    if np.any(forces < 0) or np.any(peaks < forces - 1e-9):
        raise ValueError("invalid guard contact forces/peaks")
    # Old schema-3 episodes lack green-only measurements. Require them for the
    # new controller revision, while retaining the old dataset reader.
    if metadata.get("controller", {}).get("interaction_revision", 1) >= 2:
        green = arrays.get("green_guard_contact_force_n")
        green_peak = arrays.get("green_guard_contact_peak_last_step_n")
        for value in (green, green_peak):
            if value is None or value.shape != (n, 2) or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError("invalid green-only guard contact measurement")
        if np.any(green > forces + 1e-9) or np.any(green_peak > peaks + 1e-9) or np.any(green > green_peak + 1e-9):
            raise ValueError("green-only contact exceeds total contact or peak")
        expected_green_steps = int(np.any(green_peak[1:] > 1e-3, axis=1).sum())
        if metadata.get("exploration_summary", {}).get("green_guard_contact_steps") != expected_green_steps:
            raise ValueError("green_guard_contact_steps does not match measurements")
    controller = metadata.get("controller", {})
    if controller.get("interaction_revision", 1) >= CF_INTERACTION_REVISION:
        # Counterfactual branch episodes: recorded prefix + one action law; record vs arrays.
        validate_counterfactual(metadata, arrays)
    elif controller.get("interaction_revision", 1) >= BENCH_INTERACTION_REVISION:
        # Danger-benchmark episodes must carry the scheduled intent and the controller's record.
        intent, record = controller.get("benchmark_intent"), metadata["outcome"].get("benchmark")
        if not isinstance(intent, dict) or not isinstance(record, dict):
            raise ValueError("benchmark episodes need controller.benchmark_intent and outcome.benchmark")
        validate_intent(intent)
        if controller.get("mode") not in BENCH_MODES or not set(BENCHMARK_RECORD_KEYS) <= set(record):
            raise ValueError("benchmark mode or record fields are missing")
        if record["intent"] != intent or record["recipe"] != intent["recipe"] or record["cell_intent"] != intent["cell"]:
            raise ValueError("benchmark record disagrees with the scheduled intent")
        step = record["hazard_step"]
        if step is None:
            if np.any(failed[:-1]):
                raise ValueError("guard hazard occurred but the benchmark controller did not record it")
        elif (type(step) is not int or not 0 < step < n or not failed[step] or failed[step - 1]):
            raise ValueError("benchmark hazard_step disagrees with the recorded episode_failed latch")
    expected_counts = {str(k): int((labels == k).sum()) for k in range(3)}
    summary = metadata.get("exploration_summary", {})
    if summary.get("class_counts") != expected_counts:
        raise ValueError("exploration class_counts do not match arrays")
    if not metadata["outcome"].get("stop_reason"):
        raise ValueError("exploration outcome needs stop_reason")
    return {"class_counts": expected_counts,
            "hover_success_states": int(arrays["hover_success"].sum()),
            "guard_contact_steps": int(np.any(peaks[1:] > 1e-3, axis=1).sum()),
            "out_of_frame_states": int((~np.all(arrays["block_in_frame"], axis=1)).sum())}
