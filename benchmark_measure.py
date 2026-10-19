"""Measured stratum of a danger-benchmark episode. numpy only; shared by the cell report and the split.

An attempt's intent is only a plan. The stratum an episode actually belongs to
comes from the recorded geometry through wm_training.danger_strata (the WP0
ruler): fall direction = dominant direction of the worst guard, severity from
the whole-episode peak tilt and whether the lean was sustained.
"""
from __future__ import annotations

from pathlib import Path

from benchmark_plan import BENCH_INTERACTION_REVISION, validate_intent
from wm_training.danger_strata import HARD_NEGATIVES, episode_features, load_episode_arrays, read_json

MEASURE_FORMAT = "danger_bench_measure_v1"
SEVERITY_BY_PEAK = (("S1_flat", 85.0), ("S2_lean", 40.0))   # >= 85 flat; sustained lean >= 40 is S2, else S3


def measured_cell(features):
    """direction|severity for events, or no_event with the hard-negative flags that occurred."""
    event = features["event"]
    if event is None:
        flags = [name for name in HARD_NEGATIVES if features[f"hn_{name}"].any()]
        return "no_event", {"max_tilt_deg": float(features["max_tilt_deg"].max()), "negative_flags": flags}
    peak = event["peak_tilt_deg"]
    if peak >= SEVERITY_BY_PEAK[0][1]:
        severity = "S1_flat"
    elif event["leaning"]:
        severity = "S2_lean" if peak >= SEVERITY_BY_PEAK[1][1] else "S3_just_over"
    else:
        severity = "transient"   # crossed 30 deg but neither fell flat nor stayed over for 10 frames
    return f"{event['dominant_direction']}|{severity}", {
        "peak_tilt_deg": peak, "onset_frame": event["onset_frame"], "longest_unsafe_run": event["longest_unsafe_run"],
        "frames_after_onset": event["frames"] - 1 - event["onset_frame"], "onset_direction": event["onset_direction"]}


def split_stratum(cell, intent):
    """Stratum used for the dev/test split: measured cell for events, intent class for negatives."""
    if cell != "no_event":
        return cell
    return intent["cell"] if intent["cell"].startswith("negative|") else "negative|unintended"


def load_benchmark_episode(folder):
    """Metadata + recorded geometry of one benchmark episode; audits the stored labels on the way."""
    folder = Path(folder)
    meta = read_json(folder / "metadata.json")
    controller = meta.get("controller", {})
    if meta.get("status") != "complete" or controller.get("interaction_revision", 1) < BENCH_INTERACTION_REVISION:
        raise ValueError(f"{folder}: not a complete benchmark episode")
    intent = validate_intent(controller["benchmark_intent"])
    record = meta["outcome"]["benchmark"]
    features = episode_features(load_episode_arrays(folder), float(meta["control_dt"]))
    if len(features["unsafe"]) != meta["num_states"]:
        raise ValueError(f"{folder}: state count disagrees with metadata")
    cell, detail = measured_cell(features)
    return {"intent": intent, "record": record, "features": features, "measured_cell": cell, "detail": detail,
            "stratum": split_stratum(cell, intent), "frames": int(features["n"]),
            "stop_reason": meta["outcome"]["stop_reason"], "ever_lifted": bool(meta["outcome"].get("ever_lifted", False))}
