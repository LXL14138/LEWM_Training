"""Measured-geometry strata and fixed-threshold metrics for red-guard danger.

numpy only: no torch, h5py or simulator imports, so the same code runs under
the Windows numpy-only interpreter and under WSL. Every stratum comes from
RECORDED simulator measurements in trajectory.npz (guard quaternions, COMs,
pinch position, contact peaks, frustum flags), never from collection intent
such as the mode or controller phase. Scores may come from any source (real
image latents today, WM-predicted latents for WP4); the evaluation code is
identical. The alarm threshold is an input and is never fitted here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

STRATA_FORMAT = "lewm_danger_strata_v1"
UNSAFE_TILT_DEG = 30.0
TILT_TOLERANCE_DEG = 1e-7
CONTACT_FORCE_N = 1e-3
DIRECTIONS = ("toward_camera", "away_camera", "lateral_outward", "lateral_inward")
TILT_EDGES, TILT_NAMES = (30.0, 40.0, 60.0, 85.0), ("30-40", "40-60", "60-85", "85+")
GRIP_EDGES, GRIP_NAMES = (0.0, 0.05, 0.10), ("<5cm", "5-10cm", ">=10cm")
SINCE_EDGES, SINCE_NAMES = (0, 3, 6, 11), ("0-2", "3-5", "6-10", "11+")
SAFE_TILT_EDGES, SAFE_TILT_NAMES = (0.0, 5.0, 10.0, 20.0), ("<5", "5-10", "10-20", "20-30")
FRAME_NAMES = ("in_frame", "partly_out_of_frame")
LEAN_NAMES = ("leaning", "not_leaning")
EVENT_WINDOW = (10, 5)      # frames before/after the first onset in which an alarm "catches" the event
LEAN_MAX_TILT_DEG = 60.0    # leaning = whole-episode peak below this AND sustained over threshold
LEAN_MIN_FRAMES = 10        # sustained = at least this many consecutive unsafe frames (0.5 s at 20 Hz)
PRE_ONSET_EXCLUDE = 20      # safe frames this close before an onset are pre-fall, not hard negatives
RECOVER_TILT_DEG = 10.0     # "tilted 20-30 then recovered" needs a later tilt below this
# "Gripper near a guard": the pinch centre inside the guard footprint expanded by the closed gripper's half
# extents (3 cm along X where the fingers open, 1.5 cm along Y) plus a 3 cm clearance, below 13 cm height.
# (The first version used a 5 cm pinch-to-COM radius, which misses hovering beside the +-X faces.)
NEAR_BOX_XY_M, NEAR_GRIP_MAX_Z_M = (0.02 + 0.03 + 0.03, 0.02 + 0.015 + 0.03), 0.13
HELD_NEAR_M, HELD_MIN_Z_M = 0.06, 0.07
HARD_NEGATIVES = ("contact_no_fall", "tilt_20_30_recovered", "near_no_contact", "held_green_near_guard")
FIELDS = ("time", "object_quat_wxyz", "object_com_xyz", "guard_tilt_deg", "block_in_frame",
          "pinch_xyz", "guard_contact_peak_last_step_n", "unsafe", "grasped")
FRAME_COLUMNS = ("max_tilt_deg", "direction", "tilt_band", "grip_dist_m", "grip_band", "frame_band",
                 "since_onset", "since_band", "until_onset", "contact", "clean_safe", "safe_tilt_band",
                 *[f"hn_{g}" for g in HARD_NEGATIVES])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native(value):
    """JSON-safe copy: numpy scalars/arrays become Python values; NaN is never emitted."""
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [native(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(native(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def body_up(quat_wxyz):
    """World direction of a body's +Z axis from unit wxyz quaternions (any leading shape)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Expected finite wxyz quaternions")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.allclose(norm, 1, atol=1e-5, rtol=0):
        raise ValueError("Non-unit quaternion")
    q = q / norm
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


def tilt_from_up(up):
    """Angle between the body axis and world vertical, degrees in [0, 180]."""
    up = np.asarray(up, dtype=np.float64)
    if up.shape[-1] != 3:
        raise ValueError("Expected 3-vectors")
    return np.degrees(np.arccos(np.clip(up[..., 2], -1.0, 1.0)))


def fall_direction(up, guard_index):
    """Measured lean class. Camera looks along -X; guard 0 sits at y<0, guard 1 at y>0.

    |u_x| > |u_y| is a depth lean (toward camera if u_x>0, else away). Otherwise
    the lean is lateral: inward if the top points toward the green block
    (guard 0: +Y, guard 1: -Y), else outward. Exact ties count as lateral; an
    exactly vertical axis has no direction and is labelled "upright".
    """
    up, g = np.asarray(up, dtype=np.float64), np.asarray(guard_index)
    if up.shape[-1] != 3 or g.shape != up.shape[:-1] or not np.isin(g, [0, 1]).all():
        raise ValueError("Expected up-vectors with a matching 0/1 guard index")
    ux, uy = up[..., 0], up[..., 1]
    depth = np.abs(ux) > np.abs(uy)
    inward = np.where(g == 0, uy > 0, uy < 0)
    out = np.where(depth, np.where(ux > 0, DIRECTIONS[0], DIRECTIONS[1]),
                   np.where(inward, DIRECTIONS[3], DIRECTIONS[2]))
    return np.where((ux == 0) & (uy == 0), "upright", out).astype(str)


def band_labels(values, edges, names, below):
    """Half-open bands [edges[i], edges[i+1]); the last is unbounded; NaN -> 'n/a'."""
    if len(edges) != len(names) or list(edges) != sorted(edges):
        raise ValueError("Band edges must be sorted and match the names")
    v = np.asarray(values, dtype=np.float64)
    out = np.full(v.shape, "n/a", dtype=object)
    ok = np.isfinite(v)
    idx = np.searchsorted(np.asarray(edges, dtype=np.float64), v[ok], side="right")
    out[ok] = np.asarray((below, *names), dtype=object)[idx]
    return out.astype(str)


def _longest_run(mask):
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def episode_features(arrays, dt):
    """Per-frame measured strata for one episode, strict about recorded consistency."""
    a = {}
    for key in FIELDS:
        if key not in arrays:
            raise ValueError(f"Missing trajectory field: {key}")
        a[key] = np.asarray(arrays[key])
    if a["unsafe"].ndim != 1 or len(a["unsafe"]) < 2:
        raise ValueError("Episode needs a 1-D unsafe vector with at least two states")
    n = len(a["unsafe"])
    shapes = {"time": (n,), "object_quat_wxyz": (n, 3, 4), "object_com_xyz": (n, 3, 3),
              "guard_tilt_deg": (n, 2), "block_in_frame": (n, 3), "pinch_xyz": (n, 3),
              "guard_contact_peak_last_step_n": (n, 2), "unsafe": (n,), "grasped": (n,)}
    for key, shape in shapes.items():
        if a[key].shape != shape:
            raise ValueError(f"{key} has shape {a[key].shape}, expected {shape}")
    for key in ("time", "object_quat_wxyz", "object_com_xyz", "guard_tilt_deg", "pinch_xyz",
                "guard_contact_peak_last_step_n"):
        if a[key].dtype.kind not in "fiu" or not np.isfinite(a[key]).all():
            raise ValueError(f"{key} must be finite numeric")
    for key in ("block_in_frame", "unsafe", "grasped"):
        if a[key].dtype != np.dtype(bool):
            raise ValueError(f"{key} must be boolean")
    if not (np.isfinite(dt) and dt > 0) or not np.allclose(np.diff(a["time"]), dt, atol=1e-7, rtol=0):
        raise ValueError("Timestamps disagree with control_dt")
    if np.any(a["guard_contact_peak_last_step_n"] < 0):
        raise ValueError("Negative contact force")
    up = body_up(a["object_quat_wxyz"][:, 1:])                      # guards only: (n, 2, 3)
    tilt = tilt_from_up(up)
    if not np.allclose(tilt, a["guard_tilt_deg"], atol=1e-4, rtol=0):
        raise ValueError("Recorded guard tilt disagrees with the recorded quaternion")
    unsafe = (tilt > UNSAFE_TILT_DEG + TILT_TOLERANCE_DEG).any(1)
    if not np.array_equal(unsafe, a["unsafe"]):
        raise ValueError("Recorded unsafe label disagrees with the current-state tilt rule")
    idx = np.arange(n)
    g = tilt.argmax(1)
    max_tilt = tilt[idx, g]
    com = a["object_com_xyz"]
    worst_com = com[idx, 1 + g]
    pinch = a["pinch_xyz"]
    grip_dist = np.linalg.norm(pinch[:, :2] - worst_com[:, :2], axis=1)
    contact = (a["guard_contact_peak_last_step_n"] > CONTACT_FORCE_N).any(1)
    offsets = np.abs(pinch[:, None, :2] - com[:, 1:, :2])                       # (n, 2 guards, xy)
    near_box = ((offsets[..., 0] <= NEAR_BOX_XY_M[0]) & (offsets[..., 1] <= NEAR_BOX_XY_M[1])).any(1)
    green_to_guard = np.linalg.norm(com[:, None, 0, :2] - com[:, 1:, :2], axis=2).min(1)
    onsets = np.flatnonzero(unsafe)
    onset = int(onsets[0]) if len(onsets) else -1
    since = (idx - onset).astype(np.float64) if onset >= 0 else np.full(n, np.nan)
    clean_safe = ~unsafe & ((onset < 0) | (idx < onset - PRE_ONSET_EXCLUDE))
    # Recovery check for the worst guard at t: its tilt drops below RECOVER_TILT_DEG
    # later, but before any onset (looking ahead in the record is fine for a diagnostic).
    end = onset if onset >= 0 else n
    future_min = np.full(n, np.inf)
    if end > 1:
        suffix_min = np.minimum.accumulate(tilt[:end][::-1], axis=0)[::-1]   # min over [t, end)
        future_min[:end - 1] = suffix_min[1:end][idx[:end - 1], g[:end - 1]]
    hard = {
        "contact_no_fall": clean_safe & contact,
        "tilt_20_30_recovered": clean_safe & (max_tilt >= 20) & (future_min < RECOVER_TILT_DEG),
        "near_no_contact": clean_safe & ~contact & near_box & (pinch[:, 2] < NEAR_GRIP_MAX_Z_M),
        "held_green_near_guard": clean_safe & ~contact & a["grasped"] & (green_to_guard < HELD_NEAR_M)
                                 & (com[:, 0, 2] >= HELD_MIN_Z_M),
    }
    direction = fall_direction(up[idx, g], g)
    features = {
        "n": n, "unsafe": unsafe, "worst_guard": g, "max_tilt_deg": max_tilt, "direction": direction,
        "tilt_band": band_labels(max_tilt, TILT_EDGES, TILT_NAMES, "<30"),
        "grip_dist_m": grip_dist, "grip_band": band_labels(grip_dist, GRIP_EDGES, GRIP_NAMES, "n/a"),
        "frame_band": np.where(a["block_in_frame"][idx, 1 + g], FRAME_NAMES[0], FRAME_NAMES[1]).astype(str),
        "since_onset": since, "since_band": band_labels(since, SINCE_EDGES, SINCE_NAMES, "before_onset"),
        "until_onset": -since, "contact": contact, "clean_safe": clean_safe,
        "safe_tilt_band": np.where(unsafe, "unsafe", band_labels(max_tilt, SAFE_TILT_EDGES, SAFE_TILT_NAMES, "n/a")).astype(str),
        **{f"hn_{k}": v for k, v in hard.items()},
        "event": None, "leaning": False,
    }
    if onset >= 0:
        run = _longest_run(unsafe[onset:])
        peak = float(max_tilt[onset:].max())
        leaning = bool(peak < LEAN_MAX_TILT_DEG and run >= LEAN_MIN_FRAMES)
        dominant = Counter(direction[unsafe].tolist())
        features["leaning"] = leaning
        features["event"] = {
            "onset_frame": onset, "frames": n, "unsafe_frames": int(unsafe.sum()), "longest_unsafe_run": run,
            "peak_tilt_deg": peak, "peak_band": band_labels([peak], TILT_EDGES, TILT_NAMES, "<30")[0],
            "onset_tilt_deg": float(max_tilt[onset]), "onset_direction": str(direction[onset]),
            "dominant_direction": max(DIRECTIONS, key=lambda d: (dominant.get(d, 0), -DIRECTIONS.index(d))),
            "grip_band_at_onset": str(features["grip_band"][onset]),
            "frame_band_at_onset": str(features["frame_band"][onset]),
            "leaning": leaning, "worst_guard_at_onset": int(g[onset]),
        }
    return features


def load_episode_arrays(episode_dir):
    """Read only the strata fields from a raw trajectory.npz (no images, no pickle)."""
    with np.load(Path(episode_dir) / "trajectory.npz", allow_pickle=False) as npz:
        missing = [k for k in FIELDS if k not in npz.files]
        if missing:
            raise ValueError(f"trajectory.npz lacks {missing}")
        return {k: np.asarray(npz[k]) for k in FIELDS}


def check_predictions(frame, score, n):
    frame, score = np.asarray(frame), np.asarray(score, dtype=np.float64)
    if (frame.ndim != 1 or frame.dtype.kind not in "iu" or score.shape != frame.shape or not len(frame)
            or frame.min() < 0 or frame.max() >= n or len(np.unique(frame)) != len(frame)):
        raise ValueError("Scored frames must be unique integers inside the episode")
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("Scores must be finite and inside [0, 1]")
    order = np.argsort(frame, kind="stable")
    return frame[order].astype(np.int64), score[order]


def recall_rows(positive, alarm, labels, order, count_key="unsafe_frames", hit_key="alarms"):
    labels = np.asarray(labels).astype(str)
    rows = []
    for name in order:
        sel = positive & (labels == name)
        n, k = int(sel.sum()), int((alarm & sel).sum())
        rows.append({"stratum": name, count_key: n, hit_key: k, "recall": k / n if n else None})
    return rows


def macro_average(rows, count_key="unsafe_frames", thin=5):
    """Unweighted mean recall over strata that have samples; empty strata are listed, not zero."""
    used = [r for r in rows if r[count_key] > 0]
    return {"macro_recall": float(np.mean([r["recall"] for r in used])) if used else None,
            "strata_used": [r["stratum"] for r in used],
            "strata_empty": [r["stratum"] for r in rows if r[count_key] == 0],
            "strata_thin": [r["stratum"] for r in used if r[count_key] < thin]}


def _cells(direction, band):
    return np.char.add(np.char.add(np.asarray(direction).astype(str), "|"), np.asarray(band).astype(str))


CELL_ORDER = tuple(f"{d}|{b}" for d in DIRECTIONS for b in TILT_NAMES)


def lead_quantiles(values):
    """median / quartiles / count of a list of frame offsets, or None when empty."""
    if not len(values):
        return None
    v = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "count": int(len(v))}


def connected_alarm_start(frame, alarm, in_window, onset):
    """Start of the alarm segment connected to the event window (frames relative to onset).

    Take the first alarmed frame inside the window, walk backwards while the
    scored frames stay alarmed AND consecutive (frame difference exactly 1; a
    gap in a sparse table ends the segment), and return (segment_start - onset,
    first_alarm_in_earlier_segment). The second value says whether the episode's
    first alarm (the frame `first_alarm_minus_onset` reports) lies in an earlier
    segment that is NOT connected to this one. Both are None when no scored
    frame inside the window is alarmed (event not caught).
    """
    frame, alarm, in_window = np.asarray(frame), np.asarray(alarm, dtype=bool), np.asarray(in_window, dtype=bool)
    hits = np.flatnonzero(alarm & in_window)
    if not len(hits):
        return None, None
    i = int(hits[0])
    while i > 0 and alarm[i - 1] and frame[i] - frame[i - 1] == 1:
        i -= 1
    return int(frame[i] - onset), bool(frame[alarm].min() < frame[i])


def evaluate_set(features, predictions, threshold, dt, window=EVENT_WINDOW):
    """Fixed-threshold, measured-strata metrics for one table of (episode, frame, score).

    `features[eid]` comes from episode_features; `predictions[eid]` holds
    `frame` and `score` arrays. Works for dense per-frame scores and for sparse
    tables (e.g. WM forecasts scored at target frames only); coverage of the
    event window is reported so sparse inputs cannot masquerade as dense ones.
    """
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite number in [0, 1]; it must be fixed beforehand")
    before, after = (int(v) for v in window)
    if before < 0 or after < 0:
        raise ValueError("Event window must be non-negative")
    if not predictions:
        raise ValueError("No scored episodes")
    table = {k: [] for k in ("episode_id", "frame", "score", "alarm", "unsafe", "leaning", "has_onset", *FRAME_COLUMNS)}
    events, quiet = [], []
    for eid in sorted(predictions):
        f = features[eid]
        frame, score = check_predictions(predictions[eid]["frame"], predictions[eid]["score"], f["n"])
        alarm = score >= threshold
        m = len(frame)
        table["episode_id"].append(np.full(m, eid, dtype=np.int64))
        table["frame"].append(frame)
        table["score"].append(score)
        table["alarm"].append(alarm)
        table["unsafe"].append(f["unsafe"][frame])
        table["leaning"].append(np.full(m, f["leaning"], dtype=bool))
        table["has_onset"].append(np.full(m, f["event"] is not None, dtype=bool))
        for key in FRAME_COLUMNS:
            table[key].append(np.asarray(f[key])[frame])
        starts = int((alarm & ~np.r_[False, alarm[:-1] & (np.diff(frame) == 1)]).sum())
        if f["event"] is None:
            quiet.append({"episode_id": eid, "scored_frames": m, "alarm_frames": int(alarm.sum()), "alarm_runs": starts})
            continue
        onset = f["event"]["onset_frame"]
        lo, hi = max(0, onset - before), min(f["n"] - 1, onset + after)
        in_window = (frame >= lo) & (frame <= hi)
        scored = int(in_window.sum())
        connected_start, first_in_earlier = connected_alarm_start(frame, alarm, in_window, onset)
        events.append({**f["event"], "episode_id": eid, "window_frames": hi - lo + 1, "window_scored": scored,
                       "caught": bool(alarm[in_window].any()) if scored else None,
                       "first_alarm_minus_onset": int(frame[alarm].min() - onset) if alarm.any() else None,
                       "connected_alarm_start_minus_onset": connected_start,
                       "first_alarm_in_earlier_segment": first_in_earlier,
                       "unsafe_frames_scored": int(f["unsafe"][frame].sum()),
                       "unsafe_frame_recall": float(alarm[f["unsafe"][frame]].mean()) if f["unsafe"][frame].any() else None})
    t = {k: np.concatenate(v) for k, v in table.items()}
    y, alarm = t["unsafe"].astype(bool), t["alarm"].astype(bool)
    tp, fn, fp, tn = (int(np.sum(m)) for m in (y & alarm, y & ~alarm, ~y & alarm, ~y & ~alarm))
    ratio = lambda a, b: a / b if b else None
    frame_by = {
        "direction": recall_rows(y, alarm, t["direction"], DIRECTIONS),
        "tilt_band": recall_rows(y, alarm, t["tilt_band"], TILT_NAMES),
        "grip_band": recall_rows(y, alarm, t["grip_band"], GRIP_NAMES),
        "frame_band": recall_rows(y, alarm, t["frame_band"], FRAME_NAMES),
        "since_onset": recall_rows(y, alarm, t["since_band"], SINCE_NAMES),
        "leaning": recall_rows(y, alarm, np.where(t["leaning"], *LEAN_NAMES), LEAN_NAMES),
        "direction_x_tilt_band": recall_rows(y, alarm, _cells(t["direction"], t["tilt_band"]), CELL_ORDER),
    }
    scored_events = [e for e in events if e["window_scored"]]
    ev_hit = np.asarray([e["caught"] for e in scored_events], dtype=bool)
    ev_pos = np.ones(len(scored_events), dtype=bool)
    col = lambda key: np.asarray([e[key] for e in scored_events]).astype(str)
    event_by = {
        "direction": recall_rows(ev_pos, ev_hit, col("dominant_direction"), DIRECTIONS, "events", "caught"),
        "onset_direction": recall_rows(ev_pos, ev_hit, col("onset_direction"), DIRECTIONS, "events", "caught"),
        "peak_band": recall_rows(ev_pos, ev_hit, col("peak_band"), TILT_NAMES, "events", "caught"),
        "grip_band_at_onset": recall_rows(ev_pos, ev_hit, col("grip_band_at_onset"), GRIP_NAMES, "events", "caught"),
        "frame_band_at_onset": recall_rows(ev_pos, ev_hit, col("frame_band_at_onset"), FRAME_NAMES, "events", "caught"),
        "leaning": recall_rows(ev_pos, ev_hit, np.where([e["leaning"] for e in scored_events], *LEAN_NAMES) if scored_events else [], LEAN_NAMES, "events", "caught"),
        "direction_x_peak_band": recall_rows(ev_pos, ev_hit, _cells(col("dominant_direction"), col("peak_band")) if scored_events else [], CELL_ORDER, "events", "caught"),
    }
    leads = [e["first_alarm_minus_onset"] for e in events if e["first_alarm_minus_onset"] is not None]
    connected = [e["connected_alarm_start_minus_onset"] for e in events if e["connected_alarm_start_minus_onset"] is not None]
    earlier_segment = sum(bool(e["first_alarm_in_earlier_segment"]) for e in events)
    safe = ~y
    until = t["until_onset"]
    context = [("1-5_frames_before_onset", (until >= 1) & (until <= 5)),
               ("6-20_frames_before_onset", (until >= 6) & (until <= 20)),
               ("safe_after_onset", t["has_onset"] & (until <= 0)),
               ("other_frames_unsafe_episode", t["has_onset"] & (until > 20)),
               ("never_unsafe_episode", ~t["has_onset"])]
    safe_rows = lambda groups: [{"group": name, "safe_frames": int((safe & m).sum()), "alarms": int((safe & m & alarm).sum()),
                                 "false_alarm_rate": ratio(int((safe & m & alarm).sum()), int((safe & m).sum()))} for name, m in groups]
    hard = {name: {"frames": int(t[f"hn_{name}"].sum()), "alarms": int((t[f"hn_{name}"] & alarm).sum()),
                   "false_alarm_rate": ratio(int((t[f"hn_{name}"] & alarm).sum()), int(t[f"hn_{name}"].sum()))}
            for name in HARD_NEGATIVES}
    hard_rates = [v["false_alarm_rate"] for v in hard.values() if v["false_alarm_rate"] is not None]
    quiet_frames = sum(q["scored_frames"] for q in quiet)
    minutes = quiet_frames * dt / 60.0
    summary = {
        "format": STRATA_FORMAT, "threshold": threshold, "control_dt": dt, "event_window": [before, after],
        "frames": {"scored": int(len(y)), "unsafe": int(y.sum()), "episodes": int(len(predictions)),
                   "confusion_rows_true_cols_pred_safe_unsafe": [[tn, fp], [fn, tp]],
                   "unsafe_recall": ratio(tp, tp + fn), "unsafe_precision": ratio(tp, tp + fp),
                   "false_alarm_rate": ratio(fp, fp + tn)},
        "frame_recall_by": frame_by,
        "frame_macro_recall": {k: macro_average(v) for k, v in frame_by.items()},
        "events": {"total": len(events), "scored": len(scored_events), "caught": int(ev_hit.sum()),
                   "recall": ratio(int(ev_hit.sum()), len(scored_events)),
                   "unscored_episode_ids": [e["episode_id"] for e in events if not e["window_scored"]],
                   "window_coverage_mean": float(np.mean([e["window_scored"] / e["window_frames"] for e in events])) if events else None,
                   "first_alarm_minus_onset_frames": lead_quantiles(leads),
                   # A3 (audit 2026-09-25): start of the alarm segment CONNECTED to the event window, and how many
                   # caught events owe their first alarm to an earlier, disconnected segment.
                   "connected_alarm_start_minus_onset_frames": lead_quantiles(connected),
                   "first_alarm_in_earlier_disconnected_segment": int(earlier_segment),
                   "recall_by": event_by,
                   "macro_recall": {k: macro_average(v, "events") for k, v in event_by.items()},
                   "per_event": events},
        "hard_negatives": {**hard, "macro_false_alarm_rate": float(np.mean(hard_rates)) if hard_rates else None},
        "safe_frames": {"by_context": safe_rows(context),
                        "by_tilt_band": safe_rows([(b, t["safe_tilt_band"] == b) for b in SAFE_TILT_NAMES]),
                        "by_contact": safe_rows([("guard_contact_this_step", t["contact"]), ("no_guard_contact", ~t["contact"])])},
        "never_unsafe_episodes": {"episodes": len(quiet), "episodes_with_alarm": sum(q["alarm_frames"] > 0 for q in quiet),
                                  "scored_frames": quiet_frames, "alarm_frames": sum(q["alarm_frames"] for q in quiet),
                                  "alarm_runs": sum(q["alarm_runs"] for q in quiet), "scored_minutes": minutes,
                                  "alarm_frames_per_minute": ratio(sum(q["alarm_frames"] for q in quiet), minutes) if minutes else None,
                                  "alarm_runs_per_minute": ratio(sum(q["alarm_runs"] for q in quiet), minutes) if minutes else None},
    }
    summary["primary_metrics"] = {
        "macro_event_recall_direction_x_severity": summary["events"]["macro_recall"]["direction_x_peak_band"]["macro_recall"],
        "macro_event_recall_direction": summary["events"]["macro_recall"]["direction"]["macro_recall"],
        "macro_frame_recall_direction_x_severity": summary["frame_macro_recall"]["direction_x_tilt_band"]["macro_recall"],
        "macro_frame_recall_direction": summary["frame_macro_recall"]["direction"]["macro_recall"],
        "hard_negative_macro_false_alarm_rate": summary["hard_negatives"]["macro_false_alarm_rate"],
        "note": "unweighted means over strata with at least one sample; see strata_empty/strata_thin",
    }
    return summary, t


def onset_records_to_predictions(path):
    """Turn evaluate_guard_onsets records.jsonl into score tables keyed by target frame.

    One table per (method, lead): the score refers to the label of `anchor_frame`.
    These are sparse tables (one frame per episode), so event windows are covered
    by a single frame and frame-level strata cover only anchor frames.
    """
    sets = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            for method, score in r["scores"].items():
                name = f"onsets_{method}_lead{int(r['lead'])}"
                episode = sets.setdefault(name, {}).setdefault(int(r["episode_id"]), {"frame": [], "score": []})
                episode["frame"].append(int(r["anchor_frame"]))
                episode["score"].append(float(score))
    if not sets:
        raise ValueError("No onset records")
    return {name: {eid: {k: np.asarray(v) for k, v in ep.items()} for eid, ep in table.items()} for name, table in sets.items()}
