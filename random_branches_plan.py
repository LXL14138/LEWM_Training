"""Stage 1 of the random-action branch study: choose start states and pre-draw the action sequences.

numpy only at import time (h5py is imported lazily, only to read the anchor frames' cache pixels), so the
pure functions run in both WSL venvs and the unit tests; the CLI runs in the TRAINING venv
(~/lewm-train/.venv, the one with h5py). Stage 2 is collect_random_branches.py (simulation venv).

Start states ("anchors" a; a >= 12 and a + 24 <= last frame; worst guard tilt at a <= 25 deg):

* Set V: frames of the wm_full_001 VAL REPORT-half episodes (runs/safety_classifier_001/run.json
  report_episode_ids; raw data/exploration_5000_001), stratified by the recorded state at a
  (near_goal > near_guard > carrying > free, see classify_frames).
* Set B: bench_test episodes of data/danger_bench_split_001 (raw danger_bench_001/002): fall events at
  onset - k (k uniform in 5..30, redrawn inside the event when the anchor is not admissible) and
  never-unsafe hard negatives (N1_touch / N2_rock / N3_near / N4_held_near) at a frame with the gripper near a guard.

For every anchor the plan fixes 8 rollouts (recorded, hold, 4 x random, 2 x guard) and pre-draws the
random segment directions / speeds from a per-anchor seed. It also copies the cache pixels of frames
a-10..a (wm_full_001 or wm_bench_001) so stage 2 can check that its re-rendered, letterboxed images equal
the cache. Reserved TEST episodes of wm_full_001 are never opened; nothing is fitted on any episode.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from wm_training.danger_strata import native, read_json, sha256, write_new_json

PLAN_FORMAT = "random_branches_plan_v1"
PREFIX = 10                         # recorded frames replayed before the anchor
HORIZON = 24                        # control steps after the anchor
MIN_ANCHOR = 12
SEG_STEPS, SEGMENTS = 8, 3          # 3 segments x 8 steps = HORIZON
FAMILIES = ("recorded", "hold", "random", "random", "random", "random", "guard", "guard")
SPEEDS_MPS = (0.02, 0.035, 0.05)
DIRECTION_STD = (1.0, 1.0, 0.5)
GUARD_ANGLE_DEG = 20.0
CONTROL_DT = 0.05
# Reference box: x/y and the upper z of the free-motion box of exploration.py (_free: [0.32,-0.1,0.09]..[0.5,0.1,0.25]);
# the lower z is the lowest pinch height the recorded exploration controller commands (_grasp: max(0.025, ...);
# _guard pushes at 3.5-9.5 cm), not _free's 0.09: with 0.09 a rollout could never descend below min(9 cm, anchor
# height), so from anchors above 9 cm only the top centimetre of the 10 cm guards would be reachable (RANDOM_BRANCHES.md).
# Always intersected with the scene's workspace bounds (env._workspace_bounds, measured [0.25,-0.35,0.02]..[0.6,0.35,0.35])
# and widened to contain the anchor pinch so the reference never jumps at step 0 (whatever the floor).
REF_BOX_LOW = np.array([0.32, -0.10, 0.025])
REF_BOX_HIGH = np.array([0.50, 0.10, 0.25])
WORKSPACE_LOW = np.array([0.25, -0.35, 0.02])
WORKSPACE_HIGH = np.array([0.60, 0.35, 0.35])
# Strata (Set V). "near" = within 6 cm horizontally of a guard COM, gated by height so that a gripper high above the
# guards (approach at 19 cm) or the green block resting in its slot does not count (see RANDOM_BRANCHES.md).
NEAR_RADIUS_M = 0.06
NEAR_GRIP_MAX_Z_M = 0.13            # wm_training.danger_strata NEAR_GRIP_MAX_Z_M (guard top 10 cm + 3 cm)
GREEN_NEAR_MAX_Z_M = 0.15           # grasped green COM below 15 cm = its bottom below the 10 cm guard top
NEAR_GUARD_MAX_TILT_DEG = 20.0
GOAL_NEAR_M = 0.05
SKIP_TILT_DEG = 25.0
MIN_GAP_FRAMES = HORIZON + 1        # anchors of one episode are at least this far apart (rollouts never overlap)
V_STRATA = ("near_guard", "carrying", "near_goal", "free")
V_QUOTA = {"near_guard": 40, "carrying": 40, "near_goal": 40, "free": 30}
V_PRIORITY = ("near_goal", "near_guard", "carrying", "free")
FALL_K = (5, 30)
SEVERITIES = ("S1_flat", "S2_lean", "S3_just_over")
NEGATIVE_STRATA = ("negative|N1_touch", "negative|N2_rock", "negative|N3_near", "negative|N4_held_near")
GLOBAL_ID_STRIDE = 100000           # wm_training.prepare_bench.global_id
NPZ_FIELDS = ("guard_tilt_deg", "pinch_xyz", "object_com_xyz", "grasped", "hover_error_m", "unsafe", "class_label", "time")
IMPLEMENTATION_FILES = ("random_branches_plan.py",)
WM_ACTION_COLUMNS = (0, 1, 2, 4)    # evaluate_bench_counterfactual.ACTION_COLUMNS
# Reporting-only constants of stage 2 (they change no rollout; they describe the rollouts for the S1/S2/S3 evaluator).
FREE_FLOOR_Z_M = 0.09               # lower z of exploration.py _free (the reference box goes lower, see RANDOM_BRANCHES.md)
DISTURBED_DEG = 10.0                # S2 "guard disturbed"; anchors whose guard is already >= 10 deg are counted apart
TRACKING_LOST_M = 0.02              # final |pinch - reference| above this = the tracker did not follow its reference
SATURATION_TOL = 1e-12              # |normalized dx, dy or dz| >= 1 - tol = the action sits on the +-5 cm bound


# ----- pure geometry / strata ---------------------------------------------------------------------------
def horizontal_guard_distance(points_xyz, object_com_xyz):
    """(T,3) points, (T,3,3) COMs -> (T,2) horizontal distance of each point to guard 1 and guard 2."""
    points, com = np.asarray(points_xyz, float), np.asarray(object_com_xyz, float)
    return np.linalg.norm(points[:, None, :2] - com[:, 1:, :2], axis=2)


def classify_frames(pinch_xyz, object_com_xyz, grasped, hover_error_m, guard_tilt_deg):
    """Stratum of every frame: near_goal > near_guard > carrying > free (priority order).

    near_goal  : grasped and green COM within 5 cm of the episode's hover goal.
    near_guard : worst guard tilt <= 20 deg and (pinch within 6 cm horizontally of a guard COM and below 13 cm,
                 or the grasped green block within 6 cm horizontally of a guard COM with its COM below 15 cm).
    carrying   : grasped, neither of the above.   free: everything else.
    """
    pinch, com = np.asarray(pinch_xyz, float), np.asarray(object_com_xyz, float)
    grasped, error, tilt = np.asarray(grasped, bool), np.asarray(hover_error_m, float), np.asarray(guard_tilt_deg, float)
    n = len(pinch)
    if pinch.shape != (n, 3) or com.shape != (n, 3, 3) or grasped.shape != (n,) or error.shape != (n,) or tilt.shape != (n, 2):
        raise ValueError("classify_frames: inconsistent shapes")
    grip_near = (horizontal_guard_distance(pinch, com).min(1) <= NEAR_RADIUS_M) & (pinch[:, 2] < NEAR_GRIP_MAX_Z_M)
    green_near = grasped & (horizontal_guard_distance(com[:, 0], com).min(1) <= NEAR_RADIUS_M) & (com[:, 0, 2] < GREEN_NEAR_MAX_Z_M)
    near_guard = (grip_near | green_near) & (tilt.max(1) <= NEAR_GUARD_MAX_TILT_DEG)
    out = np.full(n, "free", dtype="U10")
    out[grasped] = "carrying"
    out[near_guard] = "near_guard"
    out[grasped & (error < GOAL_NEAR_M)] = "near_goal"
    return out


def admissible_frames(frames, guard_tilt_deg):
    """Boolean (T,): a >= MIN_ANCHOR, a + HORIZON <= T - 1 and worst guard tilt at a <= SKIP_TILT_DEG."""
    tilt = np.asarray(guard_tilt_deg, float)
    if tilt.shape != (frames, 2):
        raise ValueError("guard_tilt_deg must be (T, 2)")
    idx = np.arange(frames)
    return (idx >= MIN_ANCHOR) & (idx + HORIZON <= frames - 1) & (tilt.max(1) <= SKIP_TILT_DEG)


def gripper_near_guard(pinch_xyz, object_com_xyz):
    """Hard-negative anchor rule: pinch within 6 cm horizontally of a guard COM and below 13 cm."""
    pinch = np.asarray(pinch_xyz, float)
    return (horizontal_guard_distance(pinch, object_com_xyz).min(1) <= NEAR_RADIUS_M) & (pinch[:, 2] < NEAR_GRIP_MAX_Z_M)


def pick_spaced(rng, candidates, taken, gap=MIN_GAP_FRAMES):
    """Uniform choice among candidate frames at least ``gap`` away from every frame in ``taken``; None if none."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if taken:
        far = np.all(np.abs(candidates[:, None] - np.asarray(sorted(taken))[None, :]) >= gap, axis=1)
        candidates = candidates[far]
    if not len(candidates):
        return None
    return int(candidates[rng.integers(len(candidates))])


def select_set_v(episodes, quotas, seed):
    """episodes: {episode_id: arrays (NPZ_FIELDS)} -> anchors, availability.

    Strata are filled in V_STRATA order; inside a stratum, seeded round-robin over the episodes that still have
    an admissible frame of that stratum (one anchor per episode per pass), frame uniform among the admissible
    frames that keep MIN_GAP_FRAMES to the episode's earlier anchors."""
    rng = np.random.default_rng(seed)
    strata, taken, available = {}, {}, {}
    for eid in sorted(episodes):
        a = episodes[eid]
        labels = classify_frames(a["pinch_xyz"], a["object_com_xyz"], a["grasped"], a["hover_error_m"], a["guard_tilt_deg"])
        ok = admissible_frames(len(labels), a["guard_tilt_deg"])
        strata[eid] = {s: np.flatnonzero(ok & (labels == s)) for s in V_STRATA}
        taken[eid] = []
    anchors = []
    for stratum in V_STRATA:
        pool = [eid for eid in sorted(episodes) if len(strata[eid][stratum])]
        available[stratum] = {"episodes": len(pool), "frames": int(sum(len(strata[e][stratum]) for e in pool))}
        order = [pool[i] for i in rng.permutation(len(pool))]
        chosen = 0
        while chosen < quotas[stratum] and order:
            still = []
            for eid in order:
                if chosen >= quotas[stratum]:
                    break
                frame = pick_spaced(rng, strata[eid][stratum], taken[eid])
                if frame is None:
                    continue
                taken[eid].append(frame)
                anchors.append({"set": "V", "group": f"V/{stratum}", "stratum": stratum, "episode_id": int(eid), "frame": frame,
                                "lead_k": None, "k_draws": None})
                chosen += 1
                still.append(eid)
            order = still
    return anchors, available


def round_robin(items, key, seed):
    """Seeded round-robin over groups (shuffled inside each group and in group order), like bench_restore_check.select_events."""
    rng = np.random.default_rng(seed)
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    names = sorted(groups)
    queues = {n: [groups[n][i] for i in rng.permutation(len(groups[n]))] for n in names}
    cycle = [names[i] for i in rng.permutation(len(names))]
    out = []
    while any(queues.values()):
        for n in cycle:
            if queues[n]:
                out.append(queues[n].pop(0))
    return out


def fall_anchor(rng, onset, frames, guard_tilt_deg):
    """Draw k uniformly from FALL_K; redraw (without replacement) until onset - k is admissible. -> (k, frame, draws) or None."""
    ok = admissible_frames(frames, guard_tilt_deg)
    ks = np.arange(FALL_K[0], FALL_K[1] + 1)
    for draws, k in enumerate(ks[rng.permutation(len(ks))], start=1):
        frame = onset - int(k)
        if 0 <= frame < frames and ok[frame]:
            return int(k), int(frame), draws
    return None


def negative_anchor(rng, arrays):
    """Uniform admissible frame with the gripper near a guard of a never-unsafe episode. -> frame or None."""
    if np.asarray(arrays["unsafe"], bool).any():
        raise ValueError("hard negatives must be never-unsafe episodes")
    ok = admissible_frames(len(arrays["guard_tilt_deg"]), arrays["guard_tilt_deg"]) & gripper_near_guard(arrays["pinch_xyz"], arrays["object_com_xyz"])
    candidates = np.flatnonzero(ok)
    return int(candidates[rng.integers(len(candidates))]) if len(candidates) else None


# ----- action sequences ---------------------------------------------------------------------------------
def unit(vector):
    vector = np.asarray(vector, float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("zero direction")
    return vector / norm


def random_direction(rng):
    while True:
        d = rng.normal(0.0, DIRECTION_STD)
        if np.linalg.norm(d) > 1e-9:
            return unit(d)


def guard_direction(rng, pinch_xyz, object_com_xyz):
    """Horizontal unit vector from the pinch toward the (horizontally) nearer guard COM, rotated by U(-20, 20) deg."""
    pinch, com = np.asarray(pinch_xyz, float), np.asarray(object_com_xyz, float)
    dist = horizontal_guard_distance(pinch[None], com[None])[0]
    index = int(np.argmin(dist))                                  # 0 -> object_1, 1 -> object_2
    toward = com[1 + index, :2] - pinch[:2]
    if np.linalg.norm(toward) < 1e-6:                             # exactly above the COM: push outward along y
        toward = np.array([0.0, -1.0 if index == 0 else 1.0])
    angle = math.atan2(toward[1], toward[0])
    delta = float(rng.uniform(-GUARD_ANGLE_DEG, GUARD_ANGLE_DEG))
    angle += math.radians(delta)
    return np.array([math.cos(angle), math.sin(angle), 0.0]), index + 1, delta


def draw_sequences(anchor_seed, pinch_xyz, object_com_xyz):
    """8 rollout specs for one anchor; rollout r uses np.random.default_rng([anchor_seed, r])."""
    specs = []
    for r, family in enumerate(FAMILIES):
        spec = {"rollout": r, "family": family, "directions": None, "speeds_mps": None, "guard_index": None, "guard_angle_deg": None}
        if family in ("random", "guard"):
            rng = np.random.default_rng([int(anchor_seed), r])
            dirs, speeds = [], []
            for s in range(SEGMENTS):
                if family == "guard" and s == 0:
                    d, guard, delta = guard_direction(rng, pinch_xyz, object_com_xyz)
                    spec.update(guard_index=guard, guard_angle_deg=delta)
                else:
                    d = random_direction(rng)
                dirs.append(d)
                speeds.append(float(rng.choice(SPEEDS_MPS)))
            spec.update(directions=np.asarray(dirs).tolist(), speeds_mps=speeds)
        specs.append(spec)
    return specs


def reference_box(start_xyz, workspace_low=WORKSPACE_LOW, workspace_high=WORKSPACE_HIGH):
    start = np.asarray(start_xyz, float)
    low = np.maximum(np.minimum(REF_BOX_LOW, start), workspace_low)
    high = np.minimum(np.maximum(REF_BOX_HIGH, start), workspace_high)
    if np.any(low > high):
        raise ValueError("empty reference box")
    return low, high


def reference_path(start_xyz, directions, speeds_mps, low, high, seg_steps=SEG_STEPS, dt=CONTROL_DT):
    """(SEGMENTS * seg_steps, 3): the reference moves speed*dt*d per step from the start pinch, clipped to [low, high]."""
    start = np.asarray(start_xyz, float)
    dirs, speeds = np.asarray(directions, float), np.asarray(speeds_mps, float)
    if start.shape != (3,) or dirs.ndim != 2 or dirs.shape[1] != 3 or speeds.shape != (len(dirs),):
        raise ValueError("reference_path: bad shapes")
    if not np.allclose(np.linalg.norm(dirs, axis=1), 1, atol=1e-9) or np.any(speeds <= 0) or np.any(speeds * dt > 0.05):
        raise ValueError("directions must be unit vectors and speeds within the per-step action range")
    ref, path = start.copy(), []
    for d, v in zip(dirs, speeds):
        for _ in range(seg_steps):
            ref = np.clip(ref + v * dt * d, low, high)
            path.append(ref.copy())
    return np.asarray(path)


def normalize_wm_actions(actions5, normalization):
    """Env normalized actions (..., 5) -> WM input (..., 4): columns 0,1,2,4, (x - mean) / scale of the cache normalization.

    The arithmetic is wm_training.evaluate_bench_counterfactual.normalize_actions exactly (float32 actions, float32
    mean / scale, float32 subtract and divide), so the result is bit-identical to what the evaluator computes from the
    5-column actions (that module imports torch, so it is not imported here; the unit test compares the two)."""
    a = np.asarray(actions5, dtype=np.float32)
    if a.shape[-1] != 5:
        raise ValueError("expected 5 action columns")
    mean64, scale64 = np.asarray(normalization["mean"], float), np.asarray(normalization["scale"], float)
    if mean64.shape != (4,) or scale64.shape != (4,) or np.any(scale64 <= 0) or not np.isfinite(mean64).all():
        raise ValueError("bad normalization")
    mean, scale = np.asarray(normalization["mean"], np.float32), np.asarray(normalization["scale"], np.float32)
    return ((a[..., list(WM_ACTION_COLUMNS)] - mean) / scale).astype(np.float32)


def tracking_error(pinch, reference):
    """(S, 3) pinch states, (S-1, 3) references -> (S-1,) |pinch[k+1] - reference[k]| in metres.

    reference[k] is the target of action k, so the pinch after that action is compared with it. NaN where the
    reference is NaN (the 'recorded' rollout has no reference)."""
    pinch, reference = np.asarray(pinch, float), np.asarray(reference, float)
    if pinch.ndim != 2 or pinch.shape[1] != 3 or reference.shape != (len(pinch) - 1, 3):
        raise ValueError("tracking_error: expected (S, 3) pinch and (S-1, 3) reference")
    return np.linalg.norm(pinch[1:] - reference, axis=1)


def saturated_steps(actions5):
    """(..., 5) executed env actions -> (...) bool: dx, dy or dz sits on the +-1 (= +-5 cm) action bound."""
    a = np.asarray(actions5, float)
    if a.shape[-1] != 5:
        raise ValueError("expected 5 action columns")
    return (np.abs(a[..., :3]) >= 1.0 - SATURATION_TOL).any(-1)


# ----- aggregation used by stage 2 (numpy only so it is tested anywhere) ---------------------------------
def rollout_outcome(tilt, unsafe, class_label, pinch, contact_peak, green_contact_peak, reference=None, actions=None,
                    wm_action=None, horizons=(8, 16, 24)):
    """Per-rollout numbers from (25, ...) arrays; step 0 is the anchor, steps 1..h are the future.

    reference (24, 3) / actions (24, 5) / wm_action (24, 4) are optional; the fields that need them are None without
    them (and the tracking fields are None for a NaN reference, i.e. the 'recorded' rollout)."""
    tilt, pinch = np.asarray(tilt, float), np.asarray(pinch, float)
    worst = tilt.max(1)
    hits = np.flatnonzero(np.asarray(unsafe, bool)[1:])
    disp = np.linalg.norm(pinch - pinch[0], axis=1)
    out = {"anchor_tilt_deg": float(worst[0]), "first_unsafe_step": int(hits[0]) + 1 if len(hits) else None,
           "class0_frames": int((np.asarray(class_label)[1:] == 0).sum()), "anchor_class": int(np.asarray(class_label)[0]),
           "pinch_disp_max_m": float(disp.max()), "pinch_disp_final_m": float(disp[-1]),
           "contact_steps": int((np.asarray(contact_peak)[1:] > 1e-3).any(1).sum()),
           "green_contact_steps": int((np.asarray(green_contact_peak)[1:] > 1e-3).any(1).sum()),
           "anchor_pinch_z_m": float(pinch[0, 2]), "pinch_min_z_m": float(pinch[1:, 2].min())}
    for h in horizons:
        m = float(worst[1:h + 1].max())
        out[f"max_tilt_h{h}"] = m
        out[f"rise_h{h}"] = m - float(worst[0])
    out.update(tracking_error_final_m=None, tracking_error_max_m=None, reference_min_z_m=None, reference_below_free_floor=None)
    if reference is not None and np.isfinite(np.asarray(reference, float)).all():
        reference = np.asarray(reference, float)
        err = tracking_error(pinch, reference)
        out.update(tracking_error_final_m=float(err[-1]), tracking_error_max_m=float(err.max()),
                   reference_min_z_m=float(reference[:, 2].min()),
                   reference_below_free_floor=bool(reference[:, 2].min() < min(FREE_FLOOR_Z_M, float(pinch[0, 2]))))
    out["saturated_steps"] = None if actions is None else int(saturated_steps(actions).sum())
    out["wm_action_absmax"] = None if wm_action is None else float(np.abs(np.asarray(wm_action, float)).max())
    return out


def _share(rows, test):
    return int(sum(1 for r in rows if test(r)))


def _tilt_counts(sel):
    return {"rollouts": len(sel),
            **{f"over30_h{h}": _share(sel, lambda r, h=h: r[f"max_tilt_h{h}"] > 30) for h in (8, 16, 24)},
            **{f"over10_h{h}": _share(sel, lambda r, h=h: r[f"max_tilt_h{h}"] >= DISTURBED_DEG) for h in (8, 16, 24)},
            **{f"below5_h{h}": _share(sel, lambda r, h=h: r[f"max_tilt_h{h}"] < 5) for h in (8, 16, 24)}}


def _flag(r, key):
    return r.get(key) is not None and bool(r[key])


def _above(r, key, threshold):
    return r.get(key) is not None and r[key] > threshold


def aggregate_outcomes(rows):
    """rows: dicts with group, set, family and the rollout_outcome keys -> counts per (set, group, family).

    Besides the plain tilt counts, every block separates the rollouts whose anchor already has a guard >= 10 deg
    (anchor_tilt_ge10; the anchor_tilt_lt10 sub-block counts only rollouts that start below 10 deg, where a positive
    is a new disturbance and not an already tilted guard staying tilted), and reports how well the pinch followed its
    reference (tracking; None for 'recorded'), how many rollouts contain an action on the +-5 cm bound and how large
    the WM action input gets (in units of the cache normalization scale)."""
    def block(sel):
        if not sel:
            return None
        disp = np.array([r["pinch_disp_max_m"] for r in sel])
        final = np.array([r["pinch_disp_final_m"] for r in sel])
        tracked = [r for r in sel if r.get("tracking_error_final_m") is not None]
        tracking = None
        if tracked:
            err = np.array([r["tracking_error_final_m"] for r in tracked])
            tracking = {"rollouts": len(tracked), "final_gt1cm": int((err > 0.01).sum()), "final_gt2cm": int((err > TRACKING_LOST_M).sum()),
                        "final_gt5cm": int((err > 0.05).sum()), "final_median_cm": float(100 * np.median(err)),
                        "final_p90_cm": float(100 * np.quantile(err, 0.9)), "final_max_cm": float(100 * err.max()),
                        "any_step_gt2cm": _share(tracked, lambda r: r["tracking_error_max_m"] > TRACKING_LOST_M),
                        "final_gt2cm_with_saturated_action": _share(tracked, lambda r: r["tracking_error_final_m"] > TRACKING_LOST_M and _above(r, "saturated_steps", 0)),
                        "final_gt2cm_with_contact": _share(tracked, lambda r: r["tracking_error_final_m"] > TRACKING_LOST_M and r["contact_steps"] > 0)}
        wm = np.array([r["wm_action_absmax"] for r in sel if r.get("wm_action_absmax") is not None])
        return {**_tilt_counts(sel),
                "anchor_tilt_ge10": _share(sel, lambda r: r["anchor_tilt_deg"] >= DISTURBED_DEG),
                **{f"rise10_h{h}": _share(sel, lambda r, h=h: r[f"rise_h{h}"] >= DISTURBED_DEG) for h in (8, 16, 24)},
                "anchor_tilt_lt10": _tilt_counts([r for r in sel if r["anchor_tilt_deg"] < DISTURBED_DEG]),
                "with_class0": _share(sel, lambda r: r["class0_frames"] > 0),
                "with_contact": _share(sel, lambda r: r["contact_steps"] > 0),
                "pinch_disp_max_cm": {"median": float(100 * np.median(disp)), "p90": float(100 * np.quantile(disp, 0.9)), "max": float(100 * disp.max())},
                "pinch_disp_final_cm": {"median": float(100 * np.median(final)), "p90": float(100 * np.quantile(final, 0.9))},
                "tracking": tracking,
                "with_saturated_action": _share(sel, lambda r: _above(r, "saturated_steps", 0)),
                "wm_action_absmax": None if not len(wm) else {"median": float(np.median(wm)), "p90": float(np.quantile(wm, 0.9)), "max": float(wm.max()),
                                                              "gt5": int((wm > 5).sum()), "gt10": int((wm > 10).sum())},
                "reference_below_free_floor": _share(sel, lambda r: _flag(r, "reference_below_free_floor")),
                "pinch_below_free_floor": _share(sel, lambda r: r["pinch_min_z_m"] < FREE_FLOOR_Z_M)}

    families = sorted({r["family"] for r in rows}, key=lambda f: FAMILIES.index(f))
    out = {"overall": {f: block([r for r in rows if r["family"] == f]) for f in families}}
    for key in ("set", "group"):
        out[f"by_{key}"] = {}
        for value in sorted({r[key] for r in rows}):
            out[f"by_{key}"][value] = {f: block([r for r in rows if r[key] == value and r["family"] == f]) for f in families}
    return out


def criteria_pools(rows):
    """Truth pools of the pre-declared S1 / S2 per family (the criteria themselves are evaluated elsewhere).

    S1 (h = 16): positives = worst guard > 30 deg within steps 1..16, negatives = worst guard < 5 deg throughout.
    S2 (h = 8, 16): positives = worst guard >= 10 deg within steps 1..h, negatives = the rest.
    Every pool is split by the anchor's worst guard tilt (< 10 deg: a positive is a new disturbance; >= 10 deg: an
    already tilted guard, where predicting 'stays tilted' earns the positive), by source group, and by whether the
    rollout contains an action on the +-5 cm bound (far outside the WM's training actions, see RANDOM_BRANCHES.md)."""
    def parts(sel):
        groups = sorted({r["group"] for r in sel})
        return {"all": len(sel), "anchor_tilt_lt10": _share(sel, lambda r: r["anchor_tilt_deg"] < DISTURBED_DEG),
                "anchor_tilt_ge10": _share(sel, lambda r: r["anchor_tilt_deg"] >= DISTURBED_DEG),
                "no_saturated_action": _share(sel, lambda r: not _above(r, "saturated_steps", 0)),
                "anchor_tilt_lt10_no_saturated_action": _share(sel, lambda r: r["anchor_tilt_deg"] < DISTURBED_DEG and not _above(r, "saturated_steps", 0)),
                "by_group": {g: _share(sel, lambda r, g=g: r["group"] == g) for g in groups}}

    def pool(sel, positive, negative):
        return {"positives": parts([r for r in sel if positive(r)]), "negatives": parts([r for r in sel if negative(r)])}

    out = {}
    for family in sorted({r["family"] for r in rows}, key=lambda f: FAMILIES.index(f)):
        sel = [r for r in rows if r["family"] == family]
        out[family] = {"S1_h16": pool(sel, lambda r: r["max_tilt_h16"] > 30, lambda r: r["max_tilt_h16"] < 5),
                       **{f"S2_h{h}": pool(sel, lambda r, h=h: r[f"max_tilt_h{h}"] >= DISTURBED_DEG,
                                           lambda r, h=h: r[f"max_tilt_h{h}"] < DISTURBED_DEG) for h in (8, 16)}}
    return out


# ----- stage 1 CLI (training venv: needs h5py for the cache pixels) --------------------------------------
def load_npz_fields(path, fields=NPZ_FIELDS):
    with np.load(path, allow_pickle=False) as npz:
        missing = [k for k in fields if k not in npz.files]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        arrays = {k: np.asarray(npz[k]) for k in fields}
    n = len(arrays["time"])
    shapes = {"guard_tilt_deg": (n, 2), "pinch_xyz": (n, 3), "object_com_xyz": (n, 3, 3), "grasped": (n,), "hover_error_m": (n,),
              "unsafe": (n,), "class_label": (n,)}
    for k, s in shapes.items():
        if k in arrays and arrays[k].shape != s:
            raise ValueError(f"{path}: {k} has shape {arrays[k].shape}, expected {s}")
    for k in fields:
        if arrays[k].dtype.kind == "f" and not np.isfinite(arrays[k]).all():
            raise ValueError(f"{path}: non-finite {k}")
    return arrays


def anchor_state(arrays, frame):
    return {"pinch_xyz": arrays["pinch_xyz"][frame].tolist(), "green_com_xyz": arrays["object_com_xyz"][frame][0].tolist(),
            "guard_tilt_deg": arrays["guard_tilt_deg"][frame].tolist(), "grasped": bool(arrays["grasped"][frame]),
            "hover_error_m": float(arrays["hover_error_m"][frame]), "class_label": int(arrays["class_label"][frame]),
            "unsafe": bool(arrays["unsafe"][frame])}


def make_anchor_record(index, anchor, anchor_seed, source, arrays):
    """The anchors.json entry of one anchor (selection fields + source + recorded state at a + the 8 rollout specs)."""
    frame = int(anchor["frame"])
    one = slice(frame, frame + 1)
    rule = classify_frames(arrays["pinch_xyz"][one], arrays["object_com_xyz"][one], arrays["grasped"][one],
                           arrays["hover_error_m"][one], arrays["guard_tilt_deg"][one])[0]
    return {"id": int(index), **anchor, "prefix_start": frame - PREFIX, "seed": int(anchor_seed), "source": source,
            "state": anchor_state(arrays, frame), "stratum_by_rule": str(rule),
            "sequences": draw_sequences(anchor_seed, arrays["pinch_xyz"][frame], arrays["object_com_xyz"][frame])}


def read_cache_pixels(h5_path, frame, expected_npz_sha256):
    import h5py                                                  # lazy: only the stage-1 CLI needs it
    with h5py.File(h5_path, "r") as f:
        if not bool(f.attrs.get("complete", False)):
            raise ValueError(f"{h5_path}: cache episode is not complete")
        stored = f.attrs["source_npz_sha256"]
        stored = stored.decode() if isinstance(stored, bytes) else str(stored)
        if stored != expected_npz_sha256:
            raise ValueError(f"{h5_path}: cache was built from a different trajectory.npz")
        pixels = np.asarray(f["pixels"][frame - PREFIX:frame + 1])
        n_states = f["pixels"].shape[0]
    if pixels.shape != (PREFIX + 1, 224, 224, 3) or pixels.dtype != np.uint8:
        raise ValueError(f"{h5_path}: unexpected pixel block {pixels.shape} {pixels.dtype}")
    return pixels, n_states


def build_plan(a):
    root = Path(__file__).resolve().parent
    full_index_path, bench_index_path = a.full_cache / "index.json", a.bench_cache / "index.json"
    inputs = {str(p): sha256(p) for p in (full_index_path, bench_index_path, a.classifier_run, a.bench_split)}
    full_index, bench_index = read_json(full_index_path), read_json(bench_index_path)
    classifier_run, split = read_json(a.classifier_run), read_json(a.bench_split)
    if full_index.get("status") != "complete" or bench_index.get("status") != "complete" or not bench_index.get("eval_cache"):
        raise ValueError("caches must be complete (and the bench cache an evaluation cache)")
    if bench_index.get("split_sha256") != inputs[str(a.bench_split)]:
        raise ValueError("the bench cache was not built from this split file")
    normalization = full_index["normalization"]
    for key in ("mean", "scale"):
        if not np.array_equal(normalization[key], bench_index["normalization"][key]):
            raise ValueError("the two caches use different action normalizations")
    if list(full_index["source_action_columns"]) != [0, 1, 2, 4]:
        raise ValueError("unexpected cache action columns")
    val_rows = {int(r["id"]): r for r in full_index["episodes"]["val"]}
    test_ids = {int(r["id"]) for r in full_index["episodes"]["test"]}
    report_ids = [int(i) for i in classifier_run["report_episode_ids"]]
    dev_ids = {int(i) for i in classifier_run["dev_episode_ids"]}
    if not set(report_ids) <= set(val_rows) or set(report_ids) & (test_ids | dev_ids) or len(set(report_ids)) != len(report_ids):
        raise ValueError("report_episode_ids must be distinct val episodes outside the dev half and the reserved test split")
    raw_root = Path(full_index["source_root"])
    manifest_sha = sha256(raw_root / "manifest.json")
    if manifest_sha != full_index["source_manifest_sha256"]:
        raise ValueError(f"{raw_root}: manifest differs from the one the cache was built from")
    ss = np.random.SeedSequence(a.seed)
    seed_v, seed_falls, seed_neg, seed_seq = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in ss.spawn(4)]
    started = time.perf_counter()

    # ---- Set V
    episodes, sources = {}, {}
    for eid in report_ids:
        folder = raw_root / f"episode_{eid:04d}"
        npz_sha = sha256(folder / "trajectory.npz")
        if npz_sha != val_rows[eid]["source_npz_sha256"] or sha256(folder / "metadata.json") != val_rows[eid]["source_metadata_sha256"]:
            raise ValueError(f"{folder}: raw files differ from the ones the cache was built from")
        episodes[eid] = load_npz_fields(folder / "trajectory.npz")
        sources[("V", eid)] = {"kind": "wm_full_001_val_report", "dataset_dir": str(raw_root), "dataset_manifest_sha256": manifest_sha,
                               "episode_id": eid, "global_id": None, "episode_dir": str(folder), "npz_sha256": npz_sha,
                               "metadata_sha256": val_rows[eid]["source_metadata_sha256"], "frames": int(len(episodes[eid]["time"])),
                               "cache_h5": str(a.full_cache / val_rows[eid]["path"]), "cache_episode_id": eid}
    quotas = {s: int(round(q * a.v_scale)) for s, q in V_QUOTA.items()}
    v_anchors, v_available = select_set_v(episodes, quotas, seed_v)

    # ---- Set B
    datasets = {}
    for row in split["datasets"]:
        folder = Path(row["path"])
        if sha256(folder / "manifest.json") != row["manifest_sha256"]:
            raise ValueError(f"{folder}: manifest changed since the split was written")
        datasets[int(row["index"])] = {"path": folder, "manifest_sha256": row["manifest_sha256"]}
    test_members = {(int(e["dataset"]), int(e["episode_id"])) for part in split["strata"].values() for e in part["test"]}
    dev_members = {(int(e["dataset"]), int(e["episode_id"])) for part in split["strata"].values() for e in part["dev"]}
    if test_members & dev_members:
        raise ValueError("bench-dev and bench-test overlap")
    bench_rows = {(int(r["source_dataset"]), int(r["source_id"])): r for r in bench_index["episodes"]["val"]}
    if set(bench_index["bench_split"]["bench_test"]) != {d * GLOBAL_ID_STRIDE + e for d, e in test_members}:
        raise ValueError("bench cache bench_test ids disagree with the split")
    entries = [e for e in split["episodes"] if (int(e["dataset"]), int(e["episode_id"])) in test_members]
    falls = [e for e in entries if e.get("onset_frame") is not None and "|" in e["stratum"] and e["stratum"].split("|", 1)[1] in SEVERITIES]
    negatives = [e for e in entries if e.get("onset_frame") is None and e["stratum"] in NEGATIVE_STRATA]
    b_anchors, skipped, bench_arrays = [], [], {}

    def bench_source(entry):
        key = (int(entry["dataset"]), int(entry["episode_id"]))
        folder = datasets[key[0]]["path"] / entry["path"]
        row = bench_rows[key]
        npz_sha = sha256(folder / "trajectory.npz")
        if npz_sha != row["source_npz_sha256"] or sha256(folder / "metadata.json") != row["source_metadata_sha256"]:
            raise ValueError(f"{folder}: raw files differ from the ones the bench cache was built from")
        arrays = load_npz_fields(folder / "trajectory.npz")
        if len(arrays["time"]) != int(entry["frames"]):
            raise ValueError(f"{folder}: frame count differs from the split")
        onsets = np.flatnonzero(arrays["unsafe"])
        if (int(onsets[0]) if len(onsets) else None) != entry.get("onset_frame"):
            raise ValueError(f"{folder}: onset differs from the split")
        gid = key[0] * GLOBAL_ID_STRIDE + key[1]
        src = {"kind": "danger_bench", "dataset_dir": str(datasets[key[0]]["path"]), "dataset_index": key[0],
               "dataset_manifest_sha256": datasets[key[0]]["manifest_sha256"], "episode_id": key[1], "global_id": gid,
               "episode_dir": str(folder), "npz_sha256": npz_sha, "metadata_sha256": row["source_metadata_sha256"],
               "frames": int(entry["frames"]), "onset_frame": entry.get("onset_frame"), "recipe": entry.get("recipe"),
               "cache_h5": str(a.bench_cache / row["path"]), "cache_episode_id": gid}
        return arrays, src

    rng_falls = np.random.default_rng(seed_falls)
    for entry in round_robin(falls, lambda e: e["stratum"], seed_falls):
        if sum(1 for x in b_anchors if x["group"] == "B/fall") >= a.b_falls:
            break
        arrays, src = bench_source(entry)
        got = fall_anchor(rng_falls, int(entry["onset_frame"]), int(entry["frames"]), arrays["guard_tilt_deg"])
        if got is None:
            skipped.append({"dataset": src["dataset_index"], "episode_id": src["episode_id"], "reason": "no admissible onset-k for k in 5..30"})
            continue
        k, frame, draws = got
        bench_arrays[src["global_id"]] = arrays
        sources[("B", src["global_id"])] = src
        b_anchors.append({"set": "B", "group": "B/fall", "stratum": entry["stratum"], "episode_id": src["global_id"], "frame": frame,
                          "lead_k": k, "k_draws": draws})
    rng_neg = np.random.default_rng(seed_neg)
    for entry in round_robin(negatives, lambda e: e["stratum"], seed_neg):
        if sum(1 for x in b_anchors if x["group"] == "B/negative") >= a.b_negatives:
            break
        arrays, src = bench_source(entry)
        frame = negative_anchor(rng_neg, arrays)
        if frame is None:
            skipped.append({"dataset": src["dataset_index"], "episode_id": src["episode_id"], "reason": "no admissible frame with the gripper near a guard"})
            continue
        bench_arrays[src["global_id"]] = arrays
        sources[("B", src["global_id"])] = src
        b_anchors.append({"set": "B", "group": "B/negative", "stratum": entry["stratum"], "episode_id": src["global_id"], "frame": frame,
                          "lead_k": None, "k_draws": None})

    # ---- sequences, anchor records, cache pixels
    anchors = v_anchors + b_anchors
    seq_seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in np.random.SeedSequence(seed_seq).spawn(len(anchors))]
    records, pixels = [], {}
    for i, (anchor, anchor_seed) in enumerate(zip(anchors, seq_seeds)):
        arrays = episodes[anchor["episode_id"]] if anchor["set"] == "V" else bench_arrays[anchor["episode_id"]]
        src = sources[(anchor["set"], anchor["episode_id"])]
        frame = anchor["frame"]
        block, n_states = read_cache_pixels(src["cache_h5"], frame, src["npz_sha256"])
        if n_states != src["frames"]:
            raise ValueError(f"{src['cache_h5']}: frame count differs from the raw episode")
        pixels[f"a{i:04d}"] = block
        records.append(make_anchor_record(i, anchor, anchor_seed, src, arrays))
    return {"records": records, "pixels": pixels, "inputs": inputs, "normalization": normalization, "quotas": quotas,
            "v_available": v_available, "skipped": skipped, "seeds": {"base": a.seed, "v": seed_v, "falls": seed_falls, "negatives": seed_neg,
                                                                       "sequences": seed_seq},
            "counts": {"V": dict(Counter(r["stratum"] for r in records if r["set"] == "V")),
                       "B": dict(Counter(r["group"] for r in records if r["set"] == "B")),
                       "B_strata": dict(Counter(r["stratum"] for r in records if r["set"] == "B"))},
            "pools": {"v_report_episodes": len(report_ids), "bench_test_fall_events": len(falls), "bench_test_negatives": len(negatives)},
            "elapsed_seconds": time.perf_counter() - started, "root": str(root)}


def write_plan_report(output, summary):
    c = summary["counts"]
    lines = ["# 随机动作分支：起点与动作序列计划（第 1 阶段）\n",
             f"- V 集（wm_full_001 val 报告半 {summary['pools']['v_report_episodes']} 条轨迹）：各层锚点 {json.dumps(c['V'], ensure_ascii=False)}（目标 {json.dumps(summary['quotas'], ensure_ascii=False)}）；可用帧 {json.dumps(summary['v_available'], ensure_ascii=False)}。",
             f"- B 集（bench_test）：{json.dumps(c['B'], ensure_ascii=False)}；层 {json.dumps(c['B_strata'], ensure_ascii=False)}；跳过 {len(summary['skipped'])} 条轨迹。",
             f"- 每个锚点 8 条 rollout：{', '.join(FAMILIES)}；随机段 {SEGMENTS}×{SEG_STEPS} 步，速度 ∈ {SPEEDS_MPS} m/s，方向 N(0, diag{DIRECTION_STD}²) 归一化；guard 首段水平指向较近红柱 ±{GUARD_ANGLE_DEG:g}°。",
             f"- 缓存像素：每个锚点复制 a−{PREFIX}..a 共 {PREFIX + 1} 帧（`cache_pixels.npz`），供第 2 阶段逐像素比对。",
             "- 没有打开 wm_full_001 的保留 TEST 轨迹；没有在任何轨迹上拟合任何东西（只选起点）。",
             f"- 用时 {summary['elapsed_seconds']:.1f} s。"]
    (Path(output) / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="NEW run directory (e.g. runs/random_branches_plan_001)")
    p.add_argument("--full-cache", type=Path, default=Path("data/wm_full_001"))
    p.add_argument("--bench-cache", type=Path, default=Path("data/wm_bench_001"))
    p.add_argument("--classifier-run", type=Path, default=Path("runs/safety_classifier_001/run.json"), help="holds dev/report_episode_ids")
    p.add_argument("--bench-split", type=Path, default=Path("data/danger_bench_split_001/benchmark_split.json"))
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--v-scale", type=float, default=1.0, help="multiplies the Set V quotas 40/40/40/30")
    p.add_argument("--b-falls", type=int, default=100)
    p.add_argument("--b-negatives", type=int, default=50)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    if a.seed < 0 or not 0 < a.v_scale <= 1 or a.b_falls < 0 or a.b_negatives < 0:
        raise ValueError("seed >= 0, 0 < v-scale <= 1, non-negative B counts")
    for path in (a.full_cache / "index.json", a.bench_cache / "index.json", a.classifier_run, a.bench_split):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    from wm_training.provenance import implementation_hashes
    root = Path(__file__).resolve().parent
    output.mkdir(parents=True)
    run = {"format": PLAN_FORMAT, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "input_sha256": {str(p): sha256(p) for p in (a.full_cache / "index.json", a.bench_cache / "index.json", a.classifier_run, a.bench_split)},
           "implementation_sha256": {**implementation_hashes(IMPLEMENTATION_FILES, package_dir=root),
                                     **implementation_hashes(("danger_strata.py", "provenance.py"))},
           "python": platform.python_version(), "numpy": np.__version__,
           "rules": {"prefix": PREFIX, "horizon": HORIZON, "min_anchor": MIN_ANCHOR, "skip_tilt_deg": SKIP_TILT_DEG, "min_gap_frames": MIN_GAP_FRAMES,
                     "near_radius_m": NEAR_RADIUS_M, "near_grip_max_z_m": NEAR_GRIP_MAX_Z_M, "green_near_max_z_m": GREEN_NEAR_MAX_Z_M,
                     "near_guard_max_tilt_deg": NEAR_GUARD_MAX_TILT_DEG, "goal_near_m": GOAL_NEAR_M, "v_priority": list(V_PRIORITY),
                     "fall_k": list(FALL_K), "negative_strata": list(NEGATIVE_STRATA), "families": list(FAMILIES), "speeds_mps": list(SPEEDS_MPS),
                     "direction_std": list(DIRECTION_STD), "guard_angle_deg": GUARD_ANGLE_DEG, "ref_box_low": REF_BOX_LOW.tolist(),
                     "ref_box_high": REF_BOX_HIGH.tolist(), "workspace_low": WORKSPACE_LOW.tolist(), "workspace_high": WORKSPACE_HIGH.tolist()},
           "test_episodes_opened": False, "fitted_on_evaluation_data": False}
    write_new_json(output / "run.json", run)
    try:
        plan = build_plan(a)
        anchors = {"format": PLAN_FORMAT, "normalization": {k: plan["normalization"][k] for k in ("names", "mean", "scale")},
                   "wm_action_columns": [0, 1, 2, 4], "prefix": PREFIX, "horizon": HORIZON, "families": list(FAMILIES),
                   "rules": run["rules"], "anchors": plan["records"]}
        write_new_json(output / "anchors.json", anchors)
        with (output / "cache_pixels.npz").open("xb") as handle:
            np.savez_compressed(handle, **plan["pixels"])
        summary = {"format": PLAN_FORMAT, "anchors": len(plan["records"]), "counts": plan["counts"], "quotas": plan["quotas"],
                   "v_available": plan["v_available"], "skipped": plan["skipped"], "pools": plan["pools"], "seeds": plan["seeds"],
                   "files_sha256": {"anchors.json": sha256(output / "anchors.json"), "cache_pixels.npz": sha256(output / "cache_pixels.npz")},
                   "elapsed_seconds": plan["elapsed_seconds"], "test_episodes_opened": False}
        write_plan_report(output, summary)
        write_new_json(output / "summary.json", summary)
        print(json.dumps(native({k: summary[k] for k in ("anchors", "counts", "v_available")}), ensure_ascii=False), flush=True)
        return summary
    except BaseException as error:
        write_new_json(output / "FAILED.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
