"""Yield tables of RRT-action-law branch datasets (collect_rrt_branches.py output, format rrt_branches_v1).

numpy only (runs in any venv, also the Windows numpy-only python). Reads the datasets' manifest.json and every
anchor npz (sha256-checked against the manifest; the pixel arrays are never decompressed) and the collector's report
summary.json (time / memory / disk). Every number is recomputed from the npz arrays, not copied from the manifest's
per-rollout outcomes (the per-anchor recorded continuation is cross-checked against the manifest's source tilt).

Definitions (worst guard = the larger of the two guard tilts; state 0 = anchor frame a, steps 1..h = the future):
  max_h           = max worst tilt over states 1..h, h in HORIZONS (8, 16, 20, 24, 32)
  reach10_h       = max_h >= 10 deg;          topple_h = max_h > 30 deg;      rise10_h = max_h - anchor tilt >= 10 deg
  law rollout     = every family except 'recorded' (the recorded continuation duplicates the source episode)
  action_caused_h = law rollout with topple_h whose recorded sibling does NOT exceed 30 deg within h
  fresh start     = anchor with both guards < 5 deg;  fresh topple_h = law rollout of a fresh start with topple_h
  contrastive_h   = anchor whose 8 rollouts (recorded included) contain a topple_h AND one with max_h < 10 deg
                    ('strict', the task's definition); 'loose': a topple_h AND one with max_h <= 30 deg
  calm_h          = max_h < 5 deg (the C2 pool); C3 pair = two rollouts of one anchor whose max_20 differ by >= 10 deg
  free space      = collect_rrt_branches.contact_state rule (steps 1..32: gripper never touches guards / floor; the
                    held block never touches floor / guards; an empty gripper never touches the green block)
  decision points (C1 / C2 pools as the receding-horizon planner meets them; decision_point_instances): d in {0, 8}; an
                    instance (law rollout, d) is valid if the worst tilt at state d <= 25 deg and no state 1..d exceeded 30;
                    window = states d+1..d+20; positive > 30, action-caused = positive and recorded max over 1..d+20 <= 30,
                    fresh = positive and tilt at d < 5, calm = window max < 5. Subgroups of an instance: grasped (block held
                    at state d: the planner always carries it) and planner_shape (the window's actions are ONE edge followed by
                    hold only = the planner's check shape edge 8 + hold 12). One rollout can give an instance at d = 0 AND at
                    d = 8: the 'distinct' counts count rollouts with at least one such instance.
  near-goal frames (law rollouts, states 1..32, d = |green COM - goal_xyz| with goal_xyz from the npz):
                    boundary_slow  = grasped & green speed <= 2 cm/s & 0.8 cm <= d <= 2.2 cm
                    speed_boundary = grasped & 1 cm/s <= green speed <= 2.5 cm/s & d <= 1.5 cm
                    class0         = hover_conditions_met (grasped & d <= 1.5 cm & speed <= 2 cm/s)
  executed speed    |pinch[k+1] - pinch[k]| per step (mm per 0.05 s step; the law's REFERENCE moves exactly 1 mm per edge
                    step = 2 cm/s) split into edge steps of the first segment (the arm may still carry the recorded
                    controller's motion), later edge steps and hold steps; per edge also the straight-line pinch displacement
                    over its 8 steps / 8 (the edge-average speed).
The summary also exports 'anchor_table' (every anchor's rollouts and decision-point instances, compact) so that the
stage-2 plan (rrt_data_plan.py) can recompute any per-anchor yield without the npz files. Doc: RRT_PILOT_YIELDS.md.
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

import rrt_action_law as law
from wm_training.danger_strata import native, read_json, sha256, write_new_json

FORMAT = "rrt_pilot_yields_v1"          # v1 kept: the additions (instances, speeds, anchor_table) are new keys only
DATASET_FORMAT = "rrt_branches_v1"
HORIZONS = (8, 16, 20, 24, 32)
STRATA = ("T", "F", "G", "M")
FRESH_MAX_DEG = 5.0
REACH_DEG, TOPPLE_DEG, CALM_DEG = 10.0, 30.0, 5.0
PAIR_GAP_DEG = 10.0
SLOW_MPS, SPEED_BAND_MPS = 0.02, (0.01, 0.025)
BOUNDARY_SHELL_M, WITHIN_M = (0.008, 0.022), 0.015
FLAGS = ("reference_frozen", "reference_dragged", "integral_clamped", "box_clipped", "range_clipped", "wm_clipped")
NPZ_KEYS = ("families", "stratum", "set", "anchor_id", "guard_tilt_deg", "grasped", "target_speed_mps", "object_com_xyz",
            "goal_xyz", "hover_conditions_met", "robot_env_contact", "robot_green_contact", "green_env_contact", "actions",
            "wm_action", "nominal_actions", "nominal_open_loop_actions", "tracking_error_m", "source_guard_tilt_deg",
            "anchor_velocity_xyz", "segment_index", "segment_kind", "pinch_xyz", *FLAGS)
SLOW_ANCHOR_MPS = 0.025              # anchor pinch speed split of the nominal-mismatch breakdown (the law moves at 2 cm/s)
BOOT = 2000
IMPLEMENTATION_FILES = ("rrt_pilot_yields.py", "rrt_action_law.py")


# ----- per anchor -------------------------------------------------------------------------------------------------
def rollout_rows(z, normalization):
    """Per-rollout dicts of one anchor's arrays (z: NPZ_KEYS -> numpy arrays)."""
    families = [str(f) for f in z["families"]]
    R = len(families)
    tilt = np.asarray(z["guard_tilt_deg"], float)
    if tilt.shape[:2] != (R, law.EDGE_STEPS * 4 + 1) or tilt.shape[2] != 2 or families[0] != "recorded":
        raise ValueError("unexpected anchor arrays (need R x 33 x 2 tilt and rollout 0 = recorded)")
    worst = tilt.max(-1)
    grasped, speed = np.asarray(z["grasped"], bool), np.asarray(z["target_speed_mps"], float)
    dist = np.linalg.norm(np.asarray(z["object_com_xyz"], float)[:, :, 0] - np.asarray(z["goal_xyz"], float), axis=-1)
    wm = np.asarray(z["wm_action"], float)[..., :3]
    in_range = (wm >= law.WM_RANGE_LOW) & (wm <= law.WM_RANGE_HIGH)
    rows = []
    for r, family in enumerate(families):
        maxes = {h: float(worst[r, 1:h + 1].max()) for h in HORIZONS}
        held = bool(grasped[r, 0])
        touch = bool(np.asarray(z["robot_env_contact"])[r, 1:].any()
                     or (np.asarray(z["green_env_contact"])[r, 1:].any() if held else np.asarray(z["robot_green_contact"])[r, 1:].any()))
        s = slice(1, None)
        g, v, d = grasped[r, s], speed[r, s], dist[r, s]
        row = {"rollout": r, "family": family, "law": family != "recorded", "anchor_tilt": float(worst[r, 0]),
               "max": maxes, "free_space": not touch, "grasped_start": held, "grasped_end": bool(grasped[r, -1]),
               "frames": {"boundary_slow": int((g & (v <= SLOW_MPS) & (d >= BOUNDARY_SHELL_M[0]) & (d <= BOUNDARY_SHELL_M[1])).sum()),
                          "speed_boundary": int((g & (v >= SPEED_BAND_MPS[0]) & (v <= SPEED_BAND_MPS[1]) & (d <= WITHIN_M)).sum()),
                          "class0": int(np.asarray(z["hover_conditions_met"], bool)[r, s].sum())},
               "wm_steps": int(in_range[r].shape[0]), "wm_steps_in_range": int(in_range[r].all(-1).sum())}
        if family != "recorded":
            track = np.asarray(z["tracking_error_m"], float)[r]
            row["tracking_final_m"], row["tracking_max_m"] = float(track[-1]), float(track.max())
            row["flag_steps"] = {k: int(np.asarray(z[k])[r].any(-1).sum() if np.asarray(z[k]).ndim == 3 else np.asarray(z[k])[r].sum())
                                 for k in FLAGS}
            rec = law.action_mismatch(np.asarray(z["nominal_actions"], float)[r], np.asarray(z["actions"], float)[r], normalization)
            ol = law.action_mismatch(np.asarray(z["nominal_open_loop_actions"], float)[r], np.asarray(z["actions"], float)[r], normalization)
            row["mismatch_mm"] = rec["mm"].max(-1)            # (32,) worst axis per step, receding horizon
            row["mismatch_wm"] = rec["wm"].max(-1)
            row["mismatch_open_loop_mm"] = ol["mm"].max(-1)
            row.update(executed_speed(np.asarray(z["pinch_xyz"], float)[r], np.asarray(z["segment_index"])[r], np.asarray(z["segment_kind"])[r]))
        rows.append(row)
    return rows


def executed_speed(pinch, segment_index, segment_kind):
    """Executed pinch speed of one law rollout: pinch (33, 3), segment_index (32,), segment_kind (4,) ->
    {'speed_mm': (32,) |pinch[k+1] - pinch[k]| in mm per step, 'step_kind': (32,) 'edge' / 'hold',
     'step_segment': (32,) segment number, 'edge_average_mm': [(segment, straight-line displacement over the edge / 8)]}."""
    pinch, idx, kinds = np.asarray(pinch, float), np.asarray(segment_index), np.asarray(segment_kind)
    if pinch.shape != (law.EDGE_STEPS * 4 + 1, 3) or idx.shape != (law.EDGE_STEPS * 4,) or (idx < 0).any():
        raise ValueError("executed_speed needs a law rollout (33 pinch states, 32 segment indices >= 0)")
    speed = 1000.0 * np.linalg.norm(np.diff(pinch, axis=0), axis=1)
    step_kind = kinds[idx]
    edges = []
    for seg in sorted(set(idx.tolist())):
        steps = np.flatnonzero(idx == seg)
        if kinds[seg] == "edge":
            if len(steps) != law.EDGE_STEPS or steps[-1] - steps[0] != law.EDGE_STEPS - 1:
                raise ValueError("an edge must span 8 consecutive steps")
            edges.append((int(seg), float(1000.0 * np.linalg.norm(pinch[steps[-1] + 1] - pinch[steps[0]]) / law.EDGE_STEPS)))
    return {"speed_mm": speed, "step_kind": step_kind, "step_segment": idx.astype(int), "edge_average_mm": edges}


DECISION_POINTS = (0, 8)             # the planner re-reads the image every 8 executed steps
DECISION_WINDOW = 20                 # the safety check's horizon (edge 8 + hold 12)
DECISION_MAX_START_DEG = 25.0        # like an anchor: worst tilt at the decision point <= 25 deg


DP_KEYS = ("valid", "fresh_valid", "positive", "action_caused", "fresh", "calm")
SUBGROUPS = {"grasped": lambda i: i["grasped"], "planner_shape": lambda i: i["planner_shape"],
             "grasped_planner_shape": lambda i: i["grasped"] and i["planner_shape"]}


def planner_shape(segment_index, segment_kind, d):
    """(R,) bool: the actions d..d+19 of rollout r are ONE edge (8 steps) followed by hold steps only, i.e. the
    planner's safety-check shape (edge 8 + hold 12) starting at decision point d. segment_index (R, 32) (-1 for
    'recorded' -> False), segment_kind (R, 4)."""
    idx, kinds = np.asarray(segment_index), np.asarray(segment_kind)
    e = law.EDGE_STEPS
    out = np.zeros(idx.shape[0], bool)
    for r in range(idx.shape[0]):
        seg = idx[r, d:d + DECISION_WINDOW]
        if len(seg) < DECISION_WINDOW or (seg < 0).any():
            continue
        k = kinds[r][seg]
        out[r] = bool((k[:e] == "edge").all() and len(set(seg[:e].tolist())) == 1 and (k[e:] == "hold").all())
    return out


def decision_point_instances(worst, law_mask, grasped=None, shapes=None, families=None):
    """Every valid C1 / C2 instance of one anchor at decision points d in DECISION_POINTS (law rollouts only).

    worst: (R, 33) worst-guard tilt, rollout 0 = recorded. An instance (rollout r, point d) is valid when the worst tilt at
    state d is <= 25 deg and no state 1..d exceeded 30 deg; its window is states d+1..d+20. positive: window max > 30;
    action_caused: positive and the RECORDED rollout's max over states 1..d+20 stays <= 30; fresh_valid: worst tilt at
    state d < 5 deg; fresh: positive and fresh_valid; calm: window max < 5 deg. Optional: grasped (R, 33) bool -> the
    'grasped' flag = block held at state d; shapes {d: (R,) bool} (planner_shape) -> 'planner_shape'; families (R,)."""
    worst = np.asarray(worst, float)
    out = []
    for d in DECISION_POINTS:
        end = d + DECISION_WINDOW
        if end >= worst.shape[1]:
            raise ValueError("decision window beyond the rollout")
        rec_ok = worst[0, 1:end + 1].max() <= TOPPLE_DEG
        for r in range(1, worst.shape[0]):
            if not law_mask[r]:
                continue
            if worst[r, d] > DECISION_MAX_START_DEG or (d > 0 and worst[r, 1:d + 1].max() > TOPPLE_DEG):
                continue
            window = float(worst[r, d + 1:end + 1].max())
            positive, fresh_valid = window > TOPPLE_DEG, bool(worst[r, d] < FRESH_MAX_DEG)
            out.append({"d": d, "rollout": r, "family": None if families is None else str(families[r]),
                        "valid": True, "fresh_valid": fresh_valid, "positive": positive, "action_caused": bool(positive and rec_ok),
                        "fresh": bool(positive and fresh_valid), "calm": window < CALM_DEG, "window_max": window,
                        "tilt_at_d": float(worst[r, d]),
                        "grasped": None if grasped is None else bool(np.asarray(grasped)[r, d]),
                        "planner_shape": None if shapes is None else bool(shapes[d][r])})
    return out


def count_instances(instances, keep=lambda i: True):
    """{d: {DP_KEYS: count}} over the instances that pass ``keep``."""
    out = {d: {k: 0 for k in DP_KEYS} for d in DECISION_POINTS}
    for i in instances:
        if keep(i):
            for k in DP_KEYS:
                out[i["d"]][k] += int(bool(i[k]))
    return out


def decision_point_pools(worst, law_mask):
    """C1 / C2 instance pools of one anchor: {d: counts of DP_KEYS} (see decision_point_instances)."""
    return count_instances(decision_point_instances(worst, law_mask))


def anchor_record(z, normalization):
    rows = rollout_rows(z, normalization)
    rec = rows[0]
    src = np.asarray(z["source_guard_tilt_deg"], float).max(-1)
    if not np.array_equal(np.asarray(z["guard_tilt_deg"], float)[0].max(-1), src):
        raise ValueError(f"anchor {int(z['anchor_id'])}: recorded rollout differs from the source tilt")
    law_rows = [r for r in rows if r["law"]]
    fresh = bool(rec["anchor_tilt"] < FRESH_MAX_DEG)
    speed = float(np.linalg.norm(np.asarray(z["anchor_velocity_xyz"], float))) / law.CONTROL_DT
    for r in rows:                                  # anchor context of every rollout (per-family rates in family_table)
        r["recorded_max"], r["fresh"], r["anchor_speed_mps"] = rec["max"], fresh, speed
    out = {"anchor_id": int(z["anchor_id"]), "stratum": str(z["stratum"]), "set": str(z["set"]), "anchor_tilt": rec["anchor_tilt"],
           "fresh": fresh, "anchor_speed_mps": speed, "rows": rows, "per_h": {}}
    m20 = [r["max"][20] for r in rows]
    out["c3_pairs"] = sum(1 for i in range(len(m20)) for j in range(i + 1, len(m20)) if abs(m20[i] - m20[j]) >= PAIR_GAP_DEG)
    shapes = {d: planner_shape(z["segment_index"], z["segment_kind"], d) for d in DECISION_POINTS}
    inst = decision_point_instances(np.asarray(z["guard_tilt_deg"], float).max(-1), [r["law"] for r in rows],
                                    np.asarray(z["grasped"], bool), shapes, [r["family"] for r in rows])
    out["instances"] = inst
    out["dp"] = count_instances(inst)
    out["dp_sub"] = {name: count_instances(inst, keep) for name, keep in SUBGROUPS.items()}
    out["distinct"] = {"positive_rollouts": len({i["rollout"] for i in inst if i["positive"]}),
                       "action_caused_rollouts": len({i["rollout"] for i in inst if i["action_caused"]}),
                       "calm_rollouts": len({i["rollout"] for i in inst if i["calm"]})}
    for h in HORIZONS:
        topple = [r["max"][h] > TOPPLE_DEG for r in rows]
        low = [r["max"][h] < REACH_DEG for r in rows]
        safe = [r["max"][h] <= TOPPLE_DEG for r in rows]
        law_t = [r["max"][h] > TOPPLE_DEG for r in law_rows]
        law_low = [r["max"][h] < REACH_DEG for r in law_rows]
        law_safe = [r["max"][h] <= TOPPLE_DEG for r in law_rows]
        out["per_h"][h] = {
            "law_topples": sum(law_t), "law_no_topple": sum(law_safe), "law_reach10": sum(r["max"][h] >= REACH_DEG for r in law_rows),
            "law_rise10": sum(r["max"][h] - r["anchor_tilt"] >= REACH_DEG for r in law_rows),
            "law_calm": sum(r["max"][h] < CALM_DEG for r in law_rows),
            "action_caused": sum(law_t) if rec["max"][h] <= TOPPLE_DEG else 0,
            "recorded_topples": bool(rec["max"][h] > TOPPLE_DEG),
            "contrastive_strict": bool(any(topple) and any(low)), "contrastive_loose": bool(any(topple) and any(safe)),
            "contrastive_strict_law": bool(any(law_t) and any(law_low)), "contrastive_loose_law": bool(any(law_t) and any(law_safe))}
    return out


# ----- aggregation ------------------------------------------------------------------------------------------------
def _q(values, q):
    values = np.asarray(values, float)
    return None if not len(values) else float(np.quantile(values, q))


def bootstrap_mean(values, seed=0, n=BOOT, alpha=0.1):
    """Mean and (alpha/2, 1-alpha/2) anchor-bootstrap interval of per-anchor values."""
    v = np.asarray(values, float)
    if not len(v):
        return {"mean": None, "lo": None, "hi": None, "anchors": 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(means, alpha / 2)), "hi": float(np.quantile(means, 1 - alpha / 2)),
            "anchors": int(len(v))}


def family_table(anchors, group=lambda a: a["stratum"]):
    """group -> family -> rollouts and per-rollout rates: reaching >= 10 / > 30 deg within each horizon, rise >= 10,
    action-caused (topple while the recorded sibling stays <= 30 within h), fresh topple / rise (anchor < 5 deg), calm,
    near-goal frames per rollout. ``group`` maps an anchor record to its group key (default: its stratum); 'all' too."""
    out = {}
    keys = sorted({group(a) for a in anchors}, key=lambda k: (len(str(k)), str(k)))
    for key in ["all", *keys]:
        sel = [a for a in anchors if key == "all" or group(a) == key]
        fams = sorted({r["family"] for a in sel for r in a["rows"]}, key=lambda f: (f != "recorded", f))
        block = {}
        for fam in fams:
            rows = [r for a in sel for r in a["rows"] if r["family"] == fam]
            n = len(rows)
            b = {"rollouts": n, "anchors": len(sel), "anchor_ge10": sum(r["anchor_tilt"] >= REACH_DEG for r in rows)}
            for h in HORIZONS:
                b[f"reach10_h{h}"] = sum(r["max"][h] >= REACH_DEG for r in rows) / n
                b[f"topple_h{h}"] = sum(r["max"][h] > TOPPLE_DEG for r in rows) / n
                b[f"rise10_h{h}"] = sum(r["max"][h] - r["anchor_tilt"] >= REACH_DEG for r in rows) / n
                b[f"action_caused_h{h}"] = sum(r["max"][h] > TOPPLE_DEG and r["recorded_max"][h] <= TOPPLE_DEG for r in rows) / n
                b[f"fresh_topple_h{h}"] = sum(r["fresh"] and r["max"][h] > TOPPLE_DEG for r in rows) / n
                b[f"fresh_rise10_h{h}"] = sum(r["fresh"] and r["max"][h] - r["anchor_tilt"] >= REACH_DEG for r in rows) / n
            b["calm_h20"] = sum(r["max"][20] < CALM_DEG for r in rows) / n
            b["grasp_lost"] = sum(r["grasped_start"] and not r["grasped_end"] for r in rows)
            b["free_space"] = sum(r["free_space"] for r in rows) / n
            b["frames_per_rollout"] = {k: sum(r["frames"][k] for r in rows) / n for k in ("boundary_slow", "speed_boundary", "class0")}
            block[fam] = b
        out[str(key)] = block
    return out


def anchor_yields(anchors, seed=0):
    """Per-anchor yields (mean + bootstrap 90 % interval) by stratum and by (stratum, source set)."""
    def one(sel, tag):
        per = {}
        for h in HORIZONS:
            ph = [a["per_h"][h] for a in sel]
            per[f"h{h}"] = {
                "law_topples_per_anchor": bootstrap_mean([p["law_topples"] for p in ph], seed),
                "law_reach10_per_anchor": bootstrap_mean([p["law_reach10"] for p in ph], seed),
                "law_rise10_per_anchor": bootstrap_mean([p["law_rise10"] for p in ph], seed),
                "action_caused_per_anchor": bootstrap_mean([p["action_caused"] for p in ph], seed),
                "fresh_topples_per_anchor": bootstrap_mean([p["law_topples"] if a["fresh"] else 0 for a, p in zip(sel, ph)], seed),
                "fresh_rise10_per_anchor": bootstrap_mean([p["law_rise10"] if a["fresh"] else 0 for a, p in zip(sel, ph)], seed),
                "law_calm_per_anchor": bootstrap_mean([p["law_calm"] for p in ph], seed),
                "contrastive_strict_fraction": bootstrap_mean([p["contrastive_strict"] for p in ph], seed),
                "contrastive_loose_fraction": bootstrap_mean([p["contrastive_loose"] for p in ph], seed),
                "contrastive_strict_law_fraction": bootstrap_mean([p["contrastive_strict_law"] for p in ph], seed),
                "contrastive_loose_law_fraction": bootstrap_mean([p["contrastive_loose_law"] for p in ph], seed),
                "strict_contrastive_topples_per_anchor": bootstrap_mean([p["law_topples"] if p["contrastive_strict"] else 0 for p in ph], seed),
                "strict_contrastive_no_topple_per_anchor": bootstrap_mean([p["law_no_topple"] if p["contrastive_strict"] else 0 for p in ph], seed),
                "loose_contrastive_topples_per_anchor": bootstrap_mean([p["law_topples"] if p["contrastive_loose"] else 0 for p in ph], seed),
                "loose_contrastive_no_topple_per_anchor": bootstrap_mean([p["law_no_topple"] if p["contrastive_loose"] else 0 for p in ph], seed),
                "recorded_topple_fraction": bootstrap_mean([p["recorded_topples"] for p in ph], seed)}
        frames = {k: bootstrap_mean([sum(r["frames"][k] for r in a["rows"] if r["law"]) for a in sel], seed)
                  for k in ("boundary_slow", "speed_boundary", "class0")}
        frames_recorded = {k: int(sum(a["rows"][0]["frames"][k] for a in sel)) for k in ("boundary_slow", "speed_boundary", "class0")}
        keys = DP_KEYS
        dps = {f"d{d}": {k: bootstrap_mean([a["dp"][d][k] for a in sel], seed) for k in keys} for d in DECISION_POINTS}
        dps["d0_plus_d8"] = {k: bootstrap_mean([sum(a["dp"][d][k] for d in DECISION_POINTS) for a in sel], seed) for k in keys}
        dps["anchors_with_action_caused"] = int(sum(any(a["dp"][d]["action_caused"] for d in DECISION_POINTS) for a in sel))
        dps["totals"] = {k: int(sum(a["dp"][d][k] for a in sel for d in DECISION_POINTS)) for k in keys}
        dps["totals_by_d"] = {f"d{d}": {k: int(sum(a["dp"][d][k] for a in sel)) for k in keys} for d in DECISION_POINTS}
        # subgroups (grasped at d: the planner always carries the block; planner_shape: window = one edge + hold 12)
        dps["subgroups"] = {name: {"totals": {k: int(sum(a["dp_sub"][name][d][k] for a in sel for d in DECISION_POINTS)) for k in keys},
                                   "d0_plus_d8": {k: bootstrap_mean([sum(a["dp_sub"][name][d][k] for d in DECISION_POINTS) for a in sel], seed)
                                                  for k in ("valid", "positive", "action_caused", "calm")}}
                            for name in SUBGROUPS}
        # one rollout may be an instance at d = 0 AND d = 8: distinct rollouts behind the pooled instances
        dps["distinct_rollouts"] = {k: {"total": int(sum(a["distinct"][k] for a in sel)), "per_anchor": bootstrap_mean([a["distinct"][k] for a in sel], seed)}
                                    for k in ("positive_rollouts", "action_caused_rollouts", "calm_rollouts")}
        fams = sorted({i["family"] for a in sel for i in a["instances"]})
        dps["by_family_totals"] = {f: {k: int(sum(bool(i[k]) for a in sel for i in a["instances"] if i["family"] == f)) for k in keys}
                                   for f in fams}
        return {"anchors": len(sel), "fresh_anchors": sum(a["fresh"] for a in sel), "tag": tag, "per_horizon": per,
                "law_frames_per_anchor": frames, "recorded_frames_total": frames_recorded,
                "c3_pairs_per_anchor": bootstrap_mean([a["c3_pairs"] for a in sel], seed), "decision_points": dps}
    out = {"by_stratum": {}, "by_stratum_set": {}, "by_dataset_stratum": {}, "by_dataset_stratum_set": {}}
    for stratum in STRATA:
        sel = [a for a in anchors if a["stratum"] == stratum]
        if sel:
            out["by_stratum"][stratum] = one(sel, stratum)
        for s in sorted({a["set"] for a in sel}):
            out["by_stratum_set"][f"{stratum}/{s}"] = one([a for a in sel if a["set"] == s], f"{stratum}/{s}")
        for d in sorted({a.get("tag", "") for a in sel}):
            part = [a for a in sel if a.get("tag", "") == d]
            out["by_dataset_stratum"][f"{d}/{stratum}"] = one(part, f"{d}/{stratum}")
            for s in sorted({a["set"] for a in part}):
                out["by_dataset_stratum_set"][f"{d}/{stratum}/{s}"] = one([a for a in part if a["set"] == s], f"{d}/{stratum}/{s}")
    out["all"] = one(anchors, "all")
    out["action_caused_anchors"] = action_caused_anchors(anchors)
    return out


def action_caused_anchors(anchors):
    """Every anchor with at least one action-caused decision-point instance (nothing is folded into 'about 0')."""
    out = []
    for a in anchors:
        ac = [i for i in a["instances"] if i["action_caused"]]
        if ac:
            out.append({"dataset": a.get("tag", ""), "anchor_id": a["anchor_id"], "stratum": a["stratum"], "set": a["set"],
                        "anchor_tilt": a["anchor_tilt"], "instances": len(ac), "distinct_rollouts": len({i["rollout"] for i in ac}),
                        "by_d": {f"d{d}": sum(i["d"] == d for i in ac) for d in DECISION_POINTS},
                        "families": dict(Counter(i["family"] for i in ac)), "grasped_instances": sum(bool(i["grasped"]) for i in ac),
                        "planner_shape_instances": sum(bool(i["planner_shape"]) for i in ac)})
    return out


def anchor_table(anchors):
    """Compact per-anchor export for rrt_data_plan.py: rollouts (family, per-horizon max, frames) and instances."""
    table = []
    for a in anchors:
        table.append({"dataset": a.get("tag", ""), "anchor_id": a["anchor_id"], "stratum": a["stratum"], "set": a["set"],
                      "fresh": a["fresh"], "anchor_tilt": a["anchor_tilt"], "grasped": a["rows"][0]["grasped_start"],
                      "families": [r["family"] for r in a["rows"]],
                      "rollouts": [{"family": r["family"], "max": {str(h): r["max"][h] for h in HORIZONS}, "frames": r["frames"],
                                    "anchor_tilt": r["anchor_tilt"]} for r in a["rows"]],
                      "instances": [{k: i[k] for k in ("d", "rollout", "family", *DP_KEYS, "grasped", "planner_shape", "window_max", "tilt_at_d")}
                                    for i in a["instances"]]})
    return table


def mismatch_breakdown(rows):
    """Receding-horizon nominal mismatch (mm, worst axis per step) split by step (first segment = steps 1-8, where the
    arm may still carry the recorded controller's velocity, vs steps 9-32, driven only by the law) and by anchor pinch
    speed (<= / > SLOW_ANCHOR_MPS)."""
    def stats(values):
        v = np.asarray(values, float)
        return {"steps": int(v.size)} if not v.size else {"steps": int(v.size), "median": _q(v, 0.5), "p95": _q(v, 0.95), "p99": _q(v, 0.99),
                                                          "max": float(v.max())}
    out = {}
    for name, test in (("all_anchors", lambda r: True), ("slow_anchor", lambda r: r["anchor_speed_mps"] <= SLOW_ANCHOR_MPS),
                       ("fast_anchor", lambda r: r["anchor_speed_mps"] > SLOW_ANCHOR_MPS)):
        sel = [r for r in rows if test(r)]
        out[name] = {"rollouts": len(sel), "steps_1_8": stats(np.concatenate([r["mismatch_mm"][:law.EDGE_STEPS] for r in sel]) if sel else []),
                     "steps_9_32": stats(np.concatenate([r["mismatch_mm"][law.EDGE_STEPS:] for r in sel]) if sel else [])}
    out["slow_anchor_mps"] = SLOW_ANCHOR_MPS
    return out


SPEED_BAND_MM = (0.8, 1.2)            # edge-average speed within +-20 % of the reference's 1 mm per step


def speed_stats(rows):
    """Executed pinch speed (mm per 0.05 s step) of law rollouts: edge steps of the first segment (steps 1-8, the arm may
    still carry the recorded controller's motion), edge steps of later segments, hold steps of later segments; and the
    edge-average speed (straight-line displacement over an edge / 8) of later edges, with the share inside SPEED_BAND_MM."""
    def stats(values):
        v = np.asarray(values, float)
        if not v.size:
            return {"n": 0}
        return {"n": int(v.size), "mean": float(v.mean()), "median": _q(v, 0.5), "p5": _q(v, 0.05), "p95": _q(v, 0.95), "max": float(v.max())}
    first, later, hold, edge_first, edge_later = [], [], [], [], []
    for r in rows:
        k, seg, v = r["step_kind"], r["step_segment"], r["speed_mm"]
        first += list(v[(k == "edge") & (seg == 0)])
        later += list(v[(k == "edge") & (seg > 0)])
        hold += list(v[(k == "hold") & (seg > 0)])
        for s, value in r["edge_average_mm"]:
            (edge_first if s == 0 else edge_later).append(value)
    band = np.asarray(edge_later, float)
    return {"edge_steps_first_segment": stats(first), "edge_steps_later_segments": stats(later), "hold_steps_later_segments": stats(hold),
            "edge_average_first_segment": stats(edge_first), "edge_average_later_segments": stats(edge_later),
            "edge_average_later_in_band": None if not band.size else float(((band >= SPEED_BAND_MM[0]) & (band <= SPEED_BAND_MM[1])).mean()),
            "band_mm": list(SPEED_BAND_MM), "reference_mm_per_step": 1000 * law.STEP_M}


def law_health(anchors):
    """Tracking, anti-windup activity, WM range and nominal mismatch of the law rollouts (free space vs contact)."""
    law_rows = [r for a in anchors for r in a["rows"] if r["law"]]
    rec_rows = [a["rows"][0] for a in anchors]
    out = {"wm_range": {"law_steps": sum(r["wm_steps"] for r in law_rows), "law_steps_in_range": sum(r["wm_steps_in_range"] for r in law_rows),
                        "recorded_steps": sum(r["wm_steps"] for r in rec_rows), "recorded_steps_in_range": sum(r["wm_steps_in_range"] for r in rec_rows),
                        "range_low": law.WM_RANGE_LOW.tolist(), "range_high": law.WM_RANGE_HIGH.tolist()}}
    out["wm_range"]["law_fraction_in_range"] = out["wm_range"]["law_steps_in_range"] / max(1, out["wm_range"]["law_steps"])
    out["wm_range"]["recorded_fraction_in_range"] = out["wm_range"]["recorded_steps_in_range"] / max(1, out["wm_range"]["recorded_steps"])
    for name, sel in (("free_space", [r for r in law_rows if r["free_space"]]), ("contact", [r for r in law_rows if not r["free_space"]]),
                      ("all", law_rows)):
        if not sel:
            out[name] = {"rollouts": 0}
            continue
        steps = len(sel) * law.EDGE_STEPS * 4
        mm = np.concatenate([r["mismatch_mm"] for r in sel])
        wm = np.concatenate([r["mismatch_wm"] for r in sel])
        ol = np.concatenate([r["mismatch_open_loop_mm"] for r in sel])
        track_final = np.array([r["tracking_final_m"] for r in sel])
        track_max = np.array([r["tracking_max_m"] for r in sel])
        out[name] = {
            "rollouts": len(sel), "steps": steps,
            "flags_rollout_fraction": {k: sum(r["flag_steps"][k] > 0 for r in sel) / len(sel) for k in FLAGS},
            "flags_step_fraction": {k: sum(r["flag_steps"][k] for r in sel) / steps for k in FLAGS},
            "tracking_final_mm": {"median": 1000 * _q(track_final, 0.5), "p95": 1000 * _q(track_final, 0.95), "max": 1000 * float(track_final.max())},
            "tracking_max_mm": {"median": 1000 * _q(track_max, 0.5), "p95": 1000 * _q(track_max, 0.95), "max": 1000 * float(track_max.max()),
                                "over_5mm_rollouts": int((track_max > 0.005).sum()), "over_20mm_rollouts": int((track_max > 0.02).sum())},
            "nominal_mismatch_step_mm": {"median": _q(mm, 0.5), "p95": _q(mm, 0.95), "p99": _q(mm, 0.99), "max": float(mm.max())},
            "nominal_mismatch_step_wm": {"median": _q(wm, 0.5), "p95": _q(wm, 0.95), "max": float(wm.max())},
            "nominal_mismatch_breakdown_mm": mismatch_breakdown(sel),
            "nominal_mismatch_rollout_max_mm": {"median": _q([r["mismatch_mm"].max() for r in sel], 0.5),
                                                "p90": _q([r["mismatch_mm"].max() for r in sel], 0.9),
                                                "max": float(max(r["mismatch_mm"].max() for r in sel))},
            "open_loop_mismatch_step_mm": {"median": _q(ol, 0.5), "p95": _q(ol, 0.95), "max": float(ol.max())},
            "executed_speed_mm_per_step": speed_stats(sel)}
    by_family = {}
    for fam in sorted({r["family"] for r in law_rows}):
        sel = [r for r in law_rows if r["family"] == fam]
        free = [r for r in sel if r["free_space"]]
        by_family[fam] = {"rollouts": len(sel), "free_space": len(free),
                          "frozen_rollouts_free": sum(r["flag_steps"]["reference_frozen"] > 0 for r in free),
                          "frozen_rollouts_contact": sum(r["flag_steps"]["reference_frozen"] > 0 for r in sel if not r["free_space"]),
                          "wm_clipped_rollouts": sum(r["flag_steps"]["wm_clipped"] > 0 for r in sel),
                          "tracking_final_mm_median": 1000 * _q([r["tracking_final_m"] for r in sel], 0.5),
                          "free_mismatch_max_mm": None if not free else float(max(r["mismatch_mm"].max() for r in free))}
    out["by_family"] = by_family
    return out


def cost_block(reports, anchors):
    """Throughput / memory / disk per start from the collector summaries."""
    out = []
    for path, s in reports:
        n = s["sanity"]["anchors"]
        rollouts = sum(s["counts"]["rollouts_by_family"].values())
        mem = s.get("memory", {}).get("sampled") or {}
        out.append({"report": str(path), "anchors": n, "rollouts": rollouts, "workers": s["workers"], "wall_seconds": s["wall_seconds"],
                    "wall_seconds_per_anchor": s["wall_seconds"] / n, "worker_seconds_per_rollout": s["seconds_per_rollout"],
                    "worker_seconds_per_anchor_median": s["anchor_seconds_median"], "anchors_per_hour": 3600 * n / s["wall_seconds"],
                    "bytes": s["bytes"], "bytes_per_anchor": s["bytes_per_anchor"], "bytes_per_rollout": s["bytes_per_rollout"],
                    "peak_pss_sum_gib": None if mem.get("peak_pss_sum_kb") is None else mem["peak_pss_sum_kb"] / 1024**2,
                    "min_mem_available_gib": None if mem.get("min_mem_available_kb") is None else mem["min_mem_available_kb"] / 1024**2})
    return out


# ----- IO ---------------------------------------------------------------------------------------------------------
def load_dataset(dataset, report):
    """(manifest, normalization, [anchor records]) with every npz sha256-checked against the manifest."""
    manifest = read_json(Path(dataset) / "manifest.json")
    if manifest.get("format") != DATASET_FORMAT or manifest.get("status") != "complete" or (Path(dataset) / "FAILED.json").exists():
        raise ValueError(f"{dataset}: not a complete {DATASET_FORMAT} dataset")
    summary = read_json(Path(report) / "summary.json")
    if summary.get("manifest_sha256") != sha256(Path(dataset) / "manifest.json") or (Path(report) / "FAILED.json").exists():
        raise ValueError(f"{report}: summary does not belong to {dataset} (manifest sha256) or the run failed")
    normalization = manifest["wm_action"]["normalization"]
    anchors = []
    for row in manifest["anchors"]:
        path = Path(dataset) / row["file"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"{path}: sha256 differs from the manifest")
        with np.load(path, allow_pickle=False) as npz:
            z = {k: np.asarray(npz[k]) for k in NPZ_KEYS}
        rec = anchor_record(z, normalization)
        if rec["anchor_id"] != int(row["id"]) or rec["stratum"] != row["stratum"]:
            raise ValueError(f"{path}: anchor id / stratum differ from the manifest")
        rec["dataset"], rec["tag"] = str(dataset), Path(dataset).name
        anchors.append(rec)
    return manifest, normalization, anchors, summary


def write_report(output, summary):
    multi = len(summary["datasets"]) > 1
    t = summary["family_table_by_dataset"] if multi else summary["family_table"]
    y, hl, cost = summary["yields"], summary["law_health"], summary["cost"]
    pct = (lambda v: "–" if v is None else f"{100 * v:.0f}")
    lines = ["# RRT 动作律试采：产出统计（每层 × 族）\n",
             f"- 数据：{', '.join(summary['datasets'])}；锚点 {summary['anchors']} 个（{json.dumps(summary['anchors_by_stratum'], ensure_ascii=False)}），rollout {summary['rollouts']} 条。",
             "- 全部数字由 npz 数组重算（不抄 manifest 的结果字段）；每个 npz 的 sha256 与 manifest 核对；像素数组没有解压。",
             "- 定义见 `RRT_PILOT_YIELDS.md`：最差红柱倾角在第 1..h 步的最大值；≥10° / >30°；“由动作导致” = 同锚点 recorded 续推 h 步内不超过 30°；“新起点” = 锚点两根红柱都 < 5°。\n",
             f"## 1. {'数据集/层' if multi else '层'} × 族：每条 rollout 的比例（%）\n",
             "| 组 | 族 | rollout | 锚点≥10° | ≥10° h8/16/20/24/32 | >30° h8/16/20/24/32 | 增量≥10° h20/32 | 动作致倒 h20/32 | 新起点倒 h20/32 | 安静<5° h20 | 自由空间 | 丢夹持 |",
             "|" + "---|" * 12]
    for stratum, fams in t.items():
        for fam, b in fams.items():
            lines.append(f"| {stratum} | {fam} | {b['rollouts']} | {b['anchor_ge10']} | " + " / ".join(pct(b[f'reach10_h{h}']) for h in HORIZONS)
                         + " | " + " / ".join(pct(b[f'topple_h{h}']) for h in HORIZONS) + f" | {pct(b['rise10_h20'])} / {pct(b['rise10_h32'])} | "
                         f"{pct(b['action_caused_h20'])} / {pct(b['action_caused_h32'])} | {pct(b['fresh_topple_h20'])} / {pct(b['fresh_topple_h32'])} | "
                         f"{pct(b['calm_h20'])} | {pct(b['free_space'])} | {b['grasp_lost']} |")
    lines += ["", "## 2. 每个锚点的产出（均值，括号为锚点自助法 90% 区间）\n",
              "| 组 | 锚点 | 新起点 | 动作致倒 h20 | 动作致倒 h32 | 新起点倒 h20 | 新起点倒 h32 | 新起点增量≥10° h32 | 对比起点(严) h20 | 对比起点(严) h32 | 对比起点(宽) h32 | C3 配对 |",
              "|" + "---|" * 12]
    fmt = (lambda b: "–" if b["mean"] is None else f"{b['mean']:.3f} ({b['lo']:.3f}–{b['hi']:.3f})")
    groups = ([*y["by_dataset_stratum"].items(), *y["by_dataset_stratum_set"].items()] if multi
              else [*y["by_stratum"].items(), *y["by_stratum_set"].items()])
    for key, b in [*groups, ("all", y["all"])]:
        p20, p32 = b["per_horizon"]["h20"], b["per_horizon"]["h32"]
        lines.append(f"| {key} | {b['anchors']} | {b['fresh_anchors']} | {fmt(p20['action_caused_per_anchor'])} | {fmt(p32['action_caused_per_anchor'])} | "
                     f"{fmt(p20['fresh_topples_per_anchor'])} | {fmt(p32['fresh_topples_per_anchor'])} | {fmt(p32['fresh_rise10_per_anchor'])} | "
                     f"{fmt(p20['contrastive_strict_fraction'])} | {fmt(p32['contrastive_strict_fraction'])} | {fmt(p32['contrastive_loose_fraction'])} | "
                     f"{fmt(b['c3_pairs_per_anchor'])} |")
    lines += ["", "## 2b. 决策点上的 C1 / C2 实例池（每锚点均值；只数动作律 rollout；d = 0 锚点，d = 8 第一次重读画面）\n",
              "- 实例有效：第 d 步最差倾角 ≤ 25° 且第 1..d 步没超过 30°；窗口 = 第 d+1..d+20 步。正例 = 窗口内 > 30°；由动作导致 = 正例且 recorded 第 1..d+20 步不超过 30°；"
              "新起点 = 正例且第 d 步 < 5°；安静 = 窗口内一直 < 5°。\n",
              "| 组 | 锚点 | d0 有效 / 正例 / 动作致倒 / 新起点 / 安静 | d8 有效 / 正例 / 动作致倒 / 新起点 / 安静 | 有动作致倒的锚点 |", "|---|---|---|---|---|"]
    for key, b in [*groups, ("all", y["all"])]:
        dp = b["decision_points"]
        cell = (lambda d: " / ".join(f"{dp[d][k]['mean']:.3f}" for k in ("valid", "positive", "action_caused", "fresh", "calm")))
        lines.append(f"| {key} | {b['anchors']} | {cell('d0')} | {cell('d8')} | {dp['anchors_with_action_caused']} |")
    lines += ["", "## 2c. 实例 vs 不同的 rollout；子组（合计，d = 0 与 d = 8 合并）\n",
              "- 同一条 rollout 可以在 d = 0 与 d = 8 各算一个实例（两个不同的预测问题，但不是独立事件）；“不同 rollout” = 至少有一个这种实例的 rollout 数。",
              "- 夹持 = 决策点处绿块在手里（规划器始终夹着绿块）；规划器形状 = 窗口里的动作恰好是 1 条边 + 停（安全检查的形状 8 + 12）。\n",
              "| 组 | 正例 实例 / 不同 rollout | 由动作导致 实例 / 不同 rollout | 夹持：有效 / 正例 / 由动作导致 | 规划器形状：有效 / 正例 / 由动作导致 / 安静 | 夹持且规划器形状：正例 / 由动作导致 |",
              "|---|---|---|---|---|---|"]
    for key, b in [*groups, ("all", y["all"])]:
        dp = b["decision_points"]
        t, dr, sg = dp["totals"], dp["distinct_rollouts"], dp["subgroups"]
        g, s, gs = sg["grasped"]["totals"], sg["planner_shape"]["totals"], sg["grasped_planner_shape"]["totals"]
        lines.append(f"| {key} | {t['positive']} / {dr['positive_rollouts']['total']} | {t['action_caused']} / {dr['action_caused_rollouts']['total']} | "
                     f"{g['valid']} / {g['positive']} / {g['action_caused']} | {s['valid']} / {s['positive']} / {s['action_caused']} / {s['calm']} | "
                     f"{gs['positive']} / {gs['action_caused']} |")
    lines += ["", "## 2d. 所有出现“由动作导致”实例的锚点（逐个列出）\n",
              "| 数据集 | 锚点 | 层 | 来源 | 锚点倾角 | 实例（d0 / d8） | 不同 rollout | 族 | 夹持实例 | 规划器形状实例 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in y["action_caused_anchors"]:
        lines.append(f"| {r['dataset']} | {r['anchor_id']} | {r['stratum']} | {r['set']} | {r['anchor_tilt']:.1f}° | {r['instances']}（{r['by_d']['d0']} / {r['by_d']['d8']}） | "
                     f"{r['distinct_rollouts']} | {json.dumps(r['families'], ensure_ascii=False)} | {r['grasped_instances']} | {r['planner_shape_instances']} |")
    lines += ["", "## 2e. 各族的新起点实例（d = 0 与 d = 8 合计；按层）\n", "| 层 | 族 | 有效 | 新起点有效 | 正例 | 新起点正例 |", "|---|---|---|---|---|---|"]
    for stratum, b in y["by_stratum"].items():
        for fam, c in b["decision_points"]["by_family_totals"].items():
            lines.append(f"| {stratum} | {fam} | {c['valid']} | {c['fresh_valid']} | {c['positive']} | {c['fresh']} |")
    lines += ["", "## 3. 目标附近的帧（动作律 rollout 第 1..32 步，每锚点）\n",
              "| 组 | 慢速 0.8–2.2 cm | 1–2.5 cm/s 且 ≤1.5 cm | 类 0 帧 | recorded 合计（不算新数据）|", "|---|---|---|---|---|"]
    for key, b in (y["by_dataset_stratum"] if multi else y["by_stratum"]).items():
        f = b["law_frames_per_anchor"]
        lines.append(f"| {key} | {fmt(f['boundary_slow'])} | {fmt(f['speed_boundary'])} | {fmt(f['class0'])} | {json.dumps(b['recorded_frames_total'])} |")
    w = hl["wm_range"]
    lines += ["", "## 4. 动作律本身\n",
              f"- WM 归一化动作在 B 训练范围内：动作律 {w['law_steps_in_range']}/{w['law_steps']} 步（{100 * w['law_fraction_in_range']:.3f}%）；recorded {w['recorded_steps_in_range']}/{w['recorded_steps']} 步。"]
    for name in ("free_space", "contact"):
        b = hl[name]
        if not b.get("rollouts"):
            lines.append(f"- {name}：0 条")
            continue
        fr = b["flags_rollout_fraction"]
        lines.append(f"- {'自由空间' if name == 'free_space' else '有接触'} {b['rollouts']} 条：触发比例（按 rollout）冻结 {100 * fr['reference_frozen']:.1f}%、拖拽 {100 * fr['reference_dragged']:.1f}%、"
                     f"积分截断 {100 * fr['integral_clamped']:.1f}%、WM 裁剪 {100 * fr['wm_clipped']:.1f}%、框裁剪 {100 * fr['box_clipped']:.1f}%、±5 cm 裁剪 {100 * fr['range_clipped']:.1f}%；"
                     f"跟踪误差最大值 中位 {b['tracking_max_mm']['median']:.2f} mm、p95 {b['tracking_max_mm']['p95']:.2f}、最大 {b['tracking_max_mm']['max']:.1f} mm；"
                     f"名义−执行失配（逐步、最差轴）中位 {b['nominal_mismatch_step_mm']['median']:.3f} mm、p95 {b['nominal_mismatch_step_mm']['p95']:.2f}、p99 {b['nominal_mismatch_step_mm']['p99']:.2f}、最大 {b['nominal_mismatch_step_mm']['max']:.2f} mm"
                     f"（WM 单位最大 {b['nominal_mismatch_step_wm']['max']:.2f}）；开环名义失配中位 {b['open_loop_mismatch_step_mm']['median']:.3f} mm、最大 {b['open_loop_mismatch_step_mm']['max']:.2f} mm。")
        bd = b["nominal_mismatch_breakdown_mm"]
        for part in ("slow_anchor", "fast_anchor"):
            p = bd[part]
            if not p["rollouts"]:
                continue
            s1, s2 = p["steps_1_8"], p["steps_9_32"]
            lines.append(f"  - 锚点夹爪速度 {'≤' if part == 'slow_anchor' else '>'} {100 * bd['slow_anchor_mps']:.1f} cm/s 的 {p['rollouts']} 条：第 1–8 步失配 中位 {s1['median']:.3f} / p95 {s1['p95']:.2f} / 最大 {s1['max']:.2f} mm；"
                         f"第 9–32 步 中位 {s2['median']:.3f} / p95 {s2['p95']:.2f} / p99 {s2['p99']:.2f} / 最大 {s2['max']:.2f} mm。")
        sp = b["executed_speed_mm_per_step"]
        cellsp = (lambda s: "–" if not s["n"] else f"均值 {s['mean']:.2f} / 中位 {s['median']:.2f} / p5 {s['p5']:.2f} / p95 {s['p95']:.2f} / 最大 {s['max']:.2f}（{s['n']}）")
        lines.append(f"  - 执行的夹爪速度（mm/步；参考点每条边的每步严格前进 {sp['reference_mm_per_step']:.1f} mm = 2 cm/s）：第一段的边步 {cellsp(sp['edge_steps_first_segment'])}；"
                     f"之后的边步 {cellsp(sp['edge_steps_later_segments'])}；之后的停步 {cellsp(sp['hold_steps_later_segments'])}；"
                     f"之后每条边的平均速度（8 步直线位移 / 8）{cellsp(sp['edge_average_later_segments'])}，落在 {sp['band_mm'][0]}–{sp['band_mm'][1]} mm/步 的比例 "
                     f"{'–' if sp['edge_average_later_in_band'] is None else format(100 * sp['edge_average_later_in_band'], '.1f') + '%'}。")
    lines += ["", "## 5. 吞吐、内存、磁盘\n"]
    for c in cost:
        lines.append(f"- `{c['report']}`：{c['anchors']} 个锚点、{c['workers']} 个 worker、墙钟 {c['wall_seconds']:.0f} s（每锚点 {c['wall_seconds_per_anchor']:.2f} s 墙钟，每小时 {c['anchors_per_hour']:.0f} 个锚点）；"
                     f"每条 rollout {c['worker_seconds_per_rollout']:.2f} worker·s；每锚点 {c['bytes_per_anchor'] / 1e6:.2f} MB（每条 rollout {c['bytes_per_rollout'] / 1e3:.0f} kB）；"
                     f"作业 PSS 峰值 {'–' if c['peak_pss_sum_gib'] is None else format(c['peak_pss_sum_gib'], '.2f')} GiB。")
    lines += ["", "## 6. 说明\n",
              "- 只读：没有加载 world model，没有打开评估数据；试采锚点全部来自训练侧。",
              "- 每组只有几十个锚点，区间很宽；“由动作导致”的实例集中在少数锚点上（第 2d 节逐个列出），按锚点重采样的区间反映这一点。",
              "- summary.json 的 `anchor_table` 带每个锚点的全部 rollout 与决策点实例，第 2 阶段方案（rrt_data_plan.py）从它重算所有每锚点产出。"]
    (Path(output) / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, action="append", required=True, help="rrt_branches_v1 dataset (repeatable)")
    p.add_argument("--report", type=Path, action="append", required=True, help="the collector's report dir of each dataset (same order)")
    p.add_argument("--output", type=Path, required=True, help="NEW run directory")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    a = p.parse_args(argv)
    if len(a.dataset) != len(a.report) or len({str(d.resolve()) for d in a.dataset}) != len(a.dataset):
        raise ValueError("give one --report per distinct --dataset")
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    for d, r in zip(a.dataset, a.report):
        for f in (d / "manifest.json", r / "summary.json"):
            if not f.is_file():
                raise FileNotFoundError(f)
    from wm_training.provenance import implementation_hashes
    root = Path(__file__).resolve().parent
    output.mkdir(parents=True)
    run = {"format": FORMAT, "args": {k: ([str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "input_sha256": {**{str(d / "manifest.json"): sha256(d / "manifest.json") for d in a.dataset},
                            **{str(r / "summary.json"): sha256(r / "summary.json") for r in a.report}},
           "input_note": "every anchor npz is sha256-checked against its manifest row before it is read",
           "implementation_sha256": {**implementation_hashes(IMPLEMENTATION_FILES, package_dir=root), **implementation_hashes(("danger_strata.py", "provenance.py"))},
           "python": platform.python_version(), "numpy": np.__version__,
           "definitions": {"horizons": list(HORIZONS), "reach_deg": REACH_DEG, "topple_deg": TOPPLE_DEG, "calm_deg": CALM_DEG,
                           "fresh_max_deg": FRESH_MAX_DEG, "pair_gap_deg": PAIR_GAP_DEG, "slow_mps": SLOW_MPS, "speed_band_mps": list(SPEED_BAND_MPS),
                           "boundary_shell_m": list(BOUNDARY_SHELL_M), "within_m": WITHIN_M, "bootstrap": {"resamples": BOOT, "interval": 0.9, "seed": a.seed},
                           "decision_points": list(DECISION_POINTS), "decision_window": DECISION_WINDOW,
                           "decision_max_start_deg": DECISION_MAX_START_DEG, "slow_anchor_mps": SLOW_ANCHOR_MPS,
                           "subgroups": {"grasped": "block held at state d", "planner_shape": "window actions = one edge + hold only (8 + 12)"},
                           "speed_band_mm": list(SPEED_BAND_MM)},
           "world_model_loaded": False, "evaluation_data_opened": False}
    write_new_json(output / "run.json", run)
    try:
        started = time.perf_counter()
        anchors, reports, datasets = [], [], []
        norm0 = None
        for d, r in zip(a.dataset, a.report):
            manifest, norm, recs, summary = load_dataset(d, r)
            if norm0 is not None and norm != norm0:
                raise ValueError("datasets use different WM normalizations")
            norm0 = norm
            anchors += recs
            reports.append((r, summary))
            datasets.append(str(d))
        ids = Counter((x["dataset"], x["anchor_id"]) for x in anchors)
        if max(ids.values()) > 1:
            raise ValueError("repeated anchor ids within a dataset")
        summary = {"format": FORMAT, "datasets": datasets, "anchors": len(anchors),
                   "anchors_by_stratum": dict(Counter(x["stratum"] for x in anchors)),
                   "anchors_by_stratum_set": dict(Counter(f"{x['stratum']}/{x['set']}" for x in anchors)),
                   "rollouts": sum(len(x["rows"]) for x in anchors), "families": dict(Counter(r["family"] for x in anchors for r in x["rows"])),
                   "family_table": family_table(anchors),
                   "family_table_by_dataset": family_table(anchors, group=lambda x: f"{x['tag']}/{x['stratum']}"),
                   "family_table_by_dataset_set": family_table(anchors, group=lambda x: f"{x['tag']}/{x['stratum']}/{x['set']}"),
                   "yields": anchor_yields(anchors, a.seed), "law_health": law_health(anchors),
                   "cost": cost_block(reports, anchors), "anchor_table": anchor_table(anchors),
                   "elapsed_seconds": time.perf_counter() - started}
        write_report(output, summary)
        write_new_json(output / "summary.json", summary)
        print(json.dumps(native({k: summary[k] for k in ("anchors", "anchors_by_stratum", "rollouts")}), ensure_ascii=False), flush=True)
        return summary
    except BaseException as error:
        write_new_json(output / "FAILED.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
