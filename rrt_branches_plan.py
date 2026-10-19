"""Stage 1 of the RRT-action-law branch collection: choose TRAINING-SIDE start states and pre-draw the rollouts.

numpy only at import time (h5py is imported lazily, only to copy the anchor frames' cache pixels), so the pure
functions run in both WSL venvs and the unit tests; the CLI runs in the TRAINING venv (~/lewm-train/.venv, the one
with h5py). Stage 2 is collect_rrt_branches.py (simulation venv). Doc: RRT_BRANCHES.md.

Start states ("anchors" a; a >= 12, a + 32 <= last frame, worst guard tilt at a <= 25 deg) come ONLY from
  * wm_full_001 TRAIN episodes (index episodes.train; raw data/exploration_5000_001), and
  * the TRAINING rows of wm_train_bench_001 (its bench_split part named by runs/lewm_ft_B_001/run.json
    bench_rows_used_for_training, i.e. 'bench_test' of data/danger_train_split_001; raw danger_train_001/002).
The val / test episodes of wm_full_001, the danger benchmark, random_branches_* and bench_dev rows are never
opened. Every scanned episode's trajectory.npz / metadata.json sha256 must equal the cache index.

Strata (priority T > F > G > M; "near" = horizontal distance to a guard COM <= 6 cm):
  T tilted : worst guard 5-25 deg and (pinch or grasped green block) near THAT guard;
  F fresh  : both guards < 5 deg and (pinch near a guard with pinch z < 13 cm, or grasped green near a guard with COM z < 15 cm);
  G goal   : grasped, hover_error < 5 cm, gripper command closed (the goal image id is linked when one exists);
  M motion : grasped, not near a guard (height-gated rule of F), gripper command closed.
Per stratum a seeded round-robin over episodes picks one frame per episode per pass (frames of one episode >= 33 apart).

For every anchor the plan fixes its rollouts (families below) and pre-draws their edge directions from a per-anchor
seed (rollout r uses np.random.default_rng([anchor_seed, r])); the rollouts follow rrt_action_law:
  recorded  : the recorded actions a..a+31 (sanity: must reproduce the record);
  hold      : hold 32 steps (AnchoredHold with anti-windup at the anchor pinch);
  planner   : 3 random edges (direction ~ N(0, diag(1, 1, 0.5)^2), normalised) + hold 8;
  guard     : first edge horizontal toward the nearer guard +-20 deg, then 2 random edges, + hold 8;
  edge_hold : 1 random edge + hold 24;
  approach  : (G) n in 1..3 edges along one direction chosen so that the green block's predicted resting point (the
              calibrated nominal model) is 0.8-2.2 cm from the goal point, then hold to step 32;
  push / push4 / push_hold : 3 / 4 / 1 edges, EACH horizontal toward the nearer guard +-20 deg (independent jitter),
              then hold to step 32 (added after pilot_001, where one 8 mm edge toward a guard never tipped one);
  push2_hold: 2 such edges + hold 16 (added 2026-09-28 in the pilot review): from its second decision point (step 8)
              the window is exactly the planner's safety-check shape (one edge + hold 12) after one edge of contact.
Appending a family to a set never changes the earlier rollouts (each rollout has its own rng).
Options (defaults = the pilot_001 behaviour): --f-near-radius-m (F's nearness radius; T and M keep 6 cm),
--recorded-max-tilt-h20 (keep only anchors whose recorded continuation stays below this within 20 steps),
--goal-image-strata (e.g. G: anchors of these strata only from episodes with a linked goal image, so that their
labels use the goal image's ACHIEVED green COM, user decision 2), --capacity-only (report how many anchors each
stratum can supply; no plan files).
The plan also copies the cache pixels of frames a-10..a for the stage-2 pixel check.

Stage-2 roles (added 2026-09-28, stage 2 step 1; section "stage-2 roles" below). Without --stage2-plan (and without
--role, or with --role train) the unchanged code above runs: anchors.json / cache_pixels.npz / summary.json as before, and
run.json with the old key set (the stage-2 options are left out of 'args' while unset; --eval-bench-* are refused there):
  --role train  --stage2-plan P : the frozen training command of P (runs/rrt_data_plan_005/plan.json, sha256 checked);
                                  every selection option must equal P's command; anchors.json also records role + P.
  --role dev    --stage2-plan P : branch-dev SELECTION set (P branch_dev_set): wm_full_001 val DEV half + the bench_dev
                                  rows of data/wm_train_bench_001 (raw danger_train_001/002); pooled round-robin.
  --role confirm --stage2-plan P: confirmation set random_branches_003 (P confirmation_set): wm_full_001 val REPORT half +
                                  danger-benchmark bench_test (data/wm_bench_001, 509 S1-S3 falls + 138 N1-N4 negatives);
                                  per-source quotas and the plan's multiplicities.
For dev / confirm the seed, quotas, families, F radius and goal-image rule come from P (command-line overrides are
refused); G anchors link the half's own VAL goal image (goal_images_002 set 'val', variant 0, tag dev / report); every
start from the wrong side is refused (split ids and raw-episode sha256 against the other pools); the plan directory must
carry P's name (or the next free number).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import re
import shlex
import time

import numpy as np

import random_branches_plan as rb
import rrt_action_law as law
from wm_training.danger_strata import native, read_json, sha256, write_new_json

PLAN_FORMAT = "rrt_branches_plan_v1"
PREFIX = 10
HORIZON = 32
MIN_ANCHOR = 12
MIN_GAP_FRAMES = HORIZON + 1
SKIP_TILT_DEG = 25.0
STRATA = ("T", "F", "G", "M")                      # priority order
STRATUM_NAMES = {"T": "tilted_guard", "F": "fresh_guard", "G": "near_goal", "M": "free_carry_motion"}
QUOTA = {"T": 1500, "F": 1500, "G": 500, "M": 750}   # the draft stage-2 plan (WORKLOG_FABLE.md, 2026-09-27)
T_TILT_DEG = (5.0, 25.0)
FRESH_MAX_TILT_DEG = 5.0
NEAR_RADIUS_M = rb.NEAR_RADIUS_M                   # 0.06
NEAR_GRIP_MAX_Z_M = rb.NEAR_GRIP_MAX_Z_M           # 0.13
GREEN_NEAR_MAX_Z_M = rb.GREEN_NEAR_MAX_Z_M         # 0.15
GOAL_NEAR_M = rb.GOAL_NEAR_M                       # 0.05
APPROACH_SHELL_M = (0.008, 0.022)
APPROACH_MAX_EDGES = 3
APPROACH_TRIES = 400
PLANNER_EDGES = 3
# push families (added 2026-09-28 after pilot_001: at 1 mm per step one 8 mm edge toward a guard almost never tips an
# upright 4 x 4 x 10 cm guard, whose tipping angle is atan(2 / 5) = 21.8 deg): every edge horizontal toward the nearer
# guard (rb.guard_direction, an independent U(-20, 20) deg jitter per edge), then hold to step 32.
PUSH_EDGES = {"push": 3, "push4": 4, "push_hold": 1, "push2_hold": 2}
FAMILIES = ("recorded", "hold", "planner", "guard", "edge_hold", "approach", *PUSH_EDGES)
DEFAULT_FAMILY_SETS = {
    "T": ("recorded", "hold", "planner", "planner", "planner", "guard", "guard", "edge_hold"),
    "F": ("recorded", "hold", "planner", "planner", "planner", "guard", "guard", "edge_hold"),
    "G": ("recorded", "hold", "planner", "planner", "approach", "approach", "approach", "edge_hold"),
    "M": ("recorded", "hold", "planner", "planner", "planner", "guard", "edge_hold", "edge_hold"),
}
SOURCES = ("full_train", "bench_train")
SOURCE_CACHE = {"full_train": "wm_full_001", "bench_train": "wm_train_bench_001",
                # stage-2 roles (never produced by the default training mode)
                "val_dev": "wm_full_001", "bench_dev": "wm_train_bench_001", "val_report": "wm_full_001", "bench_test": "wm_bench_001"}
GLOBAL_ID_STRIDE = 100000
NPZ_FIELDS = rb.NPZ_FIELDS + ("actions", "hover_goal_xyz")
# stage-2 cost model (RRT_BRANCHES.md section 6): the real-scene smoke run measured 431 kB and 1.15 worker-seconds per
# rollout with ONE worker (2 anchors x 8 rollouts); seconds carry a 1.3x margin for 3 workers sharing CPU / GPU;
# memory: random_branches_002 measured 6.46 GiB PSS for main + 4 workers of the same worker code (about 1.5 GiB each),
# the smoke run 1.55 GiB PSS for one process holding scene + renderer; 1.6 / 0.6 GiB are rounded up.
BYTES_PER_ROLLOUT = 450_000
SECONDS_PER_ROLLOUT = 1.5
SECONDS_PER_ANCHOR_OVERHEAD = 0.5
WORKER_PSS_GIB = 1.6
MAIN_PSS_GIB = 0.6
IMPLEMENTATION_FILES = ("rrt_branches_plan.py", "rrt_action_law.py", "random_branches_plan.py")


# ----- strata ---------------------------------------------------------------------------------------------------
def classify_rrt_frames(pinch_xyz, object_com_xyz, grasped, hover_error_m, guard_tilt_deg, f_near_radius_m=NEAR_RADIUS_M):
    """Stratum letter of every frame ('' = none); priority T > F > G > M (see the module docstring).

    ``f_near_radius_m`` (default NEAR_RADIUS_M = 6 cm, the pilot_001 rule) is the horizontal radius of F's height-gated
    "near a guard" test only; T keeps 6 cm and M is always "grasped and NOT near (6 cm, gated)", so a smaller F radius
    never turns a frame 4.5-6 cm from a guard into M."""
    if not (isinstance(f_near_radius_m, (int, float)) and math.isfinite(f_near_radius_m) and 0 < f_near_radius_m <= NEAR_RADIUS_M):
        raise ValueError("f_near_radius_m must be in (0, NEAR_RADIUS_M]")
    pinch, com = np.asarray(pinch_xyz, float), np.asarray(object_com_xyz, float)
    grasped, error, tilt = np.asarray(grasped, bool), np.asarray(hover_error_m, float), np.asarray(guard_tilt_deg, float)
    n = len(pinch)
    if pinch.shape != (n, 3) or com.shape != (n, 3, 3) or grasped.shape != (n,) or error.shape != (n,) or tilt.shape != (n, 2):
        raise ValueError("classify_rrt_frames: inconsistent shapes")
    if not (np.isfinite(pinch).all() and np.isfinite(com).all() and np.isfinite(error).all() and np.isfinite(tilt).all()):
        raise ValueError("classify_rrt_frames: non-finite input")
    d_pinch = rb.horizontal_guard_distance(pinch, com)             # (n, 2)
    d_green = rb.horizontal_guard_distance(com[:, 0], com)
    rows = np.arange(n)
    worst_index = np.argmax(tilt, axis=1)
    worst = tilt.max(1)
    near_worst = (d_pinch[rows, worst_index] <= NEAR_RADIUS_M) | (grasped & (d_green[rows, worst_index] <= NEAR_RADIUS_M))

    def gated(radius):
        return (((d_pinch.min(1) <= radius) & (pinch[:, 2] < NEAR_GRIP_MAX_Z_M))
                | (grasped & (d_green.min(1) <= radius) & (com[:, 0, 2] < GREEN_NEAR_MAX_Z_M)))
    near_gated = gated(NEAR_RADIUS_M)
    near_f = near_gated if f_near_radius_m == NEAR_RADIUS_M else gated(f_near_radius_m)
    out = np.full(n, "", dtype="U1")
    out[grasped & ~near_gated] = "M"
    out[grasped & (error < GOAL_NEAR_M)] = "G"
    out[(worst < FRESH_MAX_TILT_DEG) & near_f] = "F"
    out[(worst >= T_TILT_DEG[0]) & (worst <= T_TILT_DEG[1]) & near_worst] = "T"
    return out


def admissible_frames(frames, guard_tilt_deg, actions, labels, recorded_max_tilt_h20=None):
    """(T,) bool: a >= MIN_ANCHOR, a + HORIZON <= T - 1, worst tilt <= 25 deg, and for G / M the gripper command in
    force at a (recorded action a-1) is closing (> 0), so the held block stays held under the law's held command.

    ``recorded_max_tilt_h20`` (default None = no filter): also require that the RECORDED continuation's worst tilt over
    frames a+1..a+20 stays <= this value (e.g. 30: then any law rollout that exceeds 30 deg within 20 steps is an
    'action-caused' C1 positive). It selects on simulator truth of the recorded data, never on a model output."""
    tilt, actions, labels = np.asarray(guard_tilt_deg, float), np.asarray(actions, float), np.asarray(labels)
    if tilt.shape != (frames, 2) or actions.shape != (frames - 1, 5) or labels.shape != (frames,):
        raise ValueError("admissible_frames: inconsistent shapes")
    idx = np.arange(frames)
    worst = tilt.max(1)
    ok = (idx >= MIN_ANCHOR) & (idx + HORIZON <= frames - 1) & (worst <= SKIP_TILT_DEG)
    grip_prev = np.full(frames, np.nan)
    grip_prev[1:] = actions[:, 4]
    closed = grip_prev > 0
    ok &= ~np.isin(labels, ["G", "M"]) | closed
    if recorded_max_tilt_h20 is not None:
        if not math.isfinite(recorded_max_tilt_h20):
            raise ValueError("recorded_max_tilt_h20 must be finite")
        ahead = np.full(frames, np.inf)
        for a in np.flatnonzero(ok):                                  # ok implies a + 20 < frames
            ahead[a] = worst[a + 1:a + 21].max()
        ok &= ahead <= recorded_max_tilt_h20
    return ok


def select_anchors(episodes, quotas, seed, f_near_radius_m=NEAR_RADIUS_M, recorded_max_tilt_h20=None, stratum_episodes=None,
                   sources=SOURCES):
    """episodes: {(source, episode_id): arrays (NPZ_FIELDS)} -> (anchors, availability).

    Strata are filled in STRATA order; inside a stratum a seeded round-robin over the episodes that still have an
    admissible frame of that stratum (one anchor per episode per pass), frame uniform among the admissible frames
    that keep MIN_GAP_FRAMES to every earlier anchor of the episode (rb.pick_spaced). ``stratum_episodes`` ({stratum:
    set of episode keys}, default None) restricts a stratum to those episodes (--goal-image-strata: the episodes with a
    linked goal image); a restricted episode's frames of that stratum count as not admissible. With the default
    options the picks are identical to the pilot_001 code. ``sources`` only names the keys of the by_source
    availability counts (stage-2 roles use their own source names); it never changes a pick."""
    restrict = dict(stratum_episodes or {})
    if any(s not in STRATA for s in restrict):
        raise ValueError("stratum_episodes: unknown stratum")
    rng = np.random.default_rng(seed)
    strata, taken, available = {}, {}, {}
    keys = sorted(episodes)
    for key in keys:
        a = episodes[key]
        labels = classify_rrt_frames(a["pinch_xyz"], a["object_com_xyz"], a["grasped"], a["hover_error_m"], a["guard_tilt_deg"],
                                     f_near_radius_m)
        ok = admissible_frames(len(labels), a["guard_tilt_deg"], a["actions"], labels, recorded_max_tilt_h20)
        strata[key] = {s: (np.flatnonzero(ok & (labels == s)) if s not in restrict or key in restrict[s] else np.zeros(0, np.int64))
                       for s in STRATA}
        taken[key] = []
    anchors = []
    for stratum in STRATA:
        pool = [k for k in keys if len(strata[k][stratum])]
        available[stratum] = {"episodes": len(pool), "frames": int(sum(len(strata[k][stratum]) for k in pool)),
                              "by_source": {s: int(sum(len(strata[k][stratum]) for k in pool if k[0] == s)) for s in sources}}
        order = [pool[i] for i in rng.permutation(len(pool))]
        chosen = 0
        while chosen < quotas.get(stratum, 0) and order:
            still = []
            for key in order:
                if chosen >= quotas[stratum]:
                    break
                frame = rb.pick_spaced(rng, strata[key][stratum], taken[key], MIN_GAP_FRAMES)
                if frame is None:
                    continue
                taken[key].append(frame)
                anchors.append({"source": key[0], "episode_id": int(key[1]), "stratum": stratum, "frame": int(frame)})
                chosen += 1
                still.append(key)
            order = still
    return anchors, available


CAPACITY_QUOTA = 10**9


def anchor_capacity(episodes, seed, f_near_radius_m=NEAR_RADIUS_M, recorded_max_tilt_h20=None, stratum_episodes=None):
    """How many anchors select_anchors can place at most (read-only; --capacity-only).

    'joint': every stratum asks for CAPACITY_QUOTA in priority order T > F > G > M (the spacing rule is shared by the
    strata of one episode, so later strata get what the earlier ones leave); 'alone': each stratum on its own. The
    frame picks are random, so the numbers are what this seed's greedy round-robin reaches, not a proven maximum.
    ``stratum_episodes`` as in select_anchors (options.goal_image_strata lists the restricted strata)."""
    joint, available = select_anchors(episodes, {s: CAPACITY_QUOTA for s in STRATA}, seed, f_near_radius_m, recorded_max_tilt_h20,
                                      stratum_episodes)
    out = {"joint": {s: {"anchors": sum(x["stratum"] == s for x in joint),
                         "by_source": {src: sum(x["stratum"] == s and x["source"] == src for x in joint) for src in SOURCES}} for s in STRATA},
           "alone": {}, "available": available, "quota_used": CAPACITY_QUOTA,
           "options": {"f_near_radius_m": f_near_radius_m, "recorded_max_tilt_h20": recorded_max_tilt_h20,
                       "goal_image_strata": sorted(stratum_episodes or {})}}
    for s in STRATA:
        alone, _ = select_anchors(episodes, {t: (CAPACITY_QUOTA if t == s else 0) for t in STRATA}, seed, f_near_radius_m, recorded_max_tilt_h20,
                                  stratum_episodes)
        out["alone"][s] = {"anchors": len(alone), "by_source": {src: sum(x["source"] == src for x in alone) for src in SOURCES}}
    return out


def parse_goal_image_strata(text):
    """'G' / 'G,M' / None -> sorted tuple of strata whose anchors need a linked goal image (with an achieved COM)."""
    strata = tuple(sorted({t.strip() for t in str(text or "").split(",") if t.strip()}))
    if any(s not in STRATA for s in strata):
        raise ValueError(f"bad --goal-image-strata {text!r}")
    return strata


def goal_image_episodes(episode_keys, goal_links, strata):
    """{stratum: set of episode keys (source, id) with a linked goal image that carries an achieved COM} for ``strata``."""
    ok = {k for k in episode_keys if (goal_links.get((SOURCE_CACHE[k[0]], int(k[1]))) or {}).get("goal_xyz") is not None}
    return {s: set(ok) for s in strata}


# ----- rollout specs --------------------------------------------------------------------------------------------
def _edges_then_hold(directions):
    steps = law.EDGE_STEPS * len(directions)
    if steps > HORIZON:
        raise ValueError("too many edges for the horizon")
    segments = [["edge", law.unit(d).tolist()] for d in directions]
    if steps < HORIZON:
        segments.append(["hold", HORIZON - steps])
    return segments


def segments_from_json(segments):
    """JSON segments [["edge", [x, y, z]] | ["hold", n], ...] -> rrt_action_law segments (validated, HORIZON steps)."""
    out = law.validate_segments([(k, np.asarray(a, float) if k == "edge" else int(a)) for k, a in segments])
    if sum(law.EDGE_STEPS if k == "edge" else a for k, a in out) != HORIZON:
        raise ValueError("a rollout must have exactly HORIZON steps")
    return out


def predicted_displacement(pinch_xyz, direction, n_edges, config, model="grasped"):
    """Pinch (= held block) displacement after n edges + hold to HORIZON under the calibrated nominal model."""
    plant = law.ModelPlant(law.resolve_model(model), pinch_xyz)
    law.run_sequence(plant, segments_from_json(_edges_then_hold([direction] * n_edges)), 1.0, config)
    return plant.position - np.asarray(pinch_xyz, float)


def approach_spec(rng, pinch_xyz, green_xyz, goal_xyz, config, shell=APPROACH_SHELL_M):
    """Direction and edge count whose predicted green resting point is inside the shell around the goal point.

    Rejection sampling: n ~ U{1..3}, a point on the shell (radius ~ U(shell), direction ~ N(0, diag(1, 1, 0.5)^2))
    fixes the direction; accepted when the model-predicted resting point lies inside the shell. If nothing fits in
    APPROACH_TRIES tries (goal too far or too close), the fallback goes straight toward the goal for 3 edges
    (feasible = False)."""
    green, goal = np.asarray(green_xyz, float), np.asarray(goal_xyz, float)
    for attempt in range(1, APPROACH_TRIES + 1):
        n = int(rng.integers(1, APPROACH_MAX_EDGES + 1))
        target = goal + float(rng.uniform(*shell)) * law.random_direction(rng)
        if np.linalg.norm(target - green) < 1e-6:
            continue
        d = law.unit(target - green)
        rough = green + (law.EDGE_STEPS * law.STEP_M * n + 0.001) * d
        if not (shell[0] - 0.003 <= np.linalg.norm(rough - goal) <= shell[1] + 0.003):
            continue
        end = green + predicted_displacement(pinch_xyz, d, n, config)
        dist = float(np.linalg.norm(end - goal))
        if shell[0] <= dist <= shell[1]:
            return {"direction": d.tolist(), "edges": n, "predicted_end_xyz": end.tolist(), "predicted_goal_distance_m": dist,
                    "feasible": True, "tries": attempt}
    d = law.unit(goal - green) if np.linalg.norm(goal - green) > 1e-6 else np.array([1.0, 0.0, 0.0])
    end = green + predicted_displacement(pinch_xyz, d, APPROACH_MAX_EDGES, config)
    return {"direction": d.tolist(), "edges": APPROACH_MAX_EDGES, "predicted_end_xyz": end.tolist(),
            "predicted_goal_distance_m": float(np.linalg.norm(end - goal)), "feasible": False, "tries": APPROACH_TRIES}


def draw_rollouts(anchor_seed, families, pinch_xyz, object_com_xyz, goal_xyz, config):
    """Rollout specs of one anchor; rollout r uses np.random.default_rng([anchor_seed, r])."""
    specs = []
    for r, family in enumerate(families):
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}")
        spec = {"rollout": r, "family": family, "segments": None, "guard_index": None, "guard_angle_deg": None, "approach": None}
        rng = np.random.default_rng([int(anchor_seed), r])
        if family == "hold":
            spec["segments"] = [["hold", HORIZON]]
        elif family == "planner":
            spec["segments"] = _edges_then_hold([law.random_direction(rng) for _ in range(PLANNER_EDGES)])
        elif family == "guard":
            first, guard, delta = rb.guard_direction(rng, pinch_xyz, object_com_xyz)
            spec.update(guard_index=guard, guard_angle_deg=delta)
            spec["segments"] = _edges_then_hold([first] + [law.random_direction(rng) for _ in range(PLANNER_EDGES - 1)])
        elif family == "edge_hold":
            spec["segments"] = _edges_then_hold([law.random_direction(rng)])
        elif family in PUSH_EDGES:
            pushes = [rb.guard_direction(rng, pinch_xyz, object_com_xyz) for _ in range(PUSH_EDGES[family])]
            spec.update(guard_index=pushes[0][1], guard_angle_deg=pushes[0][2], push_angles_deg=[p[2] for p in pushes])
            spec["segments"] = _edges_then_hold([p[0] for p in pushes])
        elif family == "approach":
            if goal_xyz is None:
                raise ValueError("approach needs a goal point")
            a = approach_spec(rng, pinch_xyz, np.asarray(object_com_xyz, float)[0], goal_xyz, config)
            spec["approach"] = a
            spec["segments"] = _edges_then_hold([a["direction"]] * a["edges"])
        if spec["segments"] is not None:
            segments_from_json(spec["segments"])
        specs.append(spec)
    return specs


def parse_family_sets(texts):
    """['G=recorded,hold,approach', ...] -> {stratum: tuple} over DEFAULT_FAMILY_SETS."""
    sets = dict(DEFAULT_FAMILY_SETS)
    for text in texts or []:
        stratum, _, rest = str(text).partition("=")
        families = tuple(f.strip() for f in rest.split(",") if f.strip())
        if stratum not in STRATA or not families or any(f not in FAMILIES for f in families):
            raise ValueError(f"bad --families entry {text!r}")
        if families[0] != "recorded":
            raise ValueError("rollout 0 must be 'recorded' (the reproduction check)")
        if "approach" in families and stratum != "G":
            raise ValueError("'approach' needs a goal-near (G) anchor")
        sets[stratum] = families
    return sets


FAMILY_SET_B_SALT = 7919


def parse_family_sets_b(texts, fraction_text):
    """Second family set per stratum: (['T=recorded,...'], 'T=0.34') -> ({stratum: tuple}, {stratum: fraction}).

    Each anchor of such a stratum uses set B when np.random.default_rng([anchor_seed, FAMILY_SET_B_SALT]).random() <
    fraction (seeded, independent of the selection order), set A (the --families / default set) otherwise."""
    sets_b = {}
    for text in texts or []:
        stratum, _, rest = str(text).partition("=")
        families = tuple(f.strip() for f in rest.split(",") if f.strip())
        if stratum not in STRATA or stratum in sets_b or not families or any(f not in FAMILIES for f in families):
            raise ValueError(f"bad --families-b entry {text!r}")
        if families[0] != "recorded":
            raise ValueError("rollout 0 must be 'recorded' (the reproduction check)")
        if "approach" in families and stratum != "G":
            raise ValueError("'approach' needs a goal-near (G) anchor")
        sets_b[stratum] = families
    fractions = {}
    for item in [t for t in str(fraction_text or "").split(",") if t.strip()]:
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in sets_b or key in fractions:
            raise ValueError(f"bad --families-b-fraction entry {item!r} (needs a --families-b set for {key!r})")
        fractions[key] = float(value)
        if not 0.0 <= fractions[key] <= 1.0:
            raise ValueError("--families-b-fraction values must be in [0, 1]")
    if set(fractions) != set(sets_b):
        raise ValueError("every --families-b stratum needs a --families-b-fraction")
    return sets_b, fractions


def family_set_for(anchor_seed, stratum, sets_a, sets_b, fractions):
    """('A' | 'B', families) of one anchor (see parse_family_sets_b)."""
    if stratum in sets_b and np.random.default_rng([int(anchor_seed), FAMILY_SET_B_SALT]).random() < fractions[stratum]:
        return "B", sets_b[stratum]
    return "A", sets_a[stratum]


def parse_quotas(text, scale):
    quotas = dict(QUOTA)
    if text:
        for item in str(text).split(","):
            key, _, value = item.partition("=")
            if key.strip() not in STRATA:
                raise ValueError(f"bad quota {item!r}")
            quotas[key.strip()] = int(value)
    if not 0 < scale <= 1 or any(v < 0 for v in quotas.values()):
        raise ValueError("0 < scale <= 1 and non-negative quotas")
    return {k: int(math.ceil(v * scale)) for k, v in quotas.items()}


# ----- goal images ------------------------------------------------------------------------------------------------
GOAL_ID_KEYS = ("goal_image_id", "goal_id", "image_id", "id")
GOAL_EPISODE_KEYS = ("episode_id", "source_episode_id", "cache_episode_id")
GOAL_XYZ_KEYS = ("achieved_green_xyz", "achieved_green_com_xyz", "goal_xyz", "green_com_xyz", "achieved_goal_xyz")
GOAL_OK_STATUS = (None, "complete", "ok", "success", "accepted")      # goal_images_v2 marks usable rows 'accepted'
GOAL_IMAGES_FORMATS = ("goal_images_v1", "goal_images_v2")


def _one_of(entry, keys, what):
    found = {k: entry[k] for k in keys if k in entry and entry[k] is not None}
    values = {json.dumps(v) for v in found.values()}
    if len(values) > 1:
        raise ValueError(f"goal image entry has conflicting {what} fields: {found}")
    return next(iter(found.values())) if found else None


def load_goal_images(directory):
    """{('wm_full_001' | 'wm_train_bench_001', episode_id): {'goal_image_id', 'goal_xyz'}} from <dir>/manifest.json.

    Accepted manifest shape (RRT_BRANCHES.md section 5): a list under 'goals' / 'images' / 'entries'; per entry an id
    ({GOAL_ID_KEYS}), the episode it belongs to ({GOAL_EPISODE_KEYS}, optionally with 'cache' and 'split', possibly
    inside a 'source' dict) and the achieved green COM ({GOAL_XYZ_KEYS}); entries with a 'status' other than
    complete / ok / success / accepted are skipped; only split 'train' (or no split) is used; entries without a
    source episode (id < 0, or an explicit cache of None: the goal_images_v2 'planner' goals) are skipped. A manifest
    that declares a format must be a known goal-image format with status 'complete'. The real dataset is
    data/goal_images_002 (goal_images_v2: list 'images', keys id / source_episode_id / split / cache /
    achieved_green_xyz / status 'accepted'). A missing directory or manifest links nothing (goal_image_id = -1)."""
    if directory is None:
        return {}, {"path": None, "status": "not given"}
    manifest = Path(directory) / "manifest.json"
    if not manifest.is_file():
        return {}, {"path": str(directory), "status": "no manifest.json: no goal images linked"}
    doc = read_json(manifest)
    if "format" in doc and (doc["format"] not in GOAL_IMAGES_FORMATS or doc.get("status") != "complete"):
        raise ValueError(f"{manifest}: format {doc.get('format')!r} / status {doc.get('status')!r} is not a complete goal-image dataset")
    entries = next((doc[k] for k in ("goals", "images", "entries") if isinstance(doc.get(k), list)), None)
    if entries is None:
        raise ValueError(f"{manifest}: no list under goals / images / entries")
    out, skipped = {}, Counter()
    for e in entries:
        if e.get("status") not in GOAL_OK_STATUS:
            skipped["status"] += 1
            continue
        src = e["source"] if isinstance(e.get("source"), dict) else e
        gid, eid = _one_of(e, GOAL_ID_KEYS, "id"), _one_of(src, GOAL_EPISODE_KEYS, "episode")
        xyz = _one_of(e, GOAL_XYZ_KEYS, "goal point")
        if gid is None or eid is None:
            skipped["no id / episode"] += 1
            continue
        if int(eid) < 0 or ("cache" in src and src["cache"] is None):
            skipped["no source episode"] += 1
            continue
        split = src.get("split")
        if split not in (None, "train"):
            skipped[f"split {split}"] += 1
            continue
        cache = Path(str(src.get("cache", "wm_full_001"))).name
        key = (cache, int(eid))
        if key in out:
            raise ValueError(f"{manifest}: two goal images for {key}")
        goal = None if xyz is None else [float(v) for v in xyz]
        if goal is not None and (len(goal) != 3 or not np.isfinite(goal).all()):
            raise ValueError(f"{manifest}: bad goal point for {key}")
        out[key] = {"goal_image_id": int(gid), "goal_xyz": goal}
    return out, {"path": str(directory), "status": "linked", "manifest_sha256": sha256(manifest), "entries": len(out), "skipped": dict(skipped)}


# ----- sources ----------------------------------------------------------------------------------------------------
def load_episode_arrays(path):
    arrays = rb.load_npz_fields(path, NPZ_FIELDS)
    n = len(arrays["time"])
    if arrays["actions"].shape != (n - 1, 5) or arrays["hover_goal_xyz"].shape != (n, 3):
        raise ValueError(f"{path}: actions / hover_goal_xyz have unexpected shapes")
    return arrays


def full_train_sources(index_path, classifier_run=None):
    """(rows, info) of the wm_full_001 TRAIN episodes; refuses any overlap with val / test / classifier dev-report ids."""
    index = read_json(index_path)
    if index.get("status") != "complete" or index.get("eval_cache"):
        raise ValueError(f"{index_path}: not a complete training cache index")
    train = {int(r["id"]): r for r in index["episodes"]["train"]}
    held = {int(r["id"]) for part in ("val", "test") for r in index["episodes"][part]}
    if classifier_run is not None:
        c = read_json(classifier_run)
        held |= {int(i) for i in c["dev_episode_ids"]} | {int(i) for i in c["report_episode_ids"]}
    if set(train) & held:
        raise ValueError("wm_full_001 train ids overlap val / test / classifier halves")
    raw_root = Path(index["source_root"])
    rows = [{"source": "full_train", "episode_id": eid, "cache_episode_id": eid, "episode_dir": str(raw_root / f"episode_{eid:04d}"),
             "npz_sha256": r["source_npz_sha256"], "metadata_sha256": r["source_metadata_sha256"],
             "cache_h5": str(Path(index_path).parent / r["path"]), "frames": int(r["num_states"])} for eid, r in sorted(train.items())]
    info = {"index": str(index_path), "raw_root": str(raw_root), "source_manifest_sha256": index["source_manifest_sha256"],
            "train_episodes": len(rows), "held_out_ids_refused": len(held)}
    return rows, index["normalization"], info


def bench_train_sources(index_path, split_path, ft_run_path):
    """(rows, info) of the wm_train_bench_001 rows that B was TRAINED on (run.json bench_rows_used_for_training)."""
    index, split, ft = read_json(index_path), read_json(split_path), read_json(ft_run_path)
    part = ft["finetune"]["data"]["bench_rows_used_for_training"]
    if part != "bench_test":
        raise ValueError(f"{ft_run_path}: bench_rows_used_for_training is {part!r}, expected 'bench_test'")
    if index.get("status") != "complete" or index.get("split_sha256") != sha256(split_path):
        raise ValueError("the bench cache was not built from this split file")
    if Path(ft["bench_cache"]).name != Path(index_path).parent.name or ft["bench_cache_index_sha256"] != sha256(index_path):
        raise ValueError(f"{ft_run_path}: trained on a different bench cache index")
    datasets = {}
    for row in split["datasets"]:
        folder = Path(row["path"])
        if not folder.name.startswith("danger_train"):
            raise ValueError(f"{folder}: only danger_train datasets are training material")
        datasets[int(row["index"])] = folder
    train_ids = {int(i) for i in index["bench_split"][part]}
    other = {int(i) for k, v in index["bench_split"].items() if k != part for i in v}
    if train_ids & other:
        raise ValueError("bench_split parts overlap")
    entries = {(int(e["dataset"]), int(e["episode_id"])): e for e in split["episodes"]}
    rows = []
    for r in index["episodes"]["val"]:
        gid = int(r["id"])
        if gid not in train_ids:
            continue
        if gid != int(r["source_dataset"]) * GLOBAL_ID_STRIDE + int(r["source_id"]):
            raise ValueError(f"bench cache id {gid} does not match its source")
        entry = entries[(int(r["source_dataset"]), int(r["source_id"]))]
        rows.append({"source": "bench_train", "episode_id": gid, "cache_episode_id": gid,
                     "episode_dir": str(datasets[int(r["source_dataset"])] / entry["path"]), "npz_sha256": r["source_npz_sha256"],
                     "metadata_sha256": r["source_metadata_sha256"], "cache_h5": str(Path(index_path).parent / r["path"]),
                     "frames": int(r["num_states"])})
    if len(rows) != len(train_ids):
        raise ValueError("some training bench ids are missing from the cache index")
    info = {"index": str(index_path), "split": str(split_path), "part": part, "train_episodes": len(rows), "refused_rows": len(other),
            "source_manifest_sha256": index["source_manifest_sha256"]}
    return sorted(rows, key=lambda r: r["episode_id"]), index["normalization"], info


def anchor_state(arrays, frame):
    return {"pinch_xyz": arrays["pinch_xyz"][frame].tolist(), "green_com_xyz": arrays["object_com_xyz"][frame][0].tolist(),
            "guard_tilt_deg": arrays["guard_tilt_deg"][frame].tolist(), "grasped": bool(arrays["grasped"][frame]),
            "hover_error_m": float(arrays["hover_error_m"][frame]), "class_label": int(arrays["class_label"][frame]),
            "unsafe": bool(arrays["unsafe"][frame]), "hover_goal_xyz": arrays["hover_goal_xyz"][frame].tolist(),
            "pinch_velocity_xyz": (arrays["pinch_xyz"][frame] - arrays["pinch_xyz"][frame - 1]).tolist()}


def make_anchor_record(index, anchor, anchor_seed, source, arrays, families, goal_link, config, f_near_radius_m=NEAR_RADIUS_M,
                       goal_image_strata=()):
    frame = int(anchor["frame"])
    one = slice(frame, frame + 1)
    rule = classify_rrt_frames(arrays["pinch_xyz"][one], arrays["object_com_xyz"][one], arrays["grasped"][one],
                               arrays["hover_error_m"][one], arrays["guard_tilt_deg"][one], f_near_radius_m)[0]
    if rule != anchor["stratum"]:
        raise RuntimeError("anchor stratum disagrees with the rule")
    if anchor["stratum"] in goal_image_strata and (goal_link is None or goal_link.get("goal_xyz") is None):
        raise RuntimeError(f"a {anchor['stratum']} anchor needs a linked goal image with an achieved COM (--goal-image-strata)")
    hover_goal = arrays["hover_goal_xyz"][frame]
    goal = {"hover_goal_xyz": hover_goal.tolist(), "goal_image_id": -1, "goal_xyz": hover_goal.tolist(), "goal_source": "hover_goal"}
    if goal_link is not None:
        goal.update(goal_image_id=goal_link["goal_image_id"])
        # stage-2 roles link VAL goal images and name them (set / name / tag / variant); a training link has no such keys
        goal.update({k: v for k, v in goal_link.items() if k.startswith("goal_image_") and k != "goal_image_id"})
        if goal_link["goal_xyz"] is not None:
            goal.update(goal_xyz=goal_link["goal_xyz"], goal_source="goal_image_achieved_com")
    grip = float(arrays["actions"][frame - 1][4])
    if source["source"] != anchor["source"] or int(source["episode_id"]) != int(anchor["episode_id"]):
        raise ValueError("anchor and source record disagree")
    return {"id": int(index), "set": anchor["source"], "episode_id": int(anchor["episode_id"]), "stratum": anchor["stratum"], "frame": frame,
            "group": f"{anchor['source']}/{anchor['stratum']}", "prefix_start": frame - PREFIX,
            "seed": int(anchor_seed), "source": source, "state": anchor_state(arrays, frame), "grip_command": grip, "goal": goal,
            "families": list(families),
            "sequences": draw_rollouts(anchor_seed, families, arrays["pinch_xyz"][frame], arrays["object_com_xyz"][frame],
                                       goal["goal_xyz"] if anchor["stratum"] == "G" else None, config)}


def cost_estimate(n_anchors, rollouts, workers):
    """Disk and wall-time estimate of stage 2 (cost model constants above)."""
    rollouts = int(rollouts)
    seconds = n_anchors * SECONDS_PER_ANCHOR_OVERHEAD + rollouts * SECONDS_PER_ROLLOUT
    return {"anchors": int(n_anchors), "rollouts": rollouts, "frames": rollouts * (HORIZON + 1), "bytes": rollouts * BYTES_PER_ROLLOUT,
            "gib": rollouts * BYTES_PER_ROLLOUT / 1024**3, "worker_seconds": seconds,
            "wall_hours": {str(w): seconds / w / 3600 for w in range(1, 4)}, "workers": int(workers),
            "wall_hours_at_workers": seconds / max(1, workers) / 3600,
            "memory_gib_at_workers": MAIN_PSS_GIB + WORKER_PSS_GIB * max(1, workers),
            "cost_model": {"bytes_per_rollout": BYTES_PER_ROLLOUT, "seconds_per_rollout": SECONDS_PER_ROLLOUT,
                           "seconds_per_anchor_overhead": SECONDS_PER_ANCHOR_OVERHEAD, "worker_pss_gib": WORKER_PSS_GIB, "main_pss_gib": MAIN_PSS_GIB}}


# ----- CLI (training venv) ------------------------------------------------------------------------------------------
class CachePixelBlock:
    """Cache pixels of frames a-PREFIX..a of one anchor, read only when cache_pixels.npz is written (2026-09-28, stage 2).

    np.savez_compressed(handle, **pixels) turns every value into an array with np.asanyarray, which calls __array__, so
    only one block is in memory at a time. The file is byte-identical to the one written from the arrays themselves
    (same numpy writer, same values, same key order; proven on the pilot plans), but the stage-2 training plan no longer
    holds 5340 x 11 frames (8.2 GiB) at once, which broke the 7 GB job limit. The block is re-read through
    random_branches_plan.read_cache_pixels, i.e. the cache episode's completeness, source sha256 and block shape are
    checked again at write time."""
    __slots__ = ("h5_path", "frame", "npz_sha256")

    def __init__(self, h5_path, frame, npz_sha256):
        self.h5_path, self.frame, self.npz_sha256 = h5_path, int(frame), npz_sha256

    def __array__(self, dtype=None, copy=None):
        block, _ = rb.read_cache_pixels(self.h5_path, self.frame, self.npz_sha256)
        return block if dtype is None else block.astype(dtype, copy=False)


def build_plan(a):
    if read_json(a.ft_run).get("cache_index_sha256") != sha256(a.full_cache / "index.json"):
        raise ValueError(f"{a.ft_run}: B was trained on a different wm_full_001 index")
    full_rows, full_norm, full_info = full_train_sources(a.full_cache / "index.json", a.classifier_run)
    bench_rows, bench_norm, bench_info = bench_train_sources(a.bench_cache / "index.json", a.bench_split, a.ft_run)
    for key in ("mean", "scale"):
        if not np.array_equal(full_norm[key], bench_norm[key]):
            raise ValueError("the two caches use different action normalizations")
    config_norm = {k: full_norm[k] for k in ("names", "mean", "scale")}
    config = law.LawConfig(normalization=config_norm)
    goal_links, goal_info = load_goal_images(a.goal_images)
    held_ids = {("wm_full_001", int(r["id"])) for part in ("val", "test") for r in read_json(a.full_cache / "index.json")["episodes"][part]}
    wrong = sorted(set(goal_links) & held_ids)
    if wrong:
        raise ValueError(f"goal images marked 'train' belong to held-out wm_full_001 episodes: {wrong[:5]}")
    train_keys = {(SOURCE_CACHE[r["source"]], int(r["episode_id"])) for r in full_rows + bench_rows}
    if goal_links:
        goal_info = {**goal_info, "linked_training_episodes": len(set(goal_links) & train_keys),
                     "links_outside_training_sources": len(set(goal_links) - train_keys)}
    ss = np.random.SeedSequence(a.seed)
    seed_scan, seed_select, seed_seq = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in ss.spawn(3)]
    started = time.perf_counter()
    rows = []
    rng = np.random.default_rng(seed_scan)
    for group in (full_rows, bench_rows):
        order = [group[i] for i in rng.permutation(len(group))]
        rows += order[:a.max_source_episodes] if a.max_source_episodes else order
    episodes, sources = {}, {}
    for r in rows:
        folder = Path(r["episode_dir"])
        if sha256(folder / "trajectory.npz") != r["npz_sha256"] or sha256(folder / "metadata.json") != r["metadata_sha256"]:
            raise ValueError(f"{folder}: raw files differ from the ones the cache was built from")
        arrays = load_episode_arrays(folder / "trajectory.npz")
        if len(arrays["time"]) != r["frames"]:
            raise ValueError(f"{folder}: frame count differs from the cache index")
        key = (r["source"], r["episode_id"])
        episodes[key], sources[key] = arrays, r
    scan_seconds = time.perf_counter() - started
    gi_strata = parse_goal_image_strata(getattr(a, "goal_image_strata", None))
    if gi_strata and not goal_links:
        raise ValueError("--goal-image-strata needs --goal-images with linked training episodes")
    restrict = goal_image_episodes(episodes, goal_links, gi_strata) if gi_strata else None
    if getattr(a, "capacity_only", False):
        capacity = anchor_capacity(episodes, seed_select, a.f_near_radius_m, a.recorded_max_tilt_h20, restrict)
        return {"capacity": capacity, "goal_images": goal_info, "sources": {"full_train": full_info, "bench_train": bench_info},
                "scanned": {s: sum(1 for k in episodes if k[0] == s) for s in SOURCES},
                "seeds": {"base": a.seed, "scan": seed_scan, "select": seed_select, "sequences": seed_seq},
                "scan_seconds": scan_seconds, "elapsed_seconds": time.perf_counter() - started}
    quotas = parse_quotas(a.quota, a.scale)
    families = parse_family_sets(a.families)
    families_b, fractions_b = parse_family_sets_b(getattr(a, "families_b", None), getattr(a, "families_b_fraction", None))
    anchors, available = select_anchors(episodes, quotas, seed_select, a.f_near_radius_m, a.recorded_max_tilt_h20, restrict)
    seq_seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in np.random.SeedSequence(seed_seq).spawn(len(anchors))]
    records, pixels = [], {}
    for i, (anchor, anchor_seed) in enumerate(zip(anchors, seq_seeds)):
        key = (anchor["source"], anchor["episode_id"])
        src, arrays = sources[key], episodes[key]
        block, n_states = rb.read_cache_pixels(src["cache_h5"], anchor["frame"], src["npz_sha256"])
        if n_states != src["frames"]:
            raise ValueError(f"{src['cache_h5']}: frame count differs from the raw episode")
        del block                                            # re-read when cache_pixels.npz is written (CachePixelBlock)
        pixels[f"a{i:04d}"] = CachePixelBlock(src["cache_h5"], anchor["frame"], src["npz_sha256"])
        link = goal_links.get((SOURCE_CACHE[anchor["source"]], anchor["episode_id"]))
        source = {k: src[k] for k in ("source", "episode_id", "cache_episode_id", "episode_dir", "npz_sha256", "metadata_sha256", "cache_h5", "frames")}
        set_name, fams = family_set_for(anchor_seed, anchor["stratum"], families, families_b, fractions_b)
        record = make_anchor_record(i, anchor, anchor_seed, source, arrays, fams, link, config, a.f_near_radius_m, gi_strata)
        record["family_set"] = set_name
        records.append(record)
    rollouts = sum(len(r["sequences"]) for r in records)
    approach = [s["approach"] for r in records for s in r["sequences"] if s["approach"] is not None]
    return {"records": records, "pixels": pixels, "normalization": config_norm, "law": config.to_json(), "quotas": quotas,
            "families": {k: list(v) for k, v in families.items()},
            "families_b": {k: {"families": list(v), "fraction": fractions_b[k]} for k, v in families_b.items()},
            "available": available, "goal_images": goal_info,
            "sources": {"full_train": full_info, "bench_train": bench_info}, "scanned": {s: sum(1 for k in episodes if k[0] == s) for s in SOURCES},
            "seeds": {"base": a.seed, "scan": seed_scan, "select": seed_select, "sequences": seed_seq},
            "counts": {"by_stratum": dict(Counter(r["stratum"] for r in records)), "by_group": dict(Counter(r["group"] for r in records)),
                       "by_stratum_family_set": dict(Counter(f"{r['stratum']}/{r['family_set']}" for r in records)),
                       "rollouts": rollouts, "families": dict(Counter(s["family"] for r in records for s in r["sequences"])),
                       "goal_images_linked_G": sum(1 for r in records if r["stratum"] == "G" and r["goal"]["goal_image_id"] >= 0),
                       "approach": {"rollouts": len(approach), "feasible": sum(x["feasible"] for x in approach)}},
            "estimate": cost_estimate(len(records), rollouts, a.workers_for_estimate), "scan_seconds": scan_seconds,
            "elapsed_seconds": time.perf_counter() - started}


def write_plan_report(output, summary):
    c = summary["counts"]
    e = summary["estimate"]
    lines = ["# RRT 动作律分支采集：起点与动作计划（第 1 阶段）\n",
             f"- 起点只来自训练侧：wm_full_001 train（扫描 {summary['scanned']['full_train']} 条）与 wm_train_bench_001 的训练行（{summary['sources']['bench_train']['part']}，扫描 {summary['scanned']['bench_train']} 条）；"
             "每条扫描到的原始轨迹的 trajectory.npz / metadata.json sha256 都与缓存 index 核对过；没有打开 val / test / 基准 / random_branches。",
             f"- 各层锚点：{json.dumps(c['by_stratum'], ensure_ascii=False)}（目标 {json.dumps(summary['quotas'], ensure_ascii=False)}）；按来源/层 {json.dumps(c['by_group'], ensure_ascii=False)}。",
             f"- 可用帧：{json.dumps(summary['available'], ensure_ascii=False)}。",
             f"- rollout 共 {c['rollouts']} 条：{json.dumps(c['families'], ensure_ascii=False)}；每层的族：{json.dumps(summary['families'], ensure_ascii=False)}。",
             f"- approach：{c['approach']['rollouts']} 条，其中 {c['approach']['feasible']} 条按名义模型预测停在目标 0.8–2.2 cm 壳层内（其余是直奔目标 3 条边的回退）。",
             f"- 目标画面：{json.dumps(summary['goal_images'], ensure_ascii=False)}；G 层链接到目标画面的锚点 {c['goal_images_linked_G']} 个。",
             f"- 第 2 阶段估计（成本模型见 RRT_BRANCHES.md）：{e['gib']:.1f} GiB，{e['worker_seconds'] / 3600:.1f} worker·小时 → 1/2/3 个 worker 约 "
             f"{e['wall_hours']['1']:.1f} / {e['wall_hours']['2']:.1f} / {e['wall_hours']['3']:.1f} 小时；{e['workers']} 个 worker 的内存约 {e['memory_gib_at_workers']:.1f} GiB。",
             f"- 缓存像素：每个锚点复制 a−{PREFIX}..a 共 {PREFIX + 1} 帧（cache_pixels.npz）供第 2 阶段逐像素比对。",
             f"- 用时 {summary['elapsed_seconds']:.0f} s（扫描 {summary['scan_seconds']:.0f} s）。"]
    (Path(output) / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="NEW run directory (e.g. runs/rrt_branches_plan_001)")
    p.add_argument("--full-cache", type=Path, default=Path("data/wm_full_001"))
    p.add_argument("--bench-cache", type=Path, default=Path("data/wm_train_bench_001"))
    p.add_argument("--bench-split", type=Path, default=Path("data/danger_train_split_001/benchmark_split.json"))
    p.add_argument("--ft-run", type=Path, default=Path("runs/lewm_ft_B_001/run.json"), help="names the bench rows B was trained on")
    p.add_argument("--classifier-run", type=Path, default=Path("runs/safety_classifier_001/run.json"), help="val dev/report ids (refused)")
    p.add_argument("--goal-images", type=Path, default=None, help="goal image dataset (e.g. data/goal_images_001); optional")
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--quota", default=None, help="e.g. T=1500,F=1500,G=500,M=750 (default)")
    p.add_argument("--scale", type=float, default=1.0, help="multiplies every quota (pilot)")
    p.add_argument("--families", action="append", default=None, help="override one stratum's rollouts, e.g. G=recorded,hold,approach,approach")
    p.add_argument("--max-source-episodes", type=int, default=0, help="scan only a seeded subset of this many episodes per source (pilot); 0 = all")
    p.add_argument("--workers-for-estimate", type=int, default=3, choices=range(1, 4))
    p.add_argument("--families-b", action="append", default=None,
                   help="a second family set for one stratum, e.g. T=recorded,hold,push,push,push4,push4,push_hold,guard")
    p.add_argument("--families-b-fraction", default=None, help="share of that stratum's anchors using set B, e.g. T=0.34 (seeded per anchor)")
    p.add_argument("--f-near-radius-m", type=float, default=NEAR_RADIUS_M,
                   help="horizontal radius of F's height-gated 'near a guard' test (default 0.06 = pilot_001; T and M keep 0.06)")
    p.add_argument("--recorded-max-tilt-h20", type=float, default=None,
                   help="only anchors whose RECORDED continuation stays <= this worst tilt over a+1..a+20 (default: no filter)")
    p.add_argument("--goal-image-strata", default=None,
                   help="e.g. G: anchors of these strata only from episodes with a linked goal image carrying an achieved COM "
                        "(their goal_xyz = that COM, user decision 2); needs --goal-images (default: none)")
    p.add_argument("--capacity-only", action="store_true",
                   help="scan and report how many anchors each stratum can supply (anchor_capacity); writes run.json / summary.json / "
                        "REPORT_ZH.md only (format rrt_branches_capacity_v1: not a plan, the collector refuses it)")
    # ---- stage-2 roles. Left out, none of them appears in run.json: the default mode writes exactly the files (and the
    # run.json key set) it wrote before the roles existed.
    p.add_argument("--role", choices=ROLES, default=None,
                   help="omitted = the default training mode; train (with --stage2-plan: the plan's frozen training command), dev "
                        "(branch-dev selection set) or confirm (random_branches_003); dev / confirm need --stage2-plan and take every "
                        "selection setting from it")
    p.add_argument("--stage2-plan", type=Path, default=None,
                   help="the frozen stage-2 plan (runs/rrt_data_plan_005/plan.json; sha256 must equal FROZEN_STAGE2_PLAN_SHA256)")
    p.add_argument("--eval-bench-cache", type=Path, default=None,
                   help=f"danger-benchmark evaluation cache (--role dev / confirm only; default {DEFAULT_EVAL_BENCH_CACHE}): "
                        "bench_test sources of --role confirm; hold-out hashes of --role dev")
    p.add_argument("--eval-bench-split", type=Path, default=None,
                   help=f"danger-benchmark split file (--role dev / confirm only; default {DEFAULT_EVAL_BENCH_SPLIT})")
    return p


DEFAULT_EVAL_BENCH_CACHE = "data/wm_bench_001"
DEFAULT_EVAL_BENCH_SPLIT = "data/danger_bench_split_001/benchmark_split.json"
STAGE2_OPTIONS = ("role", "stage2_plan", "eval_bench_cache", "eval_bench_split")


def default_mode_args(a):
    """run.json 'args' of the default mode: the stage-2 options are left out while they are unset, so the record has
    the key set of the plans written before the roles existed."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if not (k in STAGE2_OPTIONS and v is None)}


CAPACITY_FORMAT = "rrt_branches_capacity_v1"


def write_capacity_report(output, summary):
    c = summary["capacity"]
    lines = ["# RRT 动作律分支采集：每层最多能取多少个起点（只读扫描）\n",
             f"- 扫描训练侧 {json.dumps(summary['scanned'], ensure_ascii=False)} 条轨迹（sha256 与缓存 index 核对）；规则同计划（a ≥ {MIN_ANCHOR}、a + {HORIZON} ≤ 末帧、倾角 ≤ {SKIP_TILT_DEG:g}°、"
             f"同一轨迹锚点相隔 ≥ {MIN_GAP_FRAMES} 帧）；配额设为 {CAPACITY_QUOTA}（轮流抽到取完为止），seed {summary['seeds']['base']}。",
             "- 按优先级 T > F > G > M 一起取（同一轨迹里后面的层只能用前面剩下的帧）：" +
             "；".join(f"{s} {c['joint'][s]['anchors']}（{json.dumps(c['joint'][s]['by_source'])}）" for s in STRATA) + "。",
             "- 各层单独取：" + "；".join(f"{s} {c['alone'][s]['anchors']}（{json.dumps(c['alone'][s]['by_source'])}）" for s in STRATA) + "。",
             f"- 可用帧：{json.dumps(c['available'], ensure_ascii=False)}。",
             "- 帧是随机抽的，数字是这个 seed 的贪心轮流能达到的数，不是证明过的最大值。",
             f"- 目标画面：{json.dumps(summary['goal_images'], ensure_ascii=False)}。",
             f"- 用时 {summary['elapsed_seconds']:.0f} s。"]
    (Path(output) / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.role not in (None, "train") or a.stage2_plan is not None:
        return main_stage2(a)                                       # stage-2 roles; the code below is the unchanged default
    if a.eval_bench_cache is not None or a.eval_bench_split is not None:
        raise ValueError("--eval-bench-cache / --eval-bench-split belong to --role dev / confirm (with --stage2-plan)")
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    if a.seed < 0 or a.max_source_episodes < 0:
        raise ValueError("seed and --max-source-episodes must be non-negative")
    if not (math.isfinite(a.f_near_radius_m) and 0 < a.f_near_radius_m <= NEAR_RADIUS_M):
        raise ValueError("--f-near-radius-m must be in (0, 0.06]")
    if a.recorded_max_tilt_h20 is not None and not (math.isfinite(a.recorded_max_tilt_h20) and a.recorded_max_tilt_h20 > 0):
        raise ValueError("--recorded-max-tilt-h20 must be a positive number of degrees")
    parse_quotas(a.quota, a.scale)
    parse_family_sets(a.families)
    parse_family_sets_b(a.families_b, a.families_b_fraction)
    if parse_goal_image_strata(a.goal_image_strata) and a.goal_images is None:
        raise ValueError("--goal-image-strata needs --goal-images")
    inputs = [a.full_cache / "index.json", a.bench_cache / "index.json", a.bench_split, a.ft_run, a.classifier_run]
    for path in inputs:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    from wm_training.provenance import implementation_hashes
    root = Path(__file__).resolve().parent
    output.mkdir(parents=True)
    run = {"format": PLAN_FORMAT, "args": default_mode_args(a),
           "input_sha256": {str(p): sha256(p) for p in inputs},
           "implementation_sha256": {**implementation_hashes(IMPLEMENTATION_FILES, package_dir=root),
                                     **implementation_hashes(("danger_strata.py", "provenance.py"))},
           "python": platform.python_version(), "numpy": np.__version__,
           "rules": {"prefix": PREFIX, "horizon": HORIZON, "min_anchor": MIN_ANCHOR, "min_gap_frames": MIN_GAP_FRAMES,
                     "skip_tilt_deg": SKIP_TILT_DEG, "strata_priority": list(STRATA), "t_tilt_deg": list(T_TILT_DEG),
                     "fresh_max_tilt_deg": FRESH_MAX_TILT_DEG, "near_radius_m": NEAR_RADIUS_M, "near_grip_max_z_m": NEAR_GRIP_MAX_Z_M,
                     "green_near_max_z_m": GREEN_NEAR_MAX_Z_M, "goal_near_m": GOAL_NEAR_M, "approach_shell_m": list(APPROACH_SHELL_M),
                     "planner_edges": PLANNER_EDGES, "direction_std": list(law.DIRECTION_STD), "guard_angle_deg": law.GUARD_ANGLE_DEG,
                     "push_edges": dict(PUSH_EDGES), "f_near_radius_m": a.f_near_radius_m, "recorded_max_tilt_h20": a.recorded_max_tilt_h20,
                     "goal_image_strata": list(parse_goal_image_strata(a.goal_image_strata))},
           "test_episodes_opened": False, "evaluation_data_opened": False, "fitted_on_evaluation_data": False}
    if a.capacity_only:
        run["format"] = CAPACITY_FORMAT
    write_new_json(output / "run.json", run)
    try:
        plan = build_plan(a)
        if a.capacity_only:
            summary = {"format": CAPACITY_FORMAT, **plan, "test_episodes_opened": False}
            write_capacity_report(output, summary)
            write_new_json(output / "summary.json", summary)
            print(json.dumps(native({"joint": plan["capacity"]["joint"], "alone": plan["capacity"]["alone"]}), ensure_ascii=False), flush=True)
            return summary
        anchors = {"format": PLAN_FORMAT, "normalization": plan["normalization"], "wm_action_columns": [0, 1, 2, 4], "prefix": PREFIX,
                   "horizon": HORIZON, "families": plan["families"], "families_b": plan["families_b"], "law": plan["law"], "rules": run["rules"],
                   "goal_images": plan["goal_images"], "anchors": plan["records"]}
        write_new_json(output / "anchors.json", anchors)
        with (output / "cache_pixels.npz").open("xb") as handle:
            np.savez_compressed(handle, **plan["pixels"])
        summary = {"format": PLAN_FORMAT, "anchors": len(plan["records"]), "counts": plan["counts"], "quotas": plan["quotas"],
                   "families": plan["families"], "families_b": plan["families_b"], "available": plan["available"], "scanned": plan["scanned"], "sources": plan["sources"],
                   "goal_images": plan["goal_images"], "seeds": plan["seeds"], "estimate": plan["estimate"],
                   "files_sha256": {"anchors.json": sha256(output / "anchors.json"), "cache_pixels.npz": sha256(output / "cache_pixels.npz")},
                   "scan_seconds": plan["scan_seconds"], "elapsed_seconds": plan["elapsed_seconds"], "test_episodes_opened": False}
        write_plan_report(output, summary)
        write_new_json(output / "summary.json", summary)
        print(json.dumps(native({k: summary[k] for k in ("anchors", "counts", "estimate")}), ensure_ascii=False), flush=True)
        return summary
    except BaseException as error:
        write_new_json(output / "FAILED.json", {"error": f"{type(error).__name__}: {error}"})
        raise


# ===== stage-2 roles (stage 2 step 1, 2026-09-28; the default training mode above never reaches this code) ==========
ROLES = ("train", "dev", "confirm")
ROLE_SOURCES = {"train": SOURCES, "dev": ("val_dev", "bench_dev"), "confirm": ("val_report", "bench_test")}
ROLE_HALF = {"dev": "dev", "confirm": "report"}
ROLE_TITLES = {"train": "训练集（只来自训练侧）", "dev": "branch-dev 选择集（只做选择，永不训练）", "confirm": "确认集（第 3 阶段只读一次）"}
FROZEN_STAGE2_PLAN_SHA256 = "102c069d84ec089df4888b30d7b2f7760dba4e48cb8a4c16ece17753239f194f"   # runs/rrt_data_plan_005/plan.json
STAGE2_PLAN_FORMAT = "rrt_data_plan_v2"
STAGE2_SECTIONS = {"train": "training_collection", "dev": "branch_dev_set", "confirm": "confirmation_set"}
RAW_DATASET_PREFIX = {"bench_train": "danger_train", "bench_dev": "danger_train", "bench_test": "danger_bench"}
# selection options a dev / confirm plan takes from plan.json (passing any of them is refused)
ROLE_FIXED_OPTIONS = ("seed", "quota", "scale", "families", "families_b", "families_b_fraction", "f_near_radius_m", "recorded_max_tilt_h20",
                      "goal_image_strata", "max_source_episodes")
TRAIN_COMMAND_PATHS = ("goal_images", "full_cache", "bench_cache", "bench_split", "ft_run", "classifier_run")


def load_stage2_plan(path):
    """(plan.json document, {'path', 'sha256'}) of the FROZEN stage-2 plan; any other file is refused."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256(path)
    if digest != FROZEN_STAGE2_PLAN_SHA256:
        raise ValueError(f"{path}: sha256 {digest} is not the frozen stage-2 plan ({FROZEN_STAGE2_PLAN_SHA256})")
    doc = read_json(path)
    if doc.get("format") != STAGE2_PLAN_FORMAT or doc.get("frozen") is not True:
        raise ValueError(f"{path}: not a frozen {STAGE2_PLAN_FORMAT} plan")
    return doc, {"path": str(path.resolve()), "sha256": digest}


def check_stage2_name(path, planned):
    """The directory ``path`` must carry the plan's name ``planned`` (e.g. data/rrt_branches_dev_001: same parent folder
    name, same stem) or, when that is taken, the NEXT free number: every lower number from the planned one must exist.
    Returns a record for run.json."""
    planned = Path(str(planned).replace("\\", "/"))
    m = re.fullmatch(r"(.+_)(\d{3})", planned.name)
    if m is None:
        raise ValueError(f"planned name {planned} has no _NNN number")
    stem, first = m.group(1), int(m.group(2))
    path = Path(path).resolve()
    mm = re.fullmatch(re.escape(stem) + r"(\d{3})", path.name)
    if mm is None or path.parent.name != planned.parent.name:
        raise ValueError(f"{path}: the stage-2 plan names this directory {planned} (or the next free number in {planned.parent.name}/)")
    number = int(mm.group(1))
    if number < first:
        raise ValueError(f"{path}: numbers below the planned {planned.name} are not allowed")
    free = [f"{stem}{k:03d}" for k in range(first, number) if not (path.parent / f"{stem}{k:03d}").exists()]
    if free:
        raise ValueError(f"{path}: {free[0]} is still free; the plan's name (or the next free number) must be used")
    return {"planned": str(planned), "used": str(path), "number": number, "next_number_rule_applied": number > first}


def frozen_training_args(doc):
    """The stage-2 plan's training command parsed by this tool's own parser."""
    words = shlex.split(doc["training_collection"]["command"])
    if not words or Path(words[0]).name != "rrt_branches_plan.py":
        raise ValueError("the stage-2 plan's training command is not an rrt_branches_plan.py command")
    return build_parser().parse_args(words[1:])


def _plan_count(text, pattern):
    m = re.search(pattern, str(text))
    if m is None:
        raise ValueError(f"cannot read {pattern!r} from the stage-2 plan text {text!r}")
    return [int(g) for g in m.groups()]


def _quota_value(value):
    """An integer quota, or 'all available, at most N' -> N (select_anchors takes everything when fewer are available)."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    (n,) = _plan_count(value, r"^all available, at most (\d+)$")
    return n


def role_spec(doc, role):
    """Selection settings of a dev / confirm plan, read from the frozen plan.json and cross-checked against its own text
    (families = the training A / B2 sets, the confirmation multiplicities m / mb, pool sizes)."""
    if role not in ("dev", "confirm"):
        raise ValueError("role_spec is for --role dev / confirm")
    t = doc["training_collection"]
    fr = frozen_training_args(doc)
    tf = t["families"]
    set_a, set_b2 = tuple(tf["T"]["A"]), tuple(tf["T"]["B2"])
    if tuple(tf["F"]) != set_b2 or tuple(tf["G"]) != set_a or tuple(tf["M"]) != set_a:
        raise ValueError("the training families are not the A / B2 sets the stage-2 plan describes")
    parse_family_sets([f"T={','.join(set_a)}", f"F={','.join(set_b2)}"])            # known families, rollout 0 = recorded
    common = {"role": role, "f_near_radius_m": float(fr.f_near_radius_m), "goal_images": str(fr.goal_images),
              "goal_image_strata": parse_goal_image_strata(fr.goal_image_strata), "half": ROLE_HALF[role]}
    if role == "dev":
        b = doc["branch_dev_set"]
        if b["families"] != tf:
            raise ValueError("branch-dev families differ from the training families (the plan says 'same families')")
        (n_val,) = _plan_count(b["sources"]["wm_full_001 val DEV half"], r"\((\d+) episodes")
        (n_bench,) = _plan_count(b["sources"]["danger_train bench_dev rows"], r"\((\d+) rows")
        return {**common, "seed": int(b["seed"]), "quotas": {s: int(b["anchors"][s]) for s in STRATA},
                "sets_a": {"T": set_a, "F": set_b2, "G": set_a, "M": set_a}, "sets_b": {"T": set_b2},
                "fractions": {"T": float(tf["T"]["share_B2"])}, "expected": {"val_dev": n_val, "bench_dev": n_bench},
                "names": {k: b[k] for k in ("dataset", "plan_dir", "report_dir")}}
    c = doc["confirmation_set"]
    fam = c["families"]
    m, mb = int(c["multiplicity"]["m"]), int(c["multiplicity"]["mb"])
    head, stochastic = set_b2[:2], set_b2[2:]
    if head != ("recorded", "hold") or len(stochastic) != 7:
        raise ValueError("set B2 is not recorded + hold + 7 stochastic families")

    def multi(k):
        return head + stochastic * k
    checks = {"val_report T": tuple(fam["val_report T"]) == multi(m), "val_report F": tuple(fam["val_report F"]) == multi(m),
              "bench_test F": tuple(fam["bench_test F"]) == multi(mb), "val_report G / M": tuple(fam["val_report G / M"]) == set_a,
              "bench_test T": fam["bench_test T"] == tf["T"]}
    if not all(checks.values()):
        raise ValueError(f"confirmation families disagree with the stated multiplicities m = {m}, mb = {mb}: {checks}")
    q = c["quotas"]
    quotas = {src: {s: _quota_value(q[src].get(s, 0)) for s in STRATA} for src in ("val_report", "bench_test")}
    (n_val,) = _plan_count(c["sources"]["val_report"], r"(\d+) episodes\)")
    n_all, n_fall, n_neg = _plan_count(c["sources"]["bench_test"], r"(\d+) episodes = (\d+) fall \+ (\d+) negatives")
    if n_all != n_fall + n_neg:
        raise ValueError("bench_test pool size is not falls + negatives")

    def fixed(name, families):
        return {"kind": "fixed", "name": name, "families": families}
    families = {"val_report": {"T": fixed(f"B2x{m}", multi(m)), "F": fixed(f"B2x{m}", multi(m)), "G": fixed("A", set_a), "M": fixed("A", set_a)},
                "bench_test": {"T": {"kind": "mixed", "A": set_a, "B": set_b2, "fraction": float(tf["T"]["share_B2"])},
                               "F": fixed(f"B2x{mb}", multi(mb)), "G": fixed("A", set_a), "M": fixed("A", set_a)}}
    return {**common, "seed": int(c["seed"]), "quotas_by_source": quotas, "families_by_source": families, "multiplicity": {"m": m, "mb": mb},
            "expected": {"val_report": n_val, "bench_test_falls": n_fall, "bench_test_negatives": n_neg},
            "names": {k: c[k] for k in ("dataset", "plan_dir", "report_dir")}}


def role_families(spec, anchor, anchor_seed):
    """('A' | 'B' | 'B2xM', families) of one anchor of a dev / confirm plan (seeded like family_set_for)."""
    if spec["role"] == "dev":
        return family_set_for(anchor_seed, anchor["stratum"], spec["sets_a"], spec["sets_b"], spec["fractions"])
    f = spec["families_by_source"][anchor["source"]][anchor["stratum"]]
    if f["kind"] == "fixed":
        return f["name"], f["families"]
    s = anchor["stratum"]
    return family_set_for(anchor_seed, s, {s: f["A"]}, {s: f["B"]}, {s: f["fraction"]})


def check_anchor_side(role, anchor):
    """Refuse an anchor whose source is not one of the role's sources, or whose raw episode lies in the wrong dataset
    family (bench_train / bench_dev: danger_train_*; bench_test: danger_bench_*; val / train: never a danger_* dataset)."""
    src = anchor["source"]
    name = src["source"]
    if name not in ROLE_SOURCES[role]:
        raise ValueError(f"anchor {anchor.get('id')}: source {name!r} is not a {role}-side source {ROLE_SOURCES[role]}")
    if anchor.get("set", name) != name:
        raise ValueError(f"anchor {anchor.get('id')}: set {anchor.get('set')!r} differs from its source {name!r}")
    folder = Path(str(src["episode_dir"]).replace("\\", "/")).parent.name
    prefix = RAW_DATASET_PREFIX.get(name)
    if (prefix is not None and not folder.startswith(prefix)) or (prefix is None and folder.startswith("danger_")):
        raise ValueError(f"anchor {anchor.get('id')}: a {name} start must not come from {folder}")
    return True


def val_half_sources(index_path, classifier_run, half, expected=None):
    """(rows, normalization, info) of the wm_full_001 val DEV ('dev') or REPORT ('report') half; refuses any overlap with
    train / test ids or the other half."""
    index = read_json(index_path)
    if index.get("status") != "complete" or index.get("eval_cache"):
        raise ValueError(f"{index_path}: not a complete training cache index")
    c = read_json(classifier_run)
    dev, report = [int(i) for i in c["dev_episode_ids"]], [int(i) for i in c["report_episode_ids"]]
    ids, other = (dev, report) if half == "dev" else (report, dev)
    if half not in ("dev", "report") or len(set(ids)) != len(ids) or set(dev) & set(report):
        raise ValueError("classifier halves must be two disjoint lists of distinct ids")
    val = {int(r["id"]): r for r in index["episodes"]["val"]}
    held = ({int(r["id"]) for part in ("train", "test") for r in index["episodes"].get(part, [])}
            | {int(i) for part in ("train", "test") for i in (index.get("split_ids") or {}).get(part, [])})
    if not set(ids) <= set(val) or set(ids) & held or set(ids) & set(other):
        raise ValueError(f"the val {half} half must be val episodes outside train / test and the other half")
    if expected is not None and len(ids) != int(expected):
        raise ValueError(f"the val {half} half has {len(ids)} episodes, the stage-2 plan says {expected}")
    raw_root = Path(index["source_root"])
    source = "val_dev" if half == "dev" else "val_report"
    rows = [{"source": source, "episode_id": eid, "cache_episode_id": eid, "episode_dir": str(raw_root / f"episode_{eid:04d}"),
             "npz_sha256": val[eid]["source_npz_sha256"], "metadata_sha256": val[eid]["source_metadata_sha256"],
             "cache_h5": str(Path(index_path).parent / val[eid]["path"]), "frames": int(val[eid]["num_states"])} for eid in sorted(ids)]
    info = {"index": str(index_path), "raw_root": str(raw_root), "half": half, "episodes": len(rows), "classifier_run": str(classifier_run),
            "source_manifest_sha256": index["source_manifest_sha256"]}
    return rows, index["normalization"], info


def bench_dev_sources(index_path, split_path, ft_run_path, expected=None):
    """(rows, normalization, info) of the bench_dev rows of wm_train_bench_001 (the part B was NOT trained on; B's
    training part must be bench_test); only danger_train datasets; bench_test rows refused."""
    index, split, ft = read_json(index_path), read_json(split_path), read_json(ft_run_path)
    if ft["finetune"]["data"]["bench_rows_used_for_training"] != "bench_test":
        raise ValueError(f"{ft_run_path}: B was not trained on bench_test, so bench_dev is not its held-out part")
    if index.get("status") != "complete" or index.get("split_sha256") != sha256(split_path):
        raise ValueError("the bench cache was not built from this split file")
    if Path(ft["bench_cache"]).name != Path(index_path).parent.name or ft["bench_cache_index_sha256"] != sha256(index_path):
        raise ValueError(f"{ft_run_path}: B was trained with a different bench cache index")
    datasets = {}
    for row in split["datasets"]:
        folder = Path(row["path"])
        if not folder.name.startswith("danger_train"):
            raise ValueError(f"{folder}: bench_dev starts come only from danger_train datasets")
        datasets[int(row["index"])] = folder
    dev_ids = {int(i) for i in index["bench_split"]["bench_dev"]}
    other = {int(i) for k, v in index["bench_split"].items() if k != "bench_dev" for i in v}
    if dev_ids & other:
        raise ValueError("bench_split parts overlap")
    if expected is not None and len(dev_ids) != int(expected):
        raise ValueError(f"bench_dev has {len(dev_ids)} rows, the stage-2 plan says {expected}")
    entries = {(int(e["dataset"]), int(e["episode_id"])): e for e in split["episodes"]}
    rows = []
    for r in index["episodes"]["val"]:
        gid = int(r["id"])
        if gid not in dev_ids:
            continue
        if gid != int(r["source_dataset"]) * GLOBAL_ID_STRIDE + int(r["source_id"]):
            raise ValueError(f"bench cache id {gid} does not match its source")
        entry = entries[(int(r["source_dataset"]), int(r["source_id"]))]
        rows.append({"source": "bench_dev", "episode_id": gid, "cache_episode_id": gid,
                     "episode_dir": str(datasets[int(r["source_dataset"])] / entry["path"]), "npz_sha256": r["source_npz_sha256"],
                     "metadata_sha256": r["source_metadata_sha256"], "cache_h5": str(Path(index_path).parent / r["path"]),
                     "frames": int(r["num_states"])})
    if len(rows) != len(dev_ids):
        raise ValueError("some bench_dev ids are missing from the cache index")
    info = {"index": str(index_path), "split": str(split_path), "part": "bench_dev", "episodes": len(rows), "refused_rows": len(other),
            "source_manifest_sha256": index["source_manifest_sha256"]}
    return sorted(rows, key=lambda r: r["episode_id"]), index["normalization"], info


def bench_test_sources(index_path, split_path, expected_falls=None, expected_negatives=None):
    """(rows, normalization, info) of the danger-benchmark bench_test pool of the confirmation set: the S1 / S2 / S3 fall
    events and the N1-N4 hard negatives of data/danger_bench_split_001 (the random_branches pool definition, 509 + 138 in
    the stage-2 plan); the other bench_test episodes (transient falls, unintended negatives) are listed, not used."""
    index, split = read_json(index_path), read_json(split_path)
    if index.get("status") != "complete" or not index.get("eval_cache") or index.get("split_sha256") != sha256(split_path):
        raise ValueError("the benchmark evaluation cache was not built from this split file")
    if split.get("format") != "danger_bench_split_v1":
        raise ValueError(f"{split_path}: not a danger-benchmark split")
    datasets = {}
    for row in split["datasets"]:
        folder = Path(row["path"])
        if not folder.name.startswith("danger_bench"):
            raise ValueError(f"{folder}: bench_test starts come only from the danger benchmark")
        datasets[int(row["index"])] = folder
    test_members = {(int(e["dataset"]), int(e["episode_id"])) for part in split["strata"].values() for e in part["test"]}
    dev_members = {(int(e["dataset"]), int(e["episode_id"])) for part in split["strata"].values() for e in part["dev"]}
    if test_members & dev_members:
        raise ValueError("bench-dev and bench-test overlap")
    if set(int(i) for i in index["bench_split"]["bench_test"]) != {d * GLOBAL_ID_STRIDE + e for d, e in test_members}:
        raise ValueError("bench cache bench_test ids disagree with the split")
    entries = [e for e in split["episodes"] if (int(e["dataset"]), int(e["episode_id"])) in test_members]
    falls = [e for e in entries if e.get("onset_frame") is not None and "|" in e["stratum"] and e["stratum"].split("|", 1)[1] in rb.SEVERITIES]
    negatives = [e for e in entries if e.get("onset_frame") is None and e["stratum"] in rb.NEGATIVE_STRATA]
    if (expected_falls is not None and len(falls) != int(expected_falls)) or (expected_negatives is not None and len(negatives) != int(expected_negatives)):
        raise ValueError(f"bench_test pool: {len(falls)} falls + {len(negatives)} negatives, the stage-2 plan says {expected_falls} + {expected_negatives}")
    pool = {(int(e["dataset"]), int(e["episode_id"])): e for e in falls + negatives}
    excluded = Counter(e["stratum"] for e in entries if (int(e["dataset"]), int(e["episode_id"])) not in pool)
    bench_rows = {(int(r["source_dataset"]), int(r["source_id"])): r for r in index["episodes"]["val"]}
    rows = []
    for key, entry in sorted(pool.items()):
        r = bench_rows[key]
        gid = key[0] * GLOBAL_ID_STRIDE + key[1]
        if int(r["id"]) != gid or int(r["num_states"]) != int(entry["frames"]):
            raise ValueError(f"bench cache row {r['id']} does not match the split entry {key}")
        rows.append({"source": "bench_test", "episode_id": gid, "cache_episode_id": gid, "episode_dir": str(datasets[key[0]] / entry["path"]),
                     "npz_sha256": r["source_npz_sha256"], "metadata_sha256": r["source_metadata_sha256"],
                     "cache_h5": str(Path(index_path).parent / r["path"]), "frames": int(r["num_states"])})
    info = {"index": str(index_path), "split": str(split_path), "part": "bench_test", "episodes": len(rows), "falls": len(falls),
            "negatives": len(negatives), "bench_test_not_in_pool": dict(excluded), "refused_bench_dev_rows": len(dev_members),
            "source_manifest_sha256": index["source_manifest_sha256"]}
    return sorted(rows, key=lambda r: r["episode_id"]), index["normalization"], info


def _source_hashes(rows):
    return {str(r["source_npz_sha256"]) for r in rows if r.get("source_npz_sha256")}


def role_hash_guard(role, rows, full_index_path, classifier_run, train_bench_index_path, eval_bench_index_path):
    """Raw-episode (trajectory.npz sha256) guard of a dev / confirm plan: every start must lie in the role's own pools and
    in no other pool (wm_full_001 train / test / the other val half, the training or held-out bench parts)."""
    full, c = read_json(full_index_path), read_json(classifier_run)
    dev, report = {int(i) for i in c["dev_episode_ids"]}, {int(i) for i in c["report_episode_ids"]}
    val = full["episodes"]["val"]
    pools = {"wm_full_001 train": _source_hashes(full["episodes"]["train"]), "wm_full_001 test": _source_hashes(full["episodes"].get("test", [])),
             "wm_full_001 val dev half": _source_hashes([r for r in val if int(r["id"]) in dev]),
             "wm_full_001 val report half": _source_hashes([r for r in val if int(r["id"]) in report])}
    for name, path in (("wm_train_bench_001", train_bench_index_path), ("wm_bench_001", eval_bench_index_path)):
        index = read_json(path)
        by_id = {int(r["id"]): r for r in index["episodes"]["val"]}
        for part, ids in index["bench_split"].items():
            pools[f"{name} {part}"] = _source_hashes([by_id[int(i)] for i in ids if int(i) in by_id])
    allowed = {"dev": ("wm_full_001 val dev half", "wm_train_bench_001 bench_dev"),
               "confirm": ("wm_full_001 val report half", "wm_bench_001 bench_test")}[role]
    digests = {str(r["npz_sha256"]) for r in rows}
    shared = {name: len(digests & pool) for name, pool in pools.items() if name not in allowed and digests & pool}
    if shared:
        raise ValueError(f"--role {role}: start episodes also belong to other pools {shared}; starts from the wrong side are refused")
    outside = digests - set().union(*(pools[name] for name in allowed))
    if outside:
        raise ValueError(f"--role {role}: {len(outside)} start episode(s) belong to none of the role's pools {allowed}")
    return {"role": role, "allowed_pools": list(allowed), "pools_checked": sorted(pools), "episodes": len(digests), "shared_with_other_pools": 0,
            "rule": "trajectory.npz sha256 of every scanned start episode vs every pool of wm_full_001 / wm_train_bench_001 / wm_bench_001"}


def load_val_goal_images(directory, half, halves, variant=0):
    """{('wm_full_001', episode_id): link} of the val goal images of one half (goal_images_v2 set 'val', the given variant,
    tag = half); every val image's tag must agree with the classifier halves. A link names its image (set / name / tag /
    variant) besides goal_image_id and goal_xyz (= achieved_green_xyz)."""
    if directory is None:
        raise ValueError("--role dev / confirm needs --goal-images")
    manifest = Path(directory) / "manifest.json"
    doc = read_json(manifest)
    if doc.get("format") != "goal_images_v2" or doc.get("status") != "complete":
        raise ValueError(f"{manifest}: not a complete goal_images_v2 dataset")
    ids = {"dev": {int(i) for i in halves["dev_episode_ids"]}, "report": {int(i) for i in halves["report_episode_ids"]}}
    out, skipped = {}, Counter()
    for e in doc.get("images") or []:
        if e.get("set") != "val":
            skipped[f"set {e.get('set')}"] += 1
            continue
        if e.get("status") != "accepted":
            skipped["status"] += 1
            continue
        if e.get("split") != "val" or e.get("cache") != "wm_full_001":
            raise ValueError(f"{manifest}: val image {e.get('name')} is not a wm_full_001 val image")
        eid, tag = int(e["source_episode_id"]), e.get("tag")
        if tag not in ids or eid not in ids[tag]:
            raise ValueError(f"{manifest}: goal image {e.get('name')} (episode {eid}) has tag {tag!r}, which disagrees with the classifier halves")
        if int(e.get("variant", -1)) != int(variant):
            skipped["other variant"] += 1
            continue
        if tag != half:
            skipped[f"tag {tag}"] += 1
            continue
        key = ("wm_full_001", eid)
        if key in out:
            raise ValueError(f"{manifest}: two variant-{variant} goal images for {key}")
        xyz = [float(v) for v in e["achieved_green_xyz"]]
        if len(xyz) != 3 or not np.isfinite(xyz).all():
            raise ValueError(f"{manifest}: bad goal point for {key}")
        out[key] = {"goal_image_id": int(e["id"]), "goal_xyz": xyz, "goal_image_set": "val", "goal_image_name": str(e["name"]),
                    "goal_image_tag": tag, "goal_image_variant": int(variant)}
    info = {"path": str(directory), "status": "linked", "manifest_sha256": sha256(manifest), "set": "val", "tag": half, "variant": int(variant),
            "entries": len(out), "half_episodes_without_goal_image": sorted(ids[half] - {k[1] for k in out}), "skipped": dict(skipped)}
    return out, info


def build_role_plan(a, spec):
    """build_plan for --role dev / confirm (same scan, strata, rollout drawing and record format; own sources)."""
    role = spec["role"]
    full_index_path = a.full_cache / "index.json"
    if read_json(a.ft_run).get("cache_index_sha256") != sha256(full_index_path):
        raise ValueError(f"{a.ft_run}: B was trained on a different wm_full_001 index")
    halves = read_json(a.classifier_run)
    if role == "dev":
        val_rows, val_norm, val_info = val_half_sources(full_index_path, a.classifier_run, "dev", spec["expected"]["val_dev"])
        bench_rows, bench_norm, bench_info = bench_dev_sources(a.bench_cache / "index.json", a.bench_split, a.ft_run, spec["expected"]["bench_dev"])
    else:
        val_rows, val_norm, val_info = val_half_sources(full_index_path, a.classifier_run, "report", spec["expected"]["val_report"])
        bench_rows, bench_norm, bench_info = bench_test_sources(a.eval_bench_cache / "index.json", a.eval_bench_split,
                                                                spec["expected"]["bench_test_falls"], spec["expected"]["bench_test_negatives"])
    for key in ("mean", "scale"):
        if not np.array_equal(val_norm[key], bench_norm[key]):
            raise ValueError("the two caches use different action normalizations")
    config_norm = {k: val_norm[k] for k in ("names", "mean", "scale")}
    config = law.LawConfig(normalization=config_norm)
    guard = role_hash_guard(role, val_rows + bench_rows, full_index_path, a.classifier_run, a.bench_cache / "index.json",
                            a.eval_bench_cache / "index.json")
    goal_links, goal_info = load_val_goal_images(a.goal_images, spec["half"], halves)
    names = ROLE_SOURCES[role]
    ss = np.random.SeedSequence(spec["seed"])
    seed_scan, seed_select, seed_seq = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in ss.spawn(3)]
    started = time.perf_counter()
    rows = []
    rng = np.random.default_rng(seed_scan)
    for group in (val_rows, bench_rows):
        rows += [group[i] for i in rng.permutation(len(group))]
    episodes, sources = {}, {}
    for r in rows:
        folder = Path(r["episode_dir"])
        if sha256(folder / "trajectory.npz") != r["npz_sha256"] or sha256(folder / "metadata.json") != r["metadata_sha256"]:
            raise ValueError(f"{folder}: raw files differ from the ones the cache was built from")
        arrays = load_episode_arrays(folder / "trajectory.npz")
        if len(arrays["time"]) != r["frames"]:
            raise ValueError(f"{folder}: frame count differs from the cache index")
        key = (r["source"], r["episode_id"])
        episodes[key], sources[key] = arrays, r
    scan_seconds = time.perf_counter() - started
    gi_strata = spec["goal_image_strata"]
    restrict = goal_image_episodes(episodes, goal_links, gi_strata) if gi_strata else None
    if role == "dev":
        anchors, available = select_anchors(episodes, spec["quotas"], seed_select, spec["f_near_radius_m"], None, restrict, sources=names)
        quotas = spec["quotas"]
    else:
        seeds = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in np.random.SeedSequence(seed_select).spawn(len(names))]
        anchors, available = [], {}
        for src, seed in zip(names, seeds):
            part = {k: v for k, v in episodes.items() if k[0] == src}
            sub = None if restrict is None else {s: {k for k in keys if k[0] == src} for s, keys in restrict.items()}
            got, av = select_anchors(part, spec["quotas_by_source"][src], seed, spec["f_near_radius_m"], None, sub, sources=names)
            anchors += got
            available[src] = av
        quotas = spec["quotas_by_source"]
    seq_seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in np.random.SeedSequence(seed_seq).spawn(len(anchors))]
    records, pixels = [], {}
    for i, (anchor, anchor_seed) in enumerate(zip(anchors, seq_seeds)):
        key = (anchor["source"], anchor["episode_id"])
        src, arrays = sources[key], episodes[key]
        block, n_states = rb.read_cache_pixels(src["cache_h5"], anchor["frame"], src["npz_sha256"])
        if n_states != src["frames"]:
            raise ValueError(f"{src['cache_h5']}: frame count differs from the raw episode")
        del block                                            # re-read when cache_pixels.npz is written (CachePixelBlock)
        pixels[f"a{i:04d}"] = CachePixelBlock(src["cache_h5"], anchor["frame"], src["npz_sha256"])
        link = goal_links.get((SOURCE_CACHE[anchor["source"]], anchor["episode_id"]))
        source = {k: src[k] for k in ("source", "episode_id", "cache_episode_id", "episode_dir", "npz_sha256", "metadata_sha256", "cache_h5", "frames")}
        set_name, fams = role_families(spec, anchor, anchor_seed)
        record = make_anchor_record(i, anchor, anchor_seed, source, arrays, fams, link, config, spec["f_near_radius_m"], gi_strata)
        record["family_set"] = set_name
        check_anchor_side(role, record)
        records.append(record)
    rollouts = sum(len(r["sequences"]) for r in records)
    families = ({"A": {k: list(v) for k, v in spec["sets_a"].items()}, "B": {k: list(v) for k, v in spec["sets_b"].items()}, "fractions": spec["fractions"]}
                if role == "dev" else json.loads(json.dumps(spec["families_by_source"])))
    return {"records": records, "pixels": pixels, "normalization": config_norm, "law": config.to_json(), "quotas": quotas, "families": families,
            "available": available, "goal_images": goal_info, "sources": {names[0]: val_info, names[1]: bench_info},
            "scanned": {s: sum(1 for k in episodes if k[0] == s) for s in names}, "guard": guard,
            "seeds": {"base": spec["seed"], "scan": seed_scan, "select": seed_select, "sequences": seed_seq},
            "counts": {"by_stratum": dict(Counter(r["stratum"] for r in records)), "by_group": dict(Counter(r["group"] for r in records)),
                       "by_stratum_family_set": dict(Counter(f"{r['stratum']}/{r['family_set']}" for r in records)),
                       "rollouts": rollouts, "families": dict(Counter(s["family"] for r in records for s in r["sequences"])),
                       "goal_images_linked_G": sum(1 for r in records if r["stratum"] == "G" and r["goal"]["goal_image_id"] >= 0),
                       "approach": {"rollouts": 0, "feasible": 0}},
            "estimate": cost_estimate(len(records), rollouts, a.workers_for_estimate), "scan_seconds": scan_seconds,
            "elapsed_seconds": time.perf_counter() - started}


def check_training_command(a, doc):
    """--role train with --stage2-plan runs the plan's frozen training command: every selection option and input path
    must equal it (the output name is checked separately)."""
    fr = frozen_training_args(doc)
    differ = [k for k in ROLE_FIXED_OPTIONS if getattr(a, k) != getattr(fr, k)]
    for k in TRAIN_COMMAND_PATHS:
        mine, theirs = getattr(a, k), getattr(fr, k)
        if (mine is None) != (theirs is None) or (mine is not None and Path(mine).resolve() != Path(theirs).resolve()):
            differ.append(k)
    if differ:
        raise ValueError(f"--role train with --stage2-plan must run the plan's training command exactly; it differs in {differ}")
    return {"command": doc["training_collection"]["command"], "identical_options": list(ROLE_FIXED_OPTIONS) + list(TRAIN_COMMAND_PATHS)}


def write_role_plan_report(output, summary):
    c, e, s2 = summary["counts"], summary["estimate"], summary["stage2_plan"]
    role = summary["role"]
    lines = [f"# RRT 动作律分支采集：{ROLE_TITLES[role]}的起点与动作计划（角色 {role}）\n",
             f"- 冻结方案 `{s2['path']}`（sha256 `{s2['sha256']}`，节 {s2['section']}，seed {s2['seed']}）；数据集名 `{s2['dataset']}`，报告目录 `{s2['report_dir']}`。",
             f"- 来源：{json.dumps(summary['sources'], ensure_ascii=False)}；扫描 {json.dumps(summary['scanned'], ensure_ascii=False)} 条（每条原始轨迹的 trajectory.npz / "
             "metadata.json sha256 与缓存 index 核对）。",
             f"- 另一侧的起点一律拒绝：{json.dumps(summary.get('guard'), ensure_ascii=False)}。",
             f"- 各层锚点：{json.dumps(c['by_stratum'], ensure_ascii=False)}（配额 {json.dumps(summary['quotas'], ensure_ascii=False)}）；按来源/层 "
             f"{json.dumps(c['by_group'], ensure_ascii=False)}；按族套 {json.dumps(c['by_stratum_family_set'], ensure_ascii=False)}。",
             f"- 可用帧：{json.dumps(summary['available'], ensure_ascii=False)}。",
             f"- rollout 共 {c['rollouts']} 条：{json.dumps(c['families'], ensure_ascii=False)}。",
             f"- 目标画面：{json.dumps(summary['goal_images'], ensure_ascii=False)}；G 层链接到目标画面的锚点 {c['goal_images_linked_G']} 个。",
             f"- 第 2 阶段估计（RRT_BRANCHES.md 的成本模型）：{e['gib']:.2f} GiB，{e['wall_hours']['3']:.2f} 小时（3 个 worker）。",
             f"- 用时 {summary['elapsed_seconds']:.0f} s（扫描 {summary['scan_seconds']:.0f} s）。"]
    if role == "confirm":
        lines.append("- 这是确认集的计划：只有采集器的仿真真值检查可以读它；第 3 阶段冻结检查点之后，world model 只能读一次（evaluate_rrt_confirmation.py --stage3）。")
    if role == "dev":
        lines.append("- 这是 branch-dev 选择集：只用于第 3 阶段的检查点 / 阈值选择，永不作训练数据。")
    (Path(output) / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_stage2(a):
    """--role dev / confirm, or --role train with --stage2-plan: same outputs as the default mode plus 'role' and
    'stage2_plan' in anchors.json / summary.json / run.json."""
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    a.role = a.role or "train"                                        # --stage2-plan alone = the plan's training command
    a.eval_bench_cache = Path(DEFAULT_EVAL_BENCH_CACHE) if a.eval_bench_cache is None else a.eval_bench_cache
    a.eval_bench_split = Path(DEFAULT_EVAL_BENCH_SPLIT) if a.eval_bench_split is None else a.eval_bench_split
    if a.stage2_plan is None:
        raise ValueError(f"--role {a.role} needs --stage2-plan (the frozen runs/rrt_data_plan_005/plan.json)")
    if a.capacity_only:
        raise ValueError("--capacity-only is a training-side scan; it has no stage-2 role mode")
    doc, stage2 = load_stage2_plan(a.stage2_plan)
    section = doc[STAGE2_SECTIONS[a.role]]
    names = {"plan_dir": check_stage2_name(output, section["plan_dir"])}
    if a.role == "train":
        spec, command = None, check_training_command(a, doc)
        if a.seed < 0 or a.max_source_episodes < 0:
            raise ValueError("seed and --max-source-episodes must be non-negative")
        rules = None
    else:
        defaults = build_parser().parse_args(["--output", "unused"])
        changed = [k for k in ROLE_FIXED_OPTIONS if getattr(a, k) != getattr(defaults, k)]
        if changed:
            raise ValueError(f"--role {a.role} takes {changed} from the stage-2 plan; do not pass them")
        spec, command = role_spec(doc, a.role), None
        if a.goal_images is None or Path(a.goal_images).resolve() != Path(spec["goal_images"]).resolve():
            raise ValueError(f"--role {a.role} needs --goal-images {spec['goal_images']} (the stage-2 plan's goal-image dataset)")
        rules = {"prefix": PREFIX, "horizon": HORIZON, "min_anchor": MIN_ANCHOR, "min_gap_frames": MIN_GAP_FRAMES, "skip_tilt_deg": SKIP_TILT_DEG,
                 "strata_priority": list(STRATA), "t_tilt_deg": list(T_TILT_DEG), "fresh_max_tilt_deg": FRESH_MAX_TILT_DEG,
                 "near_radius_m": NEAR_RADIUS_M, "near_grip_max_z_m": NEAR_GRIP_MAX_Z_M, "green_near_max_z_m": GREEN_NEAR_MAX_Z_M,
                 "goal_near_m": GOAL_NEAR_M, "planner_edges": PLANNER_EDGES, "direction_std": list(law.DIRECTION_STD),
                 "guard_angle_deg": law.GUARD_ANGLE_DEG, "push_edges": dict(PUSH_EDGES), "f_near_radius_m": spec["f_near_radius_m"],
                 "recorded_max_tilt_h20": None, "goal_image_strata": list(spec["goal_image_strata"]),
                 "goal_image_link": f"goal_images_002 set 'val', variant 0, tag {spec['half']}",
                 "selection": "pooled round-robin over both sources (stage-1 rule)" if a.role == "dev" else "per source, its own quotas and seed"}
    inputs = [a.full_cache / "index.json", a.bench_cache / "index.json", a.bench_split, a.ft_run, a.classifier_run, a.stage2_plan]
    if a.role in ("dev", "confirm"):
        inputs += [a.eval_bench_cache / "index.json", Path(a.goal_images) / "manifest.json"] + ([a.eval_bench_split] if a.role == "confirm" else [])
    for path in inputs:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    stage2_record = {**stage2, "role": a.role, "section": STAGE2_SECTIONS[a.role], "seed": int(section["seed"]),
                     **{k: section[k] for k in ("dataset", "plan_dir", "report_dir")}}
    from wm_training.provenance import implementation_hashes
    root = Path(__file__).resolve().parent
    if rules is None:
        rules = {"prefix": PREFIX, "horizon": HORIZON, "min_anchor": MIN_ANCHOR, "min_gap_frames": MIN_GAP_FRAMES,
                 "skip_tilt_deg": SKIP_TILT_DEG, "strata_priority": list(STRATA), "t_tilt_deg": list(T_TILT_DEG),
                 "fresh_max_tilt_deg": FRESH_MAX_TILT_DEG, "near_radius_m": NEAR_RADIUS_M, "near_grip_max_z_m": NEAR_GRIP_MAX_Z_M,
                 "green_near_max_z_m": GREEN_NEAR_MAX_Z_M, "goal_near_m": GOAL_NEAR_M, "approach_shell_m": list(APPROACH_SHELL_M),
                 "planner_edges": PLANNER_EDGES, "direction_std": list(law.DIRECTION_STD), "guard_angle_deg": law.GUARD_ANGLE_DEG,
                 "push_edges": dict(PUSH_EDGES), "f_near_radius_m": a.f_near_radius_m, "recorded_max_tilt_h20": a.recorded_max_tilt_h20,
                 "goal_image_strata": list(parse_goal_image_strata(a.goal_image_strata))}
    output.mkdir(parents=True)
    run = {"format": PLAN_FORMAT, "role": a.role, "stage2_plan": stage2_record, "names": names, "training_command": command,
           "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "input_sha256": {str(p): sha256(p) for p in inputs},
           "implementation_sha256": {**implementation_hashes(IMPLEMENTATION_FILES, package_dir=root),
                                     **implementation_hashes(("danger_strata.py", "provenance.py"))},
           "python": platform.python_version(), "numpy": np.__version__, "rules": rules,
           "test_episodes_opened": False, "evaluation_data_opened": a.role == "confirm", "selection_data_opened": a.role == "dev",
           "world_model_loaded": False, "fitted_on_evaluation_data": False,
           "note": {"train": "training-side starts only", "dev": "branch-dev selection set: never training material",
                    "confirm": "confirmation set: collection-side truth only; the model-based evaluation happens once, in stage 3"}[a.role]}
    write_new_json(output / "run.json", run)
    try:
        plan = build_plan(a) if a.role == "train" else build_role_plan(a, spec)
        anchors = {"format": PLAN_FORMAT, "normalization": plan["normalization"], "wm_action_columns": [0, 1, 2, 4], "prefix": PREFIX,
                   "horizon": HORIZON, "families": plan["families"], "families_b": plan.get("families_b", {}), "law": plan["law"], "rules": rules,
                   "goal_images": plan["goal_images"], "role": a.role, "stage2_plan": stage2_record, "anchors": plan["records"]}
        write_new_json(output / "anchors.json", anchors)
        with (output / "cache_pixels.npz").open("xb") as handle:
            np.savez_compressed(handle, **plan["pixels"])
        summary = {"format": PLAN_FORMAT, "role": a.role, "stage2_plan": stage2_record, "anchors": len(plan["records"]), "counts": plan["counts"],
                   "quotas": plan["quotas"], "families": plan["families"], "families_b": plan.get("families_b", {}), "available": plan["available"],
                   "scanned": plan["scanned"], "sources": plan["sources"], "guard": plan.get("guard"), "goal_images": plan["goal_images"],
                   "seeds": plan["seeds"], "estimate": plan["estimate"],
                   "files_sha256": {"anchors.json": sha256(output / "anchors.json"), "cache_pixels.npz": sha256(output / "cache_pixels.npz")},
                   "scan_seconds": plan["scan_seconds"], "elapsed_seconds": plan["elapsed_seconds"], "test_episodes_opened": False}
        if sha256(a.stage2_plan) != stage2["sha256"]:
            raise RuntimeError("the stage-2 plan changed while planning")
        if a.role == "train":
            write_plan_report(output, summary)
        else:
            write_role_plan_report(output, summary)
        write_new_json(output / "summary.json", summary)
        print(json.dumps(native({k: summary[k] for k in ("role", "anchors", "counts", "estimate")}), ensure_ascii=False), flush=True)
        return summary
    except BaseException as error:
        write_new_json(output / "FAILED.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
