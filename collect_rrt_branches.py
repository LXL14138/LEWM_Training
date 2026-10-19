"""SIMULATION ONLY: stage 2 of the RRT-action-law branch collection (training-side start states).

Reads a stage-1 plan (rrt_branches_plan.py: anchors.json + cache_pixels.npz) and, for every anchor a and each of its
rollouts (recorded, hold, planner, guard, edge_hold, approach; see rrt_branches_plan):

  restore the recorded physics_state of frame a-10 (collect_counterfactual_branches.restore_rebased), replay the 10
  recorded actions bit-exactly (qpos/qvel at a must equal the record, else it raises), then run 32 control steps of
  the rollout under rrt_action_law (LawRunner: fixed 2 cm/s edges, holds, anti-windup PI, WM-range clip) and record
  every state and the front_pixels image letterboxed EXACTLY like the cache (wm_training.prepare_data.letterbox).

Besides the random_branches_v2 arrays, every npz holds the edge directions, the NOMINAL actions the planner would
have fed to the world model (per segment at its start = receding horizon, and open loop from the anchor), the
executed actions, the tracking error and the anti-windup activity flags.

Output: one npz per anchor in a NEW dataset directory (manifest.json written once at the end), and run.json /
summary.json / REPORT_ZH.md (FAILED.json on error) in a NEW run directory. ``--plan-only`` prints the time / disk /
memory estimate and writes nothing. Doc: RRT_BRANCHES.md.

Stage-2 roles (2026-09-28): ``--role train`` / ``dev`` / ``confirm``. Without ``--role`` a role-less plan runs the
behaviour above: the npz arrays and the key sets of manifest.json / run.json are those of the collector before the roles
existed (role / stage2_plan / names are written only with --role or for a plan that carries a role). A plan is collected
only with its own role (a plan without a role is a 'train' plan; dev / confirm plans need --role explicitly); every
anchor's source must be one of the role's sources. A plan that carries a stage-2 record (every dev / confirm plan) is collected
whole, into the dataset / report names of the frozen stage-2 plan (or the next free number), and its plan.json sha256 is
re-checked. dev / confirm manifests also carry the decision-point truth pools of every anchor (rrt_pilot_yields rules;
simulator truth only, no world model).
"""
from __future__ import annotations

import argparse
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

import collect_random_branches as crb
from bench_restore_check import RENDER_KEYS, STATE_SPEC, load_episode, pinch_xyz
from collect_counterfactual_branches import restore_rebased
from collection_workers import CHILD_THREAD_LIMITS, run_workers
from exploration import ExplorationScene
import random_branches_plan as rb
import rrt_action_law as law
import rrt_branches_plan as rp
import rrt_pilot_yields as ry
from wm_training.danger_strata import native, read_json, sha256, write_new_json
from wm_training.provenance import implementation_hashes

FORMAT = "rrt_branches_v1"
IMAGE_SIZE, CAMERA = crb.IMAGE_SIZE, crb.CAMERA
H = rp.HORIZON
STEPS = H + 1
MAX_SEGMENTS = H // law.EDGE_STEPS
STEP_FIELDS = crb.STEP_FIELDS
LAW_STEP_ARRAYS = {"tracker_lag_m": ("lag_m", (), np.float64), "reference_frozen": ("reference_frozen", (), bool),
                   "integral_clamped": ("integral_clamped", (), bool), "reference_dragged": ("reference_dragged", (), bool),
                   "box_clipped": ("box_clipped", (), bool), "range_clipped": ("range_clipped", (), bool),
                   "wm_clipped": ("wm_clipped", (3,), bool), "integral_m": ("integral_m", (3,), np.float64),
                   "pi_output_m": ("u_m", (3,), np.float64), "segment_index": ("segment_index", (), np.int64)}
OUTCOME_HORIZONS = (8, 16, 20, 24, 32)
CHECK_H = law.CHECK_STEPS                         # 20: the planner's safety-check horizon (edge 8 + hold 12)
CONTACT_N = 1e-3
DISK_RESERVE_BYTES = 5 * 1024**3
ROOT_FILES = ("collect_rrt_branches.py", "rrt_branches_plan.py", "rrt_action_law.py", "collect_random_branches.py", "random_branches_plan.py",
              "bench_restore_check.py", "collect_counterfactual_branches.py", "exploration.py", "collect_grasp_trajs.py",
              "collection_workers.py", "three_block_scene.py", "scene_config.py", "task_progress.py", "rrt_pilot_yields.py")
WM_FILES = ("prepare_data.py", "danger_strata.py", "provenance.py")


# ----- one anchor -------------------------------------------------------------------------------------------------
CONTACT_FIELDS = ("robot_env_contact", "robot_green_contact", "green_env_contact")


def contact_state(env):
    """Active contacts of the current state -> (robot_env, robot_green, green_env).

    robot_env: a robot geom ('ur5e/...') touches anything but the robot and the green block (guards, floor);
    robot_green: a robot geom touches the green block (object_0; true while it is held);
    green_env: the green block touches anything but the robot (floor, guards). A rollout is FREE SPACE when, over
    steps 1..32, robot_env never holds and (held block) green_env never holds / (empty gripper) robot_green never
    holds: only then is the calibrated free-space nominal model expected to match the executed actions."""
    m, d = env.model, env.data
    robot_env = robot_green = green_env = False
    for i in range(d.ncon):
        c = d.contact[i]
        if c.efc_address < 0:
            continue
        b1, b2 = m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name
        r1, r2 = b1.startswith("ur5e/"), b2.startswith("ur5e/")
        if r1 and r2:
            continue
        if r1 or r2:
            if (b2 if r1 else b1) == "object_0":
                robot_green = True
            else:
                robot_env = True
        elif b1 == "object_0" or b2 == "object_0":
            green_env = True
    return robot_env, robot_green, green_env


def capture(env):
    out = crb.capture(env)
    out.update(zip(CONTACT_FIELDS, contact_state(env)))
    return out


def load_source(anchor):
    src = anchor["source"]
    folder = Path(src["episode_dir"])
    if sha256(folder / "trajectory.npz") != src["npz_sha256"] or sha256(folder / "metadata.json") != src["metadata_sha256"]:
        raise ValueError(f"{folder}: source files differ from the plan")
    loaded = load_episode(folder)
    with np.load(folder / "trajectory.npz", allow_pickle=False) as npz:
        loaded["class_label"] = np.asarray(npz["class_label"])
    frame = int(anchor["frame"])
    if loaded["frames"] != int(src["frames"]) or not (rp.MIN_ANCHOR <= frame and frame + H <= loaded["frames"] - 1):
        raise ValueError(f"{folder}: anchor {frame} not admissible for {loaded['frames']} frames")
    if loaded["camera"] != CAMERA or loaded["class_label"].shape != (loaded["frames"],):
        raise ValueError(f"{folder}: unexpected camera or labels")
    if float(loaded["actions"][frame - 1][4]) != float(anchor["grip_command"]):
        raise ValueError(f"{folder}: gripper command at a-1 differs from the plan")
    return loaded


def init_arrays(R):
    arrays = {k: np.zeros((R, STEPS, *shape), dtype=dtype) for k, (shape, dtype) in STEP_FIELDS.items()}
    arrays.update({k: np.zeros((R, STEPS), bool) for k in CONTACT_FIELDS})
    arrays.update(pixels=np.zeros((R, STEPS, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8), actions=np.zeros((R, H, 5)),
                  nominal_actions=np.full((R, H, 5), np.nan), nominal_open_loop_actions=np.full((R, H, 5), np.nan),
                  reference_xyz=np.full((R, H, 3), np.nan), segment_kind=np.full((R, MAX_SEGMENTS), "", dtype="U4"),
                  segment_direction=np.full((R, MAX_SEGMENTS, 3), np.nan), segment_steps=np.zeros((R, MAX_SEGMENTS), np.int64),
                  segment_speed_mps=np.full((R, MAX_SEGMENTS), np.nan), segment_continued=np.zeros((R, MAX_SEGMENTS), bool),
                  approach_predicted_end_xyz=np.full((R, 3), np.nan))
    for name, (_, shape, dtype) in LAW_STEP_ARRAYS.items():
        fill = -1 if dtype == np.int64 else (np.nan if dtype == np.float64 else False)
        arrays[name] = np.full((R, H, *shape), fill, dtype=dtype)
    return arrays


def run_law_rollout(env, arrays, r, segments, grip, config, model_name, previous_pinch, letterbox):
    """Execute ``segments`` from the current (anchor) state; fills row r. Returns nothing (arrays are written)."""
    runner = law.LawRunner(segments, grip, config)
    anchor_pinch = pinch_xyz(env)
    arrays["nominal_open_loop_actions"][r] = law.nominal_actions(segments, anchor_pinch, grip, config, model_name,
                                                                 prev_velocity=anchor_pinch - previous_pinch)
    for k in range(H):
        pinch = pinch_xyz(env)
        if runner.tracker is None or runner.tracker.finished:
            segment = segments[runner.index + 1]
            n = config.edge_steps if segment[0] == "edge" else segment[1]
            arrays["nominal_actions"][r, k:k + n] = law.nominal_actions([segment], pinch, grip, config, model_name,
                                                                        prev_velocity=pinch - previous_pinch, previous=runner.carry_state())
        action = runner.action(pinch)
        env.last_transition = None
        env.step(action)
        if env.last_transition is None or not np.array_equal(env.last_transition["action"], action):
            raise RuntimeError("the scene did not execute the law's action")
        arrays["actions"][r, k] = env.last_transition["action"]
        for key, value in capture(env).items():
            arrays[key][r, k + 1] = value
        arrays["pixels"][r, k + 1] = crb.preprocess(letterbox, env.render(camera=CAMERA))
        previous_pinch = pinch
    rec = runner.records()
    arrays["reference_xyz"][r] = rec["reference_xyz"]
    for name, (key, _, _) in LAW_STEP_ARRAYS.items():
        arrays[name][r] = rec[key]
    for i, (kind, arg) in enumerate(segments):
        arrays["segment_kind"][r, i] = kind
        arrays["segment_steps"][r, i] = config.edge_steps if kind == "edge" else arg
        arrays["segment_continued"][r, i] = rec["segment_continued"][i]
        if kind == "edge":
            arrays["segment_direction"][r, i] = arg
            arrays["segment_speed_mps"][r, i] = config.speed_mps
        else:
            arrays["segment_speed_mps"][r, i] = 0.0


def rollout_outcome(arrays, r, spec, goal_xyz, normalization):
    wm = arrays["wm_action"][r]
    reference = arrays["reference_xyz"][r]
    o = rb.rollout_outcome(arrays["guard_tilt_deg"][r], arrays["unsafe"][r], arrays["class_label"][r], arrays["pinch_xyz"][r],
                           arrays["guard_contact_peak_last_step_n"][r], arrays["green_guard_contact_peak_last_step_n"][r],
                           reference=reference, actions=arrays["actions"][r], wm_action=wm, horizons=OUTCOME_HORIZONS)
    held = bool(arrays["grasped"][r, 0])
    touching = arrays["robot_env_contact"][r, 1:].any() | (arrays["green_env_contact"][r, 1:].any() if held else arrays["robot_green_contact"][r, 1:].any())
    o.update(free_space=not bool(touching), robot_env_contact_steps=int(arrays["robot_env_contact"][r, 1:].sum()),
             green_env_contact_steps=int(arrays["green_env_contact"][r, 1:].sum()))
    o.update(rollout=r, family=spec["family"], grasped_start=held, grasped_end=bool(arrays["grasped"][r, -1]),
             hover_conditions_frames=int(arrays["hover_conditions_met"][r, 1:].sum()),
             green_goal_distance_final_m=float(np.linalg.norm(arrays["object_com_xyz"][r, -1, 0] - goal_xyz)),
             green_goal_distance_min_m=float(np.linalg.norm(arrays["object_com_xyz"][r, 1:, 0] - goal_xyz, axis=1).min()))
    if spec["family"] == "recorded":
        o.update({f"{k}_steps": None for k in ("reference_frozen", "reference_dragged", "integral_clamped", "wm_clipped", "box_clipped", "range_clipped")},
                 nominal_mismatch_max_mm=None, nominal_mismatch_median_mm=None, nominal_open_loop_mismatch_max_mm=None, approach=None)
        return o
    for k in ("reference_frozen", "reference_dragged", "integral_clamped", "box_clipped", "range_clipped"):
        o[f"{k}_steps"] = int(arrays[k][r].sum())
    o["wm_clipped_steps"] = int(arrays["wm_clipped"][r].any(-1).sum())
    receding = law.action_mismatch(arrays["nominal_actions"][r], arrays["actions"][r], normalization)["mm"]
    open_loop = law.action_mismatch(arrays["nominal_open_loop_actions"][r], arrays["actions"][r], normalization)["mm"]
    o.update(nominal_mismatch_max_mm=float(receding.max()), nominal_mismatch_median_mm=float(np.median(receding)),
             nominal_open_loop_mismatch_max_mm=float(open_loop.max()), approach=spec.get("approach"))
    return o


def run_anchor(env, job, letterbox, cache_block):
    """All rollouts of one anchor -> (npz arrays, manifest row, seconds)."""
    started = time.perf_counter()
    anchor, normalization = job["anchor"], job["normalization"]
    config = law.LawConfig(normalization=normalization)
    loaded = load_source(anchor)
    a = int(anchor["frame"])
    start = a - rp.PREFIX
    grip = float(anchor["grip_command"])
    specs = anchor["sequences"]
    if [s["family"] for s in specs] != list(anchor["families"]) or specs[0]["family"] != "recorded":
        raise ValueError("rollout families differ from the plan definition")
    if cache_block.shape != (rp.PREFIX + 1, IMAGE_SIZE, IMAGE_SIZE, 3) or cache_block.dtype != np.uint8:
        raise ValueError("cache pixel block has the wrong shape")
    ws_low, ws_high = (np.asarray(b, float) for b in env._workspace_bounds)
    if not (np.array_equal(ws_low, law.WORKSPACE_LOW) and np.array_equal(ws_high, law.WORKSPACE_HIGH)):
        raise RuntimeError("scene workspace bounds differ from the law's constants")
    goal_xyz = np.asarray(anchor["goal"]["goal_xyz"], float)
    R = len(specs)
    arrays = init_arrays(R)
    prefix_pixels = np.zeros((rp.PREFIX + 1, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    # per-state diagnostics of the prefix frames a-10..a (captured during rollout 0's exact replay; added 2026-09-28 so the
    # fine-tune cache converter needs only the npz; the pilot datasets rrt_branches_pilot_001/002 predate them)
    prefix_diag = {k: np.zeros((rp.PREFIX + 1, *shape), dtype=dtype) for k, (shape, dtype) in STEP_FIELDS.items()}
    prefix_diag.update({k: np.zeros(rp.PREFIX + 1, bool) for k in CONTACT_FIELDS})
    replay_diff, anchor_images_equal, rec = 0.0, True, {"prefix_raw_png": [], "qpos_diff": []}
    nominal_model = None
    for r, spec in enumerate(specs):
        restore_rebased(env, loaded, start)
        if not np.array_equal(env.hover_goal, loaded["hover_goal"]):
            raise RuntimeError("hover goal was not restored")
        if r == 0:
            raw = env.render(camera=CAMERA)
            prefix_pixels[0] = crb.preprocess(letterbox, raw)
            rec["prefix_raw_png"].append(crb.pixel_diff(raw, crb.recorded_png(loaded, start)))
            for key, value in capture(env).items():
                prefix_diag[key][0] = value
        for j in range(rp.PREFIX):
            frame = start + j
            env.last_transition = None
            env.step(np.asarray(loaded["actions"][frame], dtype=float))
            if env.last_transition is None or not np.array_equal(env.last_transition["action"], loaded["actions"][frame]):
                raise RuntimeError("prefix action was not executed as recorded")
            replay_diff = max(replay_diff, float(np.abs(env.data.qpos - loaded["qpos"][frame + 1]).max()),
                              float(np.abs(env.data.qvel - loaded["qvel"][frame + 1]).max()))
            if r == 0:
                raw = env.render(camera=CAMERA)
                prefix_pixels[j + 1] = crb.preprocess(letterbox, raw)
                rec["prefix_raw_png"].append(crb.pixel_diff(raw, crb.recorded_png(loaded, frame + 1)))
                for key, value in capture(env).items():
                    prefix_diag[key][j + 1] = value
        if not np.array_equal(env.data.qpos, loaded["qpos"][a]) or not np.array_equal(env.data.qvel, loaded["qvel"][a]):
            raise RuntimeError(f"{loaded['dir']} anchor {a}: state after the replayed prefix differs from the record")
        model_name = "grasped" if env.task_report()["grasped"] else "empty"
        if nominal_model not in (None, model_name):
            raise RuntimeError("grasp state at the anchor differs between rollouts")
        nominal_model = model_name
        for key, value in capture(env).items():
            arrays[key][r, 0] = value
        if r == 0:
            arrays["pixels"][r, 0] = prefix_pixels[rp.PREFIX]
            rec["raw_png"] = rec["prefix_raw_png"][rp.PREFIX]
        else:
            arrays["pixels"][r, 0] = crb.preprocess(letterbox, env.render(camera=CAMERA))
            anchor_images_equal &= bool(np.array_equal(arrays["pixels"][r, 0], arrays["pixels"][0, 0]))
        if spec["family"] == "recorded":
            for k in range(H):
                env.last_transition = None
                env.step(np.asarray(loaded["actions"][a + k], dtype=float))
                if env.last_transition is None:
                    raise RuntimeError("Missing executed transition")
                arrays["actions"][r, k] = env.last_transition["action"]
                for key, value in capture(env).items():
                    arrays[key][r, k + 1] = value
                arrays["pixels"][r, k + 1] = crb.preprocess(letterbox, env.render(camera=CAMERA))
                rec["qpos_diff"].append(float(np.abs(env.data.qpos - loaded["qpos"][a + k + 1]).max()))
        else:
            segments = rp.segments_from_json(spec["segments"])
            run_law_rollout(env, arrays, r, segments, grip, config, model_name, np.asarray(loaded["pinch_xyz"][a - 1], float), letterbox)
            if spec.get("approach"):
                arrays["approach_predicted_end_xyz"][r] = spec["approach"]["predicted_end_xyz"]
    frames = slice(a, a + STEPS)
    source_tilt, source_class, source_unsafe = loaded["guard_tilt_deg"][frames], loaded["class_label"][frames].astype(np.int8), loaded["unsafe"][frames]
    cache_frames = [crb.pixel_diff(prefix_pixels[i], cache_block[i]) for i in range(rp.PREFIX + 1)]
    sanity = {"prefix_replay_max_diff": replay_diff, "prefix_replay_exact": bool(replay_diff == 0.0),
              "recorded_max_tilt_diff_deg": float(np.abs(arrays["guard_tilt_deg"][0] - source_tilt).max()),
              "recorded_max_qpos_diff": float(max(rec["qpos_diff"])),
              "recorded_max_com_diff_m": float(np.abs(arrays["object_com_xyz"][0] - loaded["object_com_xyz"][frames]).max()),
              "recorded_class_mismatches": int((arrays["class_label"][0] != source_class).sum()),
              "recorded_unsafe_mismatches": int((arrays["unsafe"][0] != source_unsafe).sum()),
              "recorded_actions_equal": bool(np.array_equal(arrays["actions"][0], loaded["actions"][a:a + H])),
              "anchor_cache": cache_frames[rp.PREFIX], "prefix_cache_identical_frames": int(sum(f["identical"] for f in cache_frames)),
              "prefix_cache_max_abs_diff": int(max(f["max_abs_diff"] for f in cache_frames)), "raw_png": rec["raw_png"],
              "prefix_raw_png_identical_frames": int(sum(f["identical"] for f in rec["prefix_raw_png"])),
              "prefix_raw_png_max_abs_diff": int(max(f["max_abs_diff"] for f in rec["prefix_raw_png"])),
              "anchor_images_equal_across_rollouts": anchor_images_equal,
              "anchor_state_matches_plan": bool(np.array_equal(arrays["pinch_xyz"][0, 0], np.asarray(anchor["state"]["pinch_xyz"])))}
    sanity["recorded_exact"] = bool(sanity["recorded_max_tilt_diff_deg"] == 0 and sanity["recorded_max_qpos_diff"] == 0
                                    and sanity["recorded_class_mismatches"] == 0 and sanity["recorded_unsafe_mismatches"] == 0)
    # frames a-9..a come from env.step exactly like the record (bit-exact); frame a-10 is the restored state (mj_forward
    # after mj_setState), where derived kinematics may differ in the last bits (bench_restore_check.restore uses 1e-9 too)
    checks = {}
    for key, source in (("guard_tilt_deg", loaded["guard_tilt_deg"]), ("object_com_xyz", loaded["object_com_xyz"]),
                        ("object_quat_wxyz", loaded["object_quat_wxyz"]), ("pinch_xyz", loaded["pinch_xyz"]),
                        ("unsafe", loaded["unsafe"]), ("class_label", loaded["class_label"].astype(np.int8))):
        ref = np.asarray(source[start:a + 1])
        checks[key] = {"stepped_frames_exact": bool(np.array_equal(prefix_diag[key][1:], ref[1:])),
                       "restored_frame_max_abs_diff": float(np.abs(prefix_diag[key][0].astype(float) - ref[0].astype(float)).max())}
    sanity["prefix_diagnostics"] = checks
    sanity["prefix_diagnostics_match_source"] = bool(all(c["stepped_frames_exact"] and c["restored_frame_max_abs_diff"] <= 1e-9
                                                         for c in checks.values()))
    if not sanity["prefix_diagnostics_match_source"]:
        raise RuntimeError(f"{loaded['dir']} anchor {a}: prefix diagnostics differ from the record: {checks}")
    for key in prefix_diag:
        if not np.array_equal(prefix_diag[key][rp.PREFIX], arrays[key][0, 0]):
            raise RuntimeError(f"prefix {key} at the anchor differs from rollout 0's anchor state")
    arrays.update(wm_action=law.wm_normalize(arrays["actions"], normalization),
                  nominal_wm_action=law.wm_normalize(arrays["nominal_actions"], normalization), prefix_pixels=prefix_pixels,
                  tracking_error_m=np.stack([rb.tracking_error(arrays["pinch_xyz"][r], arrays["reference_xyz"][r]) for r in range(R)]),
                  action_saturated=rb.saturated_steps(arrays["actions"]),
                  **{f"prefix_{k}": v for k, v in prefix_diag.items()},
                  prefix_actions=np.asarray(loaded["actions"][start:a], dtype=np.float64),
                  prefix_wm_action=law.wm_normalize(loaded["actions"][start:a], normalization),
                  source_guard_tilt_deg=source_tilt, source_class_label=source_class, source_unsafe=source_unsafe.astype(bool),
                  hover_goal_xyz=np.asarray(loaded["hover_goal"], np.float64), goal_xyz=goal_xyz,
                  goal_image_id=np.int64(anchor["goal"]["goal_image_id"]), goal_source=np.array(anchor["goal"]["goal_source"]),
                  anchor_velocity_xyz=np.asarray(loaded["pinch_xyz"][a] - loaded["pinch_xyz"][a - 1], np.float64),
                  families=np.array([s["family"] for s in specs]), anchor_id=np.int64(anchor["id"]), source_frame=np.int64(a),
                  cache_episode_id=np.int64(anchor["source"]["cache_episode_id"]), cache=np.array(rp.SOURCE_CACHE[anchor["source"]["source"]]),
                  set=np.array(anchor["source"]["source"]), group=np.array(anchor["group"]), stratum=np.array(anchor["stratum"]),
                  grip_command=np.float64(grip), nominal_model=np.array(nominal_model))
    if not np.array_equal(arrays["wm_action"], rb.normalize_wm_actions(arrays["actions"], normalization)):
        raise RuntimeError("wm_action differs from the evaluator recipe")
    outcomes = [rollout_outcome(arrays, r, spec, goal_xyz, normalization) for r, spec in enumerate(specs)]
    row = {"id": int(anchor["id"]), "set": anchor["source"]["source"], "group": anchor["group"], "stratum": anchor["stratum"],
           "cache_episode_id": int(anchor["source"]["cache_episode_id"]), "frame": a, "grip_command": grip, "nominal_model": nominal_model,
           "goal_image_id": int(anchor["goal"]["goal_image_id"]), "sanity": sanity, "rollouts": outcomes}
    if job.get("role", "train") != "train":                  # stage-2 dev / confirm: truth pools (the default rows stay unchanged)
        row["decision_points"] = decision_point_row(arrays, [s["family"] for s in specs])
    return arrays, row, time.perf_counter() - started


def decision_point_row(arrays, families):
    """Simulator-truth pools of one anchor at the decision points d = 0, 8 (rrt_pilot_yields.decision_point_instances:
    valid / fresh_valid / positive / action_caused / fresh / calm, subgroups grasped / planner_shape / both, distinct
    rollouts) and the C3 pairs at d = 0 (true max over states 1..20 differs by >= 10 deg, recorded included)."""
    worst = np.asarray(arrays["guard_tilt_deg"], float).max(-1)
    shapes = {d: ry.planner_shape(arrays["segment_index"], arrays["segment_kind"], d) for d in ry.DECISION_POINTS}
    inst = ry.decision_point_instances(worst, [f != "recorded" for f in families], np.asarray(arrays["grasped"], bool), shapes, families)
    counts = ry.count_instances(inst)
    m20 = worst[:, 1:ry.DECISION_WINDOW + 1].max(1)
    return {"by_d": {f"d{d}": counts[d] for d in ry.DECISION_POINTS},
            "subgroups": {name: {f"d{d}": c[d] for d in ry.DECISION_POINTS} for name, keep in ry.SUBGROUPS.items()
                          for c in [ry.count_instances(inst, keep)]},
            "distinct_rollouts": {k: len({i["rollout"] for i in inst if i[k]}) for k in ("valid", "positive", "action_caused", "fresh", "calm")},
            "c3_pairs_d0": int(sum(1 for i in range(len(m20)) for j in range(i + 1, len(m20)) if abs(m20[i] - m20[j]) >= ry.PAIR_GAP_DEG))}


def decision_point_totals(rows):
    """Sums of decision_point_row over the anchors (dev / confirm roles), with the number of anchors behind every pool."""
    rows = [r for r in rows if r.get("decision_points")]
    if not rows:
        return None
    keys = ry.DP_KEYS
    out = {"anchors": len(rows), "by_d": {}, "pooled": {}, "anchors_with": {}, "subgroups": {}, "distinct_rollouts": {}}
    for d in (f"d{x}" for x in ry.DECISION_POINTS):
        out["by_d"][d] = {k: int(sum(r["decision_points"]["by_d"][d][k] for r in rows)) for k in keys}
    out["pooled"] = {k: int(sum(out["by_d"][d][k] for d in out["by_d"])) for k in keys}
    out["anchors_with"] = {k: int(sum(any(r["decision_points"]["by_d"][d][k] for d in out["by_d"]) for r in rows)) for k in keys}
    for name in ry.SUBGROUPS:
        out["subgroups"][name] = {k: int(sum(r["decision_points"]["subgroups"][name][d][k] for r in rows for d in out["by_d"])) for k in keys}
    out["distinct_rollouts"] = {k: int(sum(r["decision_points"]["distinct_rollouts"][k] for r in rows))
                                for k in ("valid", "positive", "action_caused", "fresh", "calm")}
    out["c3_pairs_d0"] = int(sum(r["decision_points"]["c3_pairs_d0"] for r in rows))
    out["definitions"] = "rrt_pilot_yields decision points d = 0, 8 (window 20, valid: tilt at d <= 25 and no state 1..d > 30); simulator truth only"
    return out


def process_job(env, job, letterbox, cache_block):
    arrays, row, seconds = run_anchor(env, job, letterbox, cache_block)
    path = crb.write_anchor(job["output_dir"], row["id"], arrays)
    row.update(file=path.name, sha256=sha256(path), bytes=path.stat().st_size, wall_seconds=seconds, worker_pid=os.getpid(),
               worker_peak_rss_kb=crb.peak_rss_kb())
    return row


def _worker(pipe, stop):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env, cache, letterbox = None, None, None
    try:
        while True:
            job = pipe.recv()
            if job is None or stop.is_set():
                return
            try:
                if env is None:
                    env = ExplorationScene()
                    letterbox, _ = crb.import_letterbox()
                    cache = np.load(job["cache_pixels"], allow_pickle=False)
                row = process_job(env, job, letterbox, np.asarray(cache[job["cache_key"]]))
            except (Exception, KeyboardInterrupt) as error:
                pipe.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
                return
            pipe.send({"status": "complete", "row": row})
    finally:
        if env is not None:
            env.close()
        if cache is not None:
            cache.close()
        pipe.close()


# ----- summary / report -------------------------------------------------------------------------------------------
def _q(values, q):
    return None if not len(values) else float(np.quantile(values, q))


def family_block(sel):
    """Counts of one (stratum, family) cell; horizons: h20 = the planner's check (edge 8 + hold 12), h32 = whole rollout."""
    if not sel:
        return None
    law_rows = [o for o in sel if o["family"] != "recorded"]
    mis = np.array([o["nominal_mismatch_max_mm"] for o in law_rows], float)
    free = np.array([o["nominal_mismatch_max_mm"] for o in law_rows if o["free_space"]], float)
    track = np.array([o["tracking_error_final_m"] for o in law_rows if o["tracking_error_final_m"] is not None], float)
    return {"rollouts": len(sel), "anchor_tilt_ge10": sum(o["anchor_tilt_deg"] >= 10 for o in sel),
            "over30_h20": sum(o["max_tilt_h20"] > 30 for o in sel), "over30_h32": sum(o["max_tilt_h32"] > 30 for o in sel),
            "ge10_h20": sum(o["max_tilt_h20"] >= 10 for o in sel), "below5_h20": sum(o["max_tilt_h20"] < 5 for o in sel),
            "rise10_h20": sum(o["rise_h20"] >= 10 for o in sel), "with_contact": sum(o["contact_steps"] > 0 for o in sel),
            "free_space": sum(o["free_space"] for o in sel),
            "with_class0": sum(o["class0_frames"] > 0 for o in sel), "grasp_lost": sum(o["grasped_start"] and not o["grasped_end"] for o in sel),
            "anti_windup": None if not law_rows else {
                "reference_frozen": sum(o["reference_frozen_steps"] > 0 for o in law_rows),
                "reference_dragged": sum(o["reference_dragged_steps"] > 0 for o in law_rows),
                "integral_clamped": sum(o["integral_clamped_steps"] > 0 for o in law_rows),
                "wm_clipped": sum(o["wm_clipped_steps"] > 0 for o in law_rows), "box_clipped": sum(o["box_clipped_steps"] > 0 for o in law_rows),
                "saturated_action": sum((o["saturated_steps"] or 0) > 0 for o in law_rows)},
            "nominal_mismatch_max_mm": None if not len(mis) else {"median": _q(mis, 0.5), "p90": _q(mis, 0.9), "max": float(mis.max()),
                                                                     "free_space_rollouts": len(free), "free_space_median": _q(free, 0.5),
                                                                     "free_space_p90": _q(free, 0.9), "free_space_max": None if not len(free) else float(free.max())},
            "tracking_final_cm": None if not len(track) else {"median": 100 * _q(track, 0.5), "p90": 100 * _q(track, 0.9), "max": 100 * float(track.max()),
                                                             "gt2cm": int((track > 0.02).sum())},
            "wm_action_absmax": {"median": _q([o["wm_action_absmax"] for o in sel], 0.5), "max": max(o["wm_action_absmax"] for o in sel)}}


def criteria_pools(rows):
    """Sizes of the truth pools behind C1 / C2 / C3 (the criteria themselves are evaluated elsewhere, on a frozen set)."""
    c1, c1_action, c2, c3 = 0, 0, 0, 0
    by_stratum = {}
    for row in rows:
        law_o = [o for o in row["rollouts"] if o["family"] != "recorded"]
        recorded = next(o for o in row["rollouts"] if o["family"] == "recorded")
        pos = [o for o in law_o if o["max_tilt_h20"] > 30]
        caused = [o for o in pos if recorded["max_tilt_h20"] <= 30]
        calm = [o for o in law_o if o["max_tilt_h20"] < 5]
        m = [o["max_tilt_h20"] for o in row["rollouts"]]
        pairs = sum(1 for i in range(len(m)) for j in range(i + 1, len(m)) if abs(m[i] - m[j]) >= 10)
        c1, c1_action, c2, c3 = c1 + len(pos), c1_action + len(caused), c2 + len(calm), c3 + pairs
        s = by_stratum.setdefault(row["stratum"], {"C1_positives": 0, "C1_action_caused": 0, "C2_calm": 0, "C3_pairs": 0})
        s["C1_positives"] += len(pos)
        s["C1_action_caused"] += len(caused)
        s["C2_calm"] += len(calm)
        s["C3_pairs"] += pairs
    return {"C1_positives_h20": c1, "C1_action_caused_h20": c1_action, "C2_calm_h20": c2, "C3_pairs_ge10deg": c3, "by_stratum": by_stratum,
            "definitions": {"C1": "law rollouts whose true worst tilt exceeds 30 deg within 20 steps; action-caused = the recorded continuation "
                                  "from the same anchor stays <= 30 deg within 20 steps",
                            "C2": "law rollouts whose true worst tilt stays < 5 deg for 20 steps",
                            "C3": "same-anchor rollout pairs (recorded included) whose true max tilt within 20 steps differs by >= 10 deg"}}


def approach_summary(rows):
    items = [o for r in rows for o in r["rollouts"] if o["family"] == "approach"]
    if not items:
        return None
    final = np.array([o["green_goal_distance_final_m"] for o in items])
    predicted = np.array([o["approach"]["predicted_goal_distance_m"] for o in items])
    return {"rollouts": len(items), "feasible": sum(o["approach"]["feasible"] for o in items),
            "final_goal_distance_cm": {"median": 100 * float(np.median(final)), "p10": 100 * _q(final, 0.1), "p90": 100 * _q(final, 0.9)},
            "final_in_shell": int(((final >= rp.APPROACH_SHELL_M[0]) & (final <= rp.APPROACH_SHELL_M[1])).sum()),
            "final_within_1p5cm": int((final <= 0.015).sum()),
            "prediction_error_cm": {"median": 100 * float(np.median(np.abs(final - predicted))), "max": 100 * float(np.abs(final - predicted).max())},
            "grasp_lost": sum(o["grasped_start"] and not o["grasped_end"] for o in items),
            "with_hover_conditions": sum(o["hover_conditions_frames"] > 0 for o in items)}


def summarize(rows, wall_seconds, workers, sampled=None):
    s = [r["sanity"] for r in rows]
    n = len(rows)
    sanity = {"anchors": n, "prefix_replay_exact_anchors": sum(x["prefix_replay_exact"] for x in s),
              "recorded_exact_anchors": sum(x["recorded_exact"] for x in s),
              "recorded_max_tilt_diff_deg": max(x["recorded_max_tilt_diff_deg"] for x in s),
              "recorded_max_qpos_diff": max(x["recorded_max_qpos_diff"] for x in s),
              "recorded_max_com_diff_m": max(x["recorded_max_com_diff_m"] for x in s),
              "recorded_class_mismatches": sum(x["recorded_class_mismatches"] for x in s),
              "recorded_unsafe_mismatches": sum(x["recorded_unsafe_mismatches"] for x in s),
              "recorded_actions_equal_anchors": sum(x["recorded_actions_equal"] for x in s),
              "anchor_frame_equals_cache_fraction": sum(x["anchor_cache"]["identical"] for x in s) / n,
              "anchor_frame_cache_max_abs_diff": max(x["anchor_cache"]["max_abs_diff"] for x in s),
              "prefix_frames_equal_cache": f"{sum(x['prefix_cache_identical_frames'] for x in s)}/{n * (rp.PREFIX + 1)}",
              "prefix_frames_cache_max_abs_diff": max(x["prefix_cache_max_abs_diff"] for x in s),
              "anchor_raw_png_identical_fraction": sum(x["raw_png"]["identical"] for x in s) / n,
              "prefix_frames_raw_png_identical": f"{sum(x['prefix_raw_png_identical_frames'] for x in s)}/{n * (rp.PREFIX + 1)}",
              "prefix_frames_raw_png_max_abs_diff": max(x["prefix_raw_png_max_abs_diff"] for x in s),
              "anchor_images_equal_across_rollouts_anchors": sum(x["anchor_images_equal_across_rollouts"] for x in s),
              "anchor_state_matches_plan_anchors": sum(x["anchor_state_matches_plan"] for x in s),
              "prefix_diagnostics_match_source_anchors": sum(x["prefix_diagnostics_match_source"] for x in s),
              "prefix_restored_frame_max_abs_diff": max(max(c["restored_frame_max_abs_diff"] for c in x["prefix_diagnostics"].values()) for x in s)}
    flat = [{**o, "stratum": r["stratum"], "set": r["set"]} for r in rows for o in r["rollouts"]]
    strata = sorted({o["stratum"] for o in flat}, key=rp.STRATA.index)
    families = [f for f in rp.FAMILIES if any(o["family"] == f for o in flat)]
    outcomes = {"overall": {f: family_block([o for o in flat if o["family"] == f]) for f in families}}
    outcomes["by_stratum"] = {st: {f: family_block([o for o in flat if o["stratum"] == st and o["family"] == f]) for f in families} for st in strata}
    counts = {"anchors_by_stratum": {st: sum(r["stratum"] == st for r in rows) for st in strata},
              "anchors_by_group": {g: sum(r["group"] == g for r in rows) for g in sorted({r["group"] for r in rows})},
              "rollouts_by_family": {f: sum(o["family"] == f for o in flat) for f in families},
              "nominal_model": {m: sum(r["nominal_model"] == m for r in rows) for m in ("empty", "grasped")},
              "G_with_goal_image": sum(r["stratum"] == "G" and r["goal_image_id"] >= 0 for r in rows)}
    extra = {} if not any(r.get("decision_points") for r in rows) else {"decision_point_pools": decision_point_totals(rows)}
    return {"sanity": sanity, "counts": counts, "outcomes": outcomes, "criteria_pools": criteria_pools(rows), **extra, "approach": approach_summary(rows),
            "worst_tracking": crb.worst_tracking(rows), "bytes": int(sum(r["bytes"] for r in rows)),
            "bytes_per_anchor": float(np.mean([r["bytes"] for r in rows])),
            "bytes_per_rollout": float(sum(r["bytes"] for r in rows) / max(1, len(flat))),
            "wall_seconds": wall_seconds, "workers": workers, "memory": crb.memory_summary(rows, sampled),
            "anchor_seconds_median": float(np.median([r["wall_seconds"] for r in rows])),
            "seconds_per_rollout": float(sum(r["wall_seconds"] for r in rows) / max(1, len(flat))),
            "anchors_per_minute": 60.0 * n / wall_seconds if wall_seconds else None}


def _fmt(v, spec=".2f"):
    return "–" if v is None else format(v, spec)


def write_report(report_dir, summary, run):
    s, o, c = summary["sanity"], summary["outcomes"], summary["counts"]
    mem = summary["memory"]
    sampled = mem.get("sampled") or {}
    gib = (lambda kb: "–" if kb is None else f"{kb / 1024**2:.2f} GiB")
    role = run.get("role", "train")
    title = {"train": "训练侧起点的仿真 rollout", "dev": "branch-dev 选择集（val DEV 半 + bench_dev 行）的仿真 rollout",
             "confirm": "确认集（val REPORT 半 + bench_test）的仿真 rollout"}[role]
    lines = [f"# RRT 动作律分支采集（第 2 阶段）：{title}\n",
             f"- 数据集：`{run['output']}`；计划：`{run['plan']}`；锚点 {s['anchors']} 个，按层 {json.dumps(c['anchors_by_stratum'], ensure_ascii=False)}，"
             f"按来源/层 {json.dumps(c['anchors_by_group'], ensure_ascii=False)}；rollout {json.dumps(c['rollouts_by_family'], ensure_ascii=False)}。",
             f"- 每条 = 恢复 a−{rp.PREFIX} 帧 + 逐位重放 {rp.PREFIX} 步记录动作 + {H} 步（recorded = 记录动作；其余 = rrt_action_law：2 cm/s 固定速度的 8 步边、停住、防饱和 PI、WM 范围裁剪）。",
             f"- 名义动作模型：{json.dumps(c['nominal_model'], ensure_ascii=False)}（锚点处是否夹着绿块）；G 层链接到目标画面的锚点 {c['G_with_goal_image']} 个。",
             f"- 渲染环境：{json.dumps(run['render_environment'], ensure_ascii=False)}；与来源 manifest 一致：{json.dumps(run['render_environment_matches_source'], ensure_ascii=False)}。",
             f"- 用时 {summary['wall_seconds']:.0f} s（{summary['workers']} 个 worker；每锚点中位 {summary['anchor_seconds_median']:.1f} s，每条 rollout {summary['seconds_per_rollout']:.2f} worker·s）；"
             f"磁盘 {summary['bytes'] / 1024**3:.3f} GiB（每条 rollout {summary['bytes_per_rollout'] / 1e3:.0f} kB）。",
             f"- 内存（实测）：作业 PSS 之和峰值 **{gib(sampled.get('peak_pss_sum_kb'))}**（{sampled.get('samples', 0)} 次采样，最多 {sampled.get('max_processes', 0)} 个进程），"
             f"RSS 相加峰值 {gib(sampled.get('peak_rss_sum_kb'))}，MemAvailable 最低 {gib(sampled.get('min_mem_available_kb'))}；worker ru_maxrss 最大 {gib(mem['worker_peak_rss_kb_max'])}。\n",
             "## 1. 复现与像素检查（本工具自己的检查）\n",
             f"- 前缀重放逐位一致：{s['prefix_replay_exact_anchors']}/{s['anchors']}（不一致直接报错）；recorded 续推完全复现：{s['recorded_exact_anchors']}/{s['anchors']}"
             f"（倾角最大差 {s['recorded_max_tilt_diff_deg']:.3g}°，qpos {s['recorded_max_qpos_diff']:.3g}，COM {s['recorded_max_com_diff_m']:.3g} m，class_label 不一致 {s['recorded_class_mismatches']}，unsafe 不一致 {s['recorded_unsafe_mismatches']}）。",
             f"- 锚点帧与缓存像素逐位相同比例 **{s['anchor_frame_equals_cache_fraction']:.3f}**（最大差 {s['anchor_frame_cache_max_abs_diff']}）；前缀帧 {s['prefix_frames_equal_cache']}（最大差 {s['prefix_frames_cache_max_abs_diff']}）；"
             f"640×480 原图与 PNG：锚点 {s['anchor_raw_png_identical_fraction']:.3f}，前缀 {s['prefix_frames_raw_png_identical']}；各 rollout 锚点图像相同 {s['anchor_images_equal_across_rollouts_anchors']}/{s['anchors']}。",
             f"- 前缀帧的逐帧诊断量（`prefix_*` 数组，给缓存转换用）与记录一致：{s['prefix_diagnostics_match_source_anchors']}/{s['anchors']}（a−9..a 逐位相同；恢复帧 a−10 最大差 {s['prefix_restored_frame_max_abs_diff']:.3g}）。\n",
             "## 2. 结果计数（最差红柱倾角；h20 = 规划器安全检查的 20 步）\n",
             "| 层 | 族 | rollout | 锚点≥10° | >30° h20 | >30° h32 | ≥10° h20 | 增量≥10° h20 | <5° h20 | 红柱接触 | 自由空间 | 丢夹持 | 冻结 | 拖拽 | 积分截断 | WM 裁剪 | 名义失配 最大 中位/p90 (mm) | 自由空间 中位/最大 (mm) | 末跟踪误差 中位/最大 (cm) |",
             "|" + "---|" * 19]
    for name, fams in (("全部", o["overall"]), *o["by_stratum"].items()):
        for fam, b in fams.items():
            if b is None:
                continue
            w, m, t = b["anti_windup"], b["nominal_mismatch_max_mm"], b["tracking_final_cm"]
            lines.append(f"| {name} | {fam} | {b['rollouts']} | {b['anchor_tilt_ge10']} | {b['over30_h20']} | {b['over30_h32']} | {b['ge10_h20']} | {b['rise10_h20']} | "
                         f"{b['below5_h20']} | {b['with_contact']} | {b['free_space']} | {b['grasp_lost']} | {'–' if w is None else w['reference_frozen']} | "
                         f"{'–' if w is None else w['reference_dragged']} | {'–' if w is None else w['integral_clamped']} | {'–' if w is None else w['wm_clipped']} | "
                         f"{'–' if m is None else _fmt(m['median']) + ' / ' + _fmt(m['p90'])} | {'–' if m is None else _fmt(m['free_space_median']) + ' / ' + _fmt(m['free_space_max'])} | "
                         f"{'–' if t is None else _fmt(t['median']) + ' / ' + _fmt(t['max'])} |")
    p = summary["criteria_pools"]
    lines += ["", "## 3. 判据真值池的大小（C1–C3 本身在冻结的确认集上评估，不在这里）\n",
              f"- C1 正例（20 步内 >30°）{p['C1_positives_h20']} 条，其中由动作导致（同锚点 recorded 续推 20 步内不超过 30°）{p['C1_action_caused_h20']} 条；"
              f"C2 安静（20 步内一直 <5°）{p['C2_calm_h20']} 条；C3 同锚点、20 步内最大倾角相差 ≥10° 的配对 {p['C3_pairs_ge10deg']} 对。按层：{json.dumps(p['by_stratum'], ensure_ascii=False)}。"]
    ap = summary["approach"]
    if ap:
        lines += ["", "## 4. 目标附近的 approach 族\n",
                  f"- {ap['rollouts']} 条（{ap['feasible']} 条按名义模型可停在 0.8–2.2 cm 壳层）；实际末位置离目标点：中位 {ap['final_goal_distance_cm']['median']:.2f} cm（p10 {ap['final_goal_distance_cm']['p10']:.2f}，p90 {ap['final_goal_distance_cm']['p90']:.2f}）；"
                  f"落在壳层内 {ap['final_in_shell']} 条，≤1.5 cm {ap['final_within_1p5cm']} 条；预测误差中位 {ap['prediction_error_cm']['median']:.2f} cm（最大 {ap['prediction_error_cm']['max']:.2f}）；丢夹持 {ap['grasp_lost']}；出现悬停条件的 {ap['with_hover_conditions']} 条。"]
    lines += ["", "## 5. 说明（测了什么、没测什么）\n",
              "- 这里只有仿真真值：没有加载 world model。recorded 续推必须逐位复现记录；其余族没有记录可比，只保证动作律按定义执行（动作律本身的检查见 tests/test_rrt_action_law.py 与 runs/rrt_action_law_calib_002）。",
              "- 名义动作（`nominal_actions`）= 规划器在每段开始时用标定的自由空间模型想象的动作（当时的夹爪位置与速度、执行中的参考点/积分状态都给了它）；有接触时它本来就不等于执行动作，所以表里把自由空间的 rollout 单列（整条 rollout 里夹爪不碰环境；夹着绿块时绿块不碰地面 / 红柱；空手时不碰绿块；按每步的 MuJoCo 接触判断，见数组 robot_env_contact / robot_green_contact / green_env_contact）。",
              "- 新状态的图像没有缓存可比，其正确性依赖同一渲染后端与同一 letterbox 函数（锚点与前缀帧的逐位一致说明两者都对）。",
              "- 悬停计时、episode_failed 锁存不在 physics_state 里，从 a−10 帧重新开始；class_label 只依赖当前状态。",
              {"train": "- 起点只来自训练侧（wm_full_001 train、wm_train_bench_001 的训练行）；评估数据没有打开。",
               "dev": "- 角色 dev：起点只来自 wm_full_001 val DEV 半与 wm_train_bench_001 的 bench_dev 行；这是选择集，永不作训练数据。",
               "confirm": "- 角色 confirm：起点只来自 wm_full_001 val REPORT 半与危险基准 bench_test；这里只有采集器自己的仿真真值检查，"
                          "没有加载任何 world model；模型评估在第 3 阶段只做一次（evaluate_rrt_confirmation.py --stage3）。"}[role]]
    dp = summary.get("decision_point_pools")
    if dp:
        lines += ["", "## 6. 决策点上的真值池（d = 0 与 d = 8；只数动作律 rollout；不加载模型）\n",
                  f"- 合计：{json.dumps(dp['pooled'], ensure_ascii=False)}；按 d：{json.dumps(dp['by_d'], ensure_ascii=False)}。",
                  f"- 有该类实例的锚点数：{json.dumps(dp['anchors_with'], ensure_ascii=False)}；不同 rollout：{json.dumps(dp['distinct_rollouts'], ensure_ascii=False)}。",
                  f"- 子组（合计）：{json.dumps(dp['subgroups'], ensure_ascii=False)}；C3 配对（d = 0）{dp['c3_pairs_d0']} 对。"]
    if run.get("stage2_plan"):
        s2 = run["stage2_plan"]
        lines += ["", f"- 冻结方案 `{s2['path']}`（sha256 `{s2['sha256']}`，节 {s2['section']}）；名字检查：{json.dumps(run.get('names'), ensure_ascii=False)}。"]
    (Path(report_dir) / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----- CLI --------------------------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, required=True, help="stage-1 run directory (rrt_branches_plan.py)")
    p.add_argument("--output", type=Path, default=None, help="NEW dataset directory, e.g. data/rrt_branches_001")
    p.add_argument("--report", type=Path, default=None, help="NEW run directory for run.json / summary.json / REPORT_ZH.md")
    p.add_argument("--workers", type=int, choices=range(1, 4), default=1, help="at most 3 simulation workers")
    p.add_argument("--anchor-ids", default=None, help="comma-separated plan anchor ids (pilot); default all")
    p.add_argument("--keep-fraction", type=float, default=1.0, help="keep the first ceil(f*n) anchors of every group")
    p.add_argument("--plan-only", action="store_true", help="print the time / disk / memory estimate and exit (writes nothing)")
    p.add_argument("--role", choices=rp.ROLES, default=None,
                   help="the plan's role (rrt_branches_plan.py --role): omitted = a training plan (a plan without a role, or a 'train' "
                        "plan); dev (branch-dev set) and confirm (random_branches_003) plans need their --role explicitly")
    return p


def effective_role(doc, role):
    """The role a plan is collected with. Without --role only training plans are accepted (a plan written without a role
    - every default-mode plan - or a 'train' plan); a dev / confirm plan needs its own --role explicitly."""
    plan_role = doc.get("role", "train")                     # plans written before the roles existed are training plans
    if role is None:
        if plan_role != "train":
            raise ValueError(f"a {plan_role!r} plan is collected only with --role {plan_role}")
        return "train"
    if plan_role != role:
        raise ValueError(f"a {plan_role!r} plan; it is collected only with --role {plan_role} (given --role {role})")
    return role


def records_stage2(a_role, doc):
    """True when the run must record role / stage2_plan / names (an explicit --role, or a plan that carries a role or a
    stage-2 record). False = the default mode: manifest.json and run.json keep the key set of the collector before the
    roles existed."""
    return a_role is not None or "role" in doc or bool(doc.get("stage2_plan"))


def load_plan(plan_dir, role=None):
    plan_summary = read_json(plan_dir / "summary.json")
    if plan_summary.get("format") != rp.PLAN_FORMAT or (plan_dir / "FAILED.json").exists():
        raise ValueError(f"{plan_dir}: not a successful {rp.PLAN_FORMAT} run")
    files_sha = {name: sha256(plan_dir / name) for name in ("anchors.json", "cache_pixels.npz")}
    if files_sha != plan_summary["files_sha256"]:
        raise ValueError(f"{plan_dir}: anchors.json / cache_pixels.npz changed since the plan was written")
    doc = read_json(plan_dir / "anchors.json")
    if (doc["prefix"], doc["horizon"]) != (rp.PREFIX, rp.HORIZON) or doc["format"] != rp.PLAN_FORMAT:
        raise ValueError("plan constants differ from this code")
    law.LawConfig(normalization=doc["normalization"])
    try:
        role = effective_role(doc, role)
    except ValueError as error:
        raise ValueError(f"{plan_dir}: {error}") from None
    if role != "train" and not doc.get("stage2_plan"):
        raise ValueError(f"{plan_dir}: a {role} plan must carry its stage-2 plan record")
    for anchor in doc["anchors"]:
        if anchor["source"]["source"] not in rp.ROLE_SOURCES[role]:
            raise ValueError(f"anchor {anchor['id']}: not a {'training' if role == 'train' else role}-side source")
        rp.check_anchor_side(role, anchor)
    return doc, files_sha, plan_summary


def stage2_names(doc, plan_dir, output, report):
    """Name and plan.json checks of a plan that carries a stage-2 record: plan / dataset / report directories carry the
    frozen plan's names (or the next free number) and the frozen plan.json is unchanged."""
    s2 = doc["stage2_plan"]
    path = Path(str(s2["path"]))
    if not path.is_file() or sha256(path) != s2["sha256"] or s2["sha256"] != rp.FROZEN_STAGE2_PLAN_SHA256:
        raise ValueError(f"the plan's stage-2 plan {path} is missing, changed or not the frozen plan")
    return {"plan": rp.check_stage2_name(plan_dir, s2["plan_dir"]), "output": rp.check_stage2_name(output, s2["dataset"]),
            "report": rp.check_stage2_name(report, s2["report_dir"])}


def main(argv=None):
    a = build_parser().parse_args(argv)
    plan_dir = a.plan.resolve()
    doc, files_sha, plan_summary = load_plan(plan_dir, a.role)
    role, stage2_mode = effective_role(doc, a.role), records_stage2(a.role, doc)
    ids = None if a.anchor_ids is None else [int(t) for t in a.anchor_ids.split(",") if t.strip()]
    anchors = crb.select_anchors(doc["anchors"], ids, a.keep_fraction)
    if not anchors:
        raise ValueError("no anchors selected")
    workers = min(a.workers, len(anchors))
    estimate = rp.cost_estimate(len(anchors), sum(len(x["sequences"]) for x in anchors), workers)
    if a.plan_only:
        print(json.dumps(native({"plan": str(plan_dir), "role": role, "estimate": estimate}), ensure_ascii=False, indent=1), flush=True)
        return estimate
    if a.output is None or a.report is None:
        raise ValueError("--output and --report are required unless --plan-only")
    output, report = a.output.resolve(), a.report.resolve()
    for path in (output, report):
        if path.exists():
            raise FileExistsError(f"Use a NEW directory: {path}")
    if output == report or output.is_relative_to(report) or report.is_relative_to(output):
        raise ValueError("dataset and report directories must be separate")
    names = None
    if doc.get("stage2_plan"):
        if ids is not None or a.keep_fraction != 1.0:
            raise ValueError("a stage-2 plan is collected whole (no --anchor-ids / --keep-fraction)")
        names = stage2_names(doc, plan_dir, output, report)
    for anchor in anchors:
        source = Path(anchor["source"]["episode_dir"]).resolve().parent
        if output.is_relative_to(source) or report.is_relative_to(source):
            raise ValueError("outputs must be outside the source datasets")
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < estimate["bytes"] + DISK_RESERVE_BYTES:
        raise OSError("not enough free disk space")
    render_env = {k: os.environ.get(k) for k in RENDER_KEYS}
    matches = {}
    for folder in sorted({str(Path(x["source"]["episode_dir"]).parent) for x in anchors}):
        recorded = read_json(Path(folder) / "manifest.json").get("execution", {}).get("render_environment")
        matches[folder] = (recorded == render_env) if recorded else None
    root = Path(__file__).resolve().parent
    config = law.LawConfig(normalization=doc["normalization"])
    run = {"format": FORMAT, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if not (k == "role" and v is None)},
           "plan": str(plan_dir), "output": str(output), "report": str(report),
           "input_sha256": {**{str(plan_dir / k): v for k, v in files_sha.items()}, str(plan_dir / "summary.json"): sha256(plan_dir / "summary.json")},
           "source_note": "per-episode sha256 of trajectory.npz / metadata.json are in anchors.json and re-checked before each anchor",
           "implementation_sha256": {**implementation_hashes(ROOT_FILES, package_dir=root), **implementation_hashes(WM_FILES)},
           "anchors": [x["id"] for x in anchors], "workers": workers, "render_environment": render_env,
           "render_environment_matches_source": matches, "state_spec": STATE_SPEC, "law": config.to_json(),
           "nominal_models": {k: m.to_json() for k, m in law.NOMINAL_MODELS.items()},
           "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                        "pillow": version("pillow"), "ogbench": version("ogbench")},
           "estimate": estimate, "world_model_loaded": False, "test_episodes_opened": False,
           "evaluation_data_opened": role == "confirm"}
    stage2_keys = {"role": role, "stage2_plan": doc.get("stage2_plan"), "names": names} if stage2_mode else {}
    run.update(stage2_keys)                                  # the default mode keeps the run.json key set of the old collector
    if role != "train":
        run["role_note"] = {"dev": "branch-dev selection set: val DEV half + bench_dev rows; never training material",
                            "confirm": "confirmation set: collection-side simulator truth only; no world model reads it before stage 3"}[role]
    output.mkdir(parents=True)
    report.mkdir(parents=True)
    write_new_json(report / "run.json", run)
    jobs = [{"id": x["id"], "anchor": x, "normalization": doc["normalization"], "output_dir": str(output),
             "cache_pixels": str(plan_dir / "cache_pixels.npz"), "cache_key": f"a{x['id']:04d}", "role": role} for x in anchors]
    rows, started = [], time.perf_counter()

    def record(row):
        rows.append(row)
        elapsed = time.perf_counter() - started
        s = row["sanity"]
        eta = elapsed / len(rows) * (len(jobs) - len(rows))
        print(f"[{len(rows)}/{len(jobs)}] anchor {row['id']:04d} {row['group']} ep{row['cache_episode_id']} f{row['frame']} | "
              f"recorded_exact={s['recorded_exact']} cache_px={s['anchor_cache']['identical']}/{s['anchor_cache']['max_abs_diff']} | "
              f"over30@20={sum(o['max_tilt_h20'] > 30 for o in row['rollouts'])} | {row['wall_seconds']:.1f}s | "
              f"{row['bytes'] / 1e6:.1f} MB | elapsed {elapsed:.0f}s eta {eta:.0f}s", flush=True)

    sampler = crb.MemorySampler().start()
    try:
        print(f"{len(jobs)} anchors, {estimate['rollouts']} rollouts with {workers} worker(s); estimate {json.dumps(native(estimate))}", flush=True)
        if workers == 1:
            env = ExplorationScene()
            letterbox, _ = crb.import_letterbox()
            try:
                with np.load(plan_dir / "cache_pixels.npz", allow_pickle=False) as cache:
                    for job in jobs:
                        record(process_job(env, job, letterbox, np.asarray(cache[job["cache_key"]])))
            finally:
                env.close()
        else:
            run_workers(jobs, _worker, workers, record)
        rows.sort(key=lambda r: r["id"])
        sampled = sampler.stop()
        _, how = crb.import_letterbox()
        summary = {"format": FORMAT, **summarize(rows, time.perf_counter() - started, workers, sampled), "letterbox_import": how,
                   "child_thread_limits": CHILD_THREAD_LIMITS if workers > 1 else {}, "estimate": estimate}
        manifest = {"format": FORMAT, "status": "complete", "plan": str(plan_dir), "plan_files_sha256": files_sha, **stage2_keys,
                    "prefix": rp.PREFIX, "horizon": H, "image_size": IMAGE_SIZE, "camera": CAMERA, "law": config.to_json(),
                    "nominal_models": run["nominal_models"],
                    "alignment": "pixels[r, i] -- actions[r, i] --> pixels[r, i+1]; index 0 = anchor frame a",
                    "prefix_alignment": "prefix_pixels[j] = frame a-10+j (j = 10 is the anchor); prefix_actions[j] = recorded action a-10+j",
                    "wm_action": {"columns": [0, 1, 2, 4], "normalization": doc["normalization"],
                                  "rule": "((float32(action)[..., [0,1,2,4]] - float32(mean)) / float32(scale)).astype(float32)",
                                  "identical_to": "wm_training.evaluate_bench_counterfactual.normalize_actions"},
                    "arrays": ARRAY_DOC, "render_environment": render_env, "render_environment_matches_source": matches,
                    "versions": run["versions"], "anchors": [{"plan_anchor": next(x for x in anchors if x["id"] == r["id"]), **r} for r in rows],
                    "summary": summary}
        write_new_json(output / "manifest.json", manifest)
        write_report(report, summary, run)
        write_new_json(report / "summary.json", {**summary, "manifest_sha256": sha256(output / "manifest.json")})
        print(json.dumps(native(summary["sanity"]), ensure_ascii=False), flush=True)
        return summary
    except BaseException as error:
        sampler.stop()
        failure = {"error": f"{type(error).__name__}: {error}", "anchors_done": len(rows),
                   "note": "partial output; npz files of finished anchors are complete but the dataset has no manifest"}
        write_new_json(report / "FAILED.json", failure)
        write_new_json(output / "FAILED.json", failure)
        raise


ARRAY_DOC = {
    "actions": "(R, 32, 5) executed env actions (the recorded ones for rollout 0)",
    "wm_action": "(R, 32, 4) float32 WM input of the executed actions",
    "nominal_actions": "(R, 32, 5) nominal actions the planner would feed the WM, recomputed at each segment start from the then current "
                       "pinch, pinch velocity and tracker carry state (receding horizon); NaN for 'recorded'",
    "nominal_wm_action": "(R, 32, 4) float32 WM input of nominal_actions",
    "nominal_open_loop_actions": "(R, 32, 5) nominal actions of the whole rollout imagined once at the anchor; NaN for 'recorded'",
    "reference_xyz": "(R, 32, 3) tracker reference after each step's update; NaN for 'recorded'",
    "tracking_error_m": "(R, 32) |pinch[r, k+1] - reference_xyz[r, k]|",
    "tracker_lag_m": "(R, 32) |reference_{k-1} - pinch_k| seen by the tracker at step k (the freeze test)",
    "reference_frozen / integral_clamped / reference_dragged / box_clipped / range_clipped": "(R, 32) bool anti-windup and clipping activity",
    "wm_clipped": "(R, 32, 3) bool: dx/dy/dz clipped to B's WM training range",
    "integral_m / pi_output_m": "(R, 32, 3) PI integral and PI output (metres, before clipping)",
    "segment_index": "(R, 32) segment of each step (-1 for 'recorded')",
    "segment_kind / segment_direction / segment_steps / segment_speed_mps / segment_continued": "(R, 4[, 3]) per segment ('' unused)",
    "approach_predicted_end_xyz": "(R, 3) model-predicted green resting point of 'approach' rollouts, NaN otherwise",
    "action_saturated": "(R, 32) bool: a dx/dy/dz on the +-1 (+-5 cm) bound",
    "robot_env_contact / robot_green_contact / green_env_contact": "(R, 33) bool active MuJoCo contacts: robot vs guards/floor, robot vs "
                                                                   "green block, green block vs floor/guards (free space = contact_state doc)",
    "goal_xyz / goal_image_id / goal_source / hover_goal_xyz": "goal point used for labels (goal image achieved COM when linked, else hover goal)",
    "prefix_<STEP_FIELD> / prefix_<contact field>": "(11, ...) per-state diagnostics of frames a-10..a (index 10 = the anchor = "
                                                     "rollout 0's state 0), captured during rollout 0's prefix replay; frames a-9..a equal "
                                                     "the source record bit for bit, frame a-10 within 1e-9 (added 2026-09-28; absent in "
                                                     "data/rrt_branches_pilot_001 / _002)",
    "other": "pixels, prefix_*, source_*, STEP_FIELDS arrays and metadata exactly as random_branches_v2 (RANDOM_BRANCHES.md section 3)",
}


if __name__ == "__main__":
    main()
