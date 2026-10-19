"""Pure definitions for counterfactual BRANCH datasets (proposal E4-a). numpy only, no simulator imports.

A branch dataset re-simulates recorded fall events of the TRAINING material: the simulator is
restored from the recorded ``physics_state`` a few frames before "restore_frame = onset - lead",
the recorded actions are replayed up to the restore frame (the "prefix", so that windows straddle
the decision point), and from the restore frame on ONE of four action laws runs for ``hold_steps``
control steps with rendered images:

* ``continue``  - the recorded actions as far as the recording goes, then a closed-loop hold;
* ``hold``      - AnchoredHold (closed-loop PI hold of the pinch at the restore frame);
* ``retreat``   - AnchoredHold whose reference moves horizontally away from the tilting guard
                  (bench_restore_check.retreat_direction) for ``move_steps``, then holds;
* ``push_back`` - AnchoredHold whose reference moves horizontally AGAINST the guard's lean
                  (against the horizontal part of the guard's up-vector) for ``move_steps``, then holds.

Which outcome a branch reaches is measured afterwards (wm_training.danger_strata via
counterfactual_split.py), never assumed. Episodes are written in the exploration on-disk format
(schema 3, guided_exploration_v1) with interaction_revision 4 and a ``counterfactual`` record; the
validator (exploration_validation.validate_counterfactual) checks the record against the arrays.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

CF_FORMAT = "counterfactual_branches_v1"
CF_INTERACTION_REVISION = 4
CF_CONTROLLER_TYPE = "counterfactual_branch_simulator_truth"
BRANCHES = ("continue", "hold", "retreat", "push_back")
BRANCH_PHASE = {"continue": "CF_CONTINUE", "hold": "CF_HOLD", "retreat": "CF_RETREAT", "push_back": "CF_PUSHBACK"}
CF_PHASES = tuple(BRANCH_PHASE[b] for b in BRANCHES)          # recorder stores action_phase as <U16
HOLD_PHASE = BRANCH_PHASE["hold"]
BRANCH_MODE = {b: f"cf_{b}" for b in BRANCHES}                  # manifest/metadata "mode" of a branch episode
CF_MODES = tuple(sorted(BRANCH_MODE.values()))
PARTS = ("test", "dev", "all")
DEFAULT_LEADS = (2, 4, 6, 8, 10, 15, 20)
DEFAULT_HOLD_STEPS = 200
DEFAULT_PREFIX = 10          # recorded frames replayed before the restore frame (>= MIN_PREFIX)
MIN_PREFIX = 3               # events whose restore frame is < 3 are skipped (no 3-frame history before the branch point)
DEFAULT_MOVE_STEPS = 20      # retreat / push_back: 1 s of motion at MOVE_SPEED, then hold
DEFAULT_MOVE_SPEED_MPS = 0.04
LEAN_MIN_HORIZONTAL = 0.05   # |u_xy| of the guard's up-vector below this (about 2.9 deg) -> use the onset lean instead
# How the horizontal direction of a moving branch was chosen: push_back = against the guard's lean at the restore
# frame ("restore") or, if upright there, at the onset ("onset"); retreat = away from the guard COM ("guard_com"),
# falling back to bench_restore_check.retreat_direction's rules (against the lean, then toward the base).
LEAN_SOURCES = ("restore", "onset", "guard_com", "base_minus_x")
REPLAY_TOLERANCE = 1e-9      # replaying recorded actions from a restored frame is expected to be bit-exact (BENCH_RESTORE.md)
PLAN_KEYS = ("source_dataset", "source_dataset_path", "source_episode_id", "source_path", "source_manifest_sha256",
             "source_npz_sha256", "source_stratum", "source_part", "source_frames", "onset_frame", "lead", "restore_frame",
             "prefix_frames", "branch", "phase", "hold_steps", "move_steps", "move_speed_mps", "grip_policy")
RESULT_KEYS = ("active_steps", "grip_command", "direction_xy", "lean_source", "worst_guard_index",
               "replay_max_qpos_diff", "replay_exact", "max_tilt_deg", "reached_unsafe", "first_unsafe_step",
               "final_tilt_deg", "pinch_drift_max_mm", "time_rebased_to_zero")
CF_RECORD_KEYS = PLAN_KEYS + RESULT_KEYS


def parse_branches(text):
    names = [t.strip() for t in str(text).split(",") if t.strip()]
    if not names or len(set(names)) != len(names) or any(n not in BRANCHES for n in names):
        raise ValueError(f"branches must be a comma-separated subset of {BRANCHES} without repeats")
    return [b for b in BRANCHES if b in names]         # canonical order


def restore_frame_for(onset_frame, lead):
    """Frame at which the branch starts, or None when it would leave fewer than MIN_PREFIX recorded frames before it."""
    if type(onset_frame) is not int or type(lead) is not int or lead <= 0:
        raise ValueError("onset_frame and lead must be integers, lead positive")
    frame = onset_frame - lead
    return frame if frame >= MIN_PREFIX else None


def plan_jobs(events, leads, branches, hold_steps, prefix, move_steps, move_speed_mps):
    """Deterministic job list: events in the given order x leads x branches; skipped (event, lead) pairs are listed."""
    leads = [int(k) for k in leads]
    if not leads or len(set(leads)) != len(leads) or any(k <= 0 for k in leads):
        raise ValueError("leads must be distinct positive integers")
    branches = list(branches)
    if not branches or any(b not in BRANCHES for b in branches):
        raise ValueError("Unknown branch")
    if type(hold_steps) is not int or hold_steps <= 0 or type(prefix) is not int or prefix < MIN_PREFIX:
        raise ValueError(f"hold_steps must be positive and prefix >= {MIN_PREFIX}")
    if type(move_steps) is not int or move_steps < 0 or not np.isfinite(move_speed_mps) or move_speed_mps <= 0:
        raise ValueError("move_steps must be a non-negative integer and move_speed_mps positive")
    jobs, skipped = [], []
    for event in events:
        onset, frames = int(event["onset_frame"]), int(event["frames"])
        if not 0 < onset < frames:
            raise ValueError(f"Event {event.get('dataset')}/{event.get('episode_id')} has an invalid onset frame")
        for lead in leads:
            frame = restore_frame_for(onset, lead)
            if frame is None:
                skipped.append({"dataset": event["dataset"], "episode_id": event["episode_id"], "lead": lead,
                                "onset_frame": onset, "reason": f"restore frame {onset - lead} < {MIN_PREFIX}"})
                continue
            for branch in branches:
                jobs.append({"id": len(jobs), "dataset": int(event["dataset"]), "episode_id": int(event["episode_id"]),
                             "source_path": event["path"], "episode_dir": event["episode_dir"], "stratum": event["stratum"],
                             "part": event["part"], "onset_frame": onset, "frames": frames, "lead": int(lead),
                             "restore_frame": frame, "prefix": int(min(prefix, frame)), "branch": branch,
                             "hold_steps": hold_steps, "move_steps": int(min(move_steps, hold_steps)),
                             "move_speed_mps": float(move_speed_mps)})
    if not jobs:
        raise ValueError("No branch jobs: every event/lead pair was skipped")
    return jobs, skipped


def jobs_digest(jobs):
    """sha256 over the job specification (what will be simulated), like benchmark_plan.schedule_digest."""
    keys = ("id", "dataset", "episode_id", "source_path", "onset_frame", "lead", "restore_frame", "prefix", "branch",
            "hold_steps", "move_steps", "move_speed_mps")
    body = [{k: job[k] for k in keys} for job in jobs]
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def push_back_direction(up_restore_xy, up_onset_xy):
    """Horizontal unit vector AGAINST the guard's lean: -u_xy of the restore frame, or of the onset frame when
    the guard is (almost) upright at the restore frame. Returns (direction, source)."""
    lean = np.asarray(up_restore_xy, dtype=float)
    source = "restore"
    if lean.shape != (2,) or not np.isfinite(lean).all():
        raise ValueError("up_restore_xy must be two finite values")
    if np.linalg.norm(lean) < LEAN_MIN_HORIZONTAL:
        lean = np.asarray(up_onset_xy, dtype=float)
        source = "onset"
        if lean.shape != (2,) or not np.isfinite(lean).all() or np.linalg.norm(lean) < 1e-6:
            raise ValueError("The guard has no horizontal lean at the onset frame either")
    return -lean / np.linalg.norm(lean), source


def expected_phases(record):
    """Phase labels of the branch part of an episode: BRANCH_PHASE for the active steps, then CF_HOLD."""
    active, hold_steps = int(record["active_steps"]), int(record["hold_steps"])
    return [BRANCH_PHASE[record["branch"]]] * active + [HOLD_PHASE] * (hold_steps - active)


def validate_branch_record(record, plan=None):
    """Structural checks of an outcome.counterfactual record (and its agreement with the controller plan)."""
    if not isinstance(record, dict) or set(CF_RECORD_KEYS) - set(record):
        raise ValueError(f"counterfactual record needs the keys {sorted(set(CF_RECORD_KEYS) - set(record or {}))}")
    if plan is not None:
        if not isinstance(plan, dict) or set(plan) != set(PLAN_KEYS) or any(record[k] != plan[k] for k in PLAN_KEYS):
            raise ValueError("outcome.counterfactual disagrees with controller.counterfactual")
    branch = record["branch"]
    if branch not in BRANCHES or record["phase"] != BRANCH_PHASE[branch]:
        raise ValueError("Unknown branch or wrong phase name")
    ints = {k: record[k] for k in ("source_dataset", "source_episode_id", "source_frames", "onset_frame", "lead",
                                   "restore_frame", "prefix_frames", "hold_steps", "move_steps", "active_steps",
                                   "worst_guard_index")}
    if any(type(v) is not int for v in ints.values()):
        raise ValueError("counterfactual record integers must be plain ints")
    if (ints["lead"] <= 0 or ints["restore_frame"] != ints["onset_frame"] - ints["lead"] or ints["restore_frame"] < MIN_PREFIX
            or not MIN_PREFIX <= ints["prefix_frames"] <= ints["restore_frame"] or ints["hold_steps"] <= 0
            or not 0 <= ints["move_steps"] <= ints["hold_steps"] or not 0 <= ints["active_steps"] <= ints["hold_steps"]
            or ints["worst_guard_index"] not in (1, 2) or not 0 < ints["onset_frame"] < ints["source_frames"]):
        raise ValueError("counterfactual record frame/step bookkeeping is inconsistent")
    if branch == "hold" and ints["active_steps"] != ints["hold_steps"]:
        raise ValueError("hold branches are active for every step")
    if branch in ("retreat", "push_back") and ints["active_steps"] != ints["move_steps"]:
        raise ValueError("retreat/push_back are active for move_steps steps")
    if branch == "continue" and ints["active_steps"] != min(ints["hold_steps"], ints["source_frames"] - 1 - ints["restore_frame"]):
        raise ValueError("continue replays every remaining recorded action (at most hold_steps)")
    direction = record["direction_xy"]
    if branch in ("retreat", "push_back"):
        d = np.asarray(direction, dtype=float)
        if d.shape != (2,) or not np.isfinite(d).all() or abs(np.linalg.norm(d) - 1) > 1e-6:
            raise ValueError("retreat/push_back need a horizontal unit direction")
        if record["lean_source"] not in LEAN_SOURCES:
            raise ValueError("Unknown lean_source")
    elif direction is not None or record["lean_source"] is not None:
        raise ValueError("continue/hold have no direction")
    if record["grip_policy"] != "keep" or type(record["grip_command"]) is not float or abs(record["grip_command"]) > 1:
        raise ValueError("Unexpected grip bookkeeping")
    if type(record["reached_unsafe"]) is not bool or type(record["replay_exact"]) is not bool or record["time_rebased_to_zero"] is not True:
        raise ValueError("Boolean fields of the counterfactual record are invalid")
    first = record["first_unsafe_step"]
    if (first is None) == record["reached_unsafe"] or (first is not None and (type(first) is not int or not 0 < first <= ints["hold_steps"])):
        raise ValueError("first_unsafe_step disagrees with reached_unsafe")
    for key in ("max_tilt_deg", "replay_max_qpos_diff", "pinch_drift_max_mm", "move_speed_mps"):
        v = record[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError(f"{key} must be a finite non-negative number")
    final = np.asarray(record["final_tilt_deg"], dtype=float)
    if final.shape != (2,) or not np.isfinite(final).all():
        raise ValueError("final_tilt_deg must hold two finite angles")
    return record


def validate_counterfactual(metadata, arrays):
    """Array-level checks of one branch episode (called by exploration_validation for interaction_revision 4).

    The prefix frames are the recorded pre-onset history (all safe), the branch part follows the
    phase pattern of the record, and the recorded outcome numbers agree with the saved arrays.
    """
    controller = metadata.get("controller", {})
    if controller.get("type") != CF_CONTROLLER_TYPE or controller.get("interaction_revision") != CF_INTERACTION_REVISION:
        raise ValueError("counterfactual episodes need the counterfactual controller type and interaction_revision 4")
    plan, record = controller.get("counterfactual"), metadata["outcome"].get("counterfactual")
    if not isinstance(plan, dict) or not isinstance(record, dict):
        raise ValueError("counterfactual episodes need controller.counterfactual and outcome.counterfactual")
    validate_branch_record(record, plan)
    if controller.get("mode") != BRANCH_MODE[record["branch"]] or metadata["outcome"].get("mode") != BRANCH_MODE[record["branch"]]:
        raise ValueError("counterfactual mode disagrees with the branch")
    n, prefix, hold = int(metadata["num_states"]), record["prefix_frames"], record["hold_steps"]
    if n != prefix + hold + 1:
        raise ValueError("num_states must be prefix_frames + hold_steps + 1")
    phases = [str(p) for p in arrays["action_phase"]]
    if len(phases) != prefix + hold or any(p in CF_PHASES for p in phases[:prefix]) or phases[prefix:] != expected_phases(record):
        raise ValueError("action_phase does not follow the branch pattern (recorded prefix, active phase, CF_HOLD)")
    unsafe, tilt = np.asarray(arrays["unsafe"], dtype=bool), np.asarray(arrays["guard_tilt_deg"], dtype=float)
    if unsafe[:prefix + 1].any():
        raise ValueError("prefix frames and the restore frame must be safe (they precede the onset)")
    hits = np.flatnonzero(unsafe[prefix:])
    first = int(hits[0]) if len(hits) else None
    if record["reached_unsafe"] != bool(len(hits)) or record["first_unsafe_step"] != first:
        raise ValueError("reached_unsafe/first_unsafe_step disagree with the unsafe array")
    if abs(float(tilt[prefix:].max()) - float(record["max_tilt_deg"])) > 1e-6 or not np.allclose(tilt[-1], record["final_tilt_deg"], atol=1e-6, rtol=0):
        raise ValueError("max/final tilt of the record disagree with guard_tilt_deg")
    if abs(float(arrays["time"][0])) > 1e-9:
        raise ValueError("branch episodes are re-based to time zero at their first frame")
    return {"branch": record["branch"], "lead": record["lead"], "reached_unsafe": record["reached_unsafe"]}
