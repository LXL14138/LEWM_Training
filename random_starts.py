"""Bounded, reproducible reset-pose sampling for the fixed three-block task.

Only arm_start changes; object layout, fixed yaw, physics and safety limits do
not. Validation is an instantaneous simulator check, not a path planner or a
guarantee that the later scripted approach is collision-free. No physics step
is executed here. Use validate_start again after a controller performs reset.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np

from scene_config import ARM_START_MIN_Z_M

DEFAULT_START_LOW = (0.28, -0.02, 0.24)
DEFAULT_START_HIGH = (0.32, 0.02, 0.28)
SUPPORTED_START_LOW = np.array([0.25, -0.35, ARM_START_MIN_Z_M])
SUPPORTED_START_HIGH = np.array([0.60, 0.35, 0.35])
POSITION_TOLERANCE_M = 0.002
ROTATION_TOLERANCE_RAD = 0.03
JOINT_LIMIT_NUMERICAL_TOL = 1e-7
CONTACT_DISTANCE_NUMERICAL_TOL_M = 1e-7

# These are the explicit internal exclusions in OGBench's Robotiq 2F85 MJCF.
# Usually MuJoCo already omits them; spelling them out avoids treating normal
# same-side linkage interfaces as self-collision if a contact pair is exposed.
# This is NOT a blanket exemption for all contacts within the gripper: opposite
# fingers or a finger touching an arm link still count as self-collision.
_ALLOWED_LINKAGE_PAIRS = {
    frozenset(("base", "left_driver")),
    frozenset(("base", "right_driver")),
    frozenset(("base", "left_spring_link")),
    frozenset(("base", "right_spring_link")),
    frozenset(("right_coupler", "right_follower")),
    frozenset(("left_coupler", "left_follower")),
}


class StartSamplingError(RuntimeError):
    """No valid reset found; complete rejected candidates remain inspectable."""

    def __init__(self, rejected, low, high):
        self.rejected = rejected
        self.attempts = len(rejected)
        self.low = np.asarray(low).tolist()
        self.high = np.asarray(high).tolist()
        super().__init__(
            f"No valid robot start after {self.attempts} attempts. "
            + json.dumps({"low": self.low, "high": self.high, "rejected": rejected})
        )


def _contact_kind(model, geom1, geom2):
    """Classify a near/touching pair; all other external contacts fail closed."""
    bodies = [int(model.geom_bodyid[g]) for g in (geom1, geom2)]
    names = [model.body(body).name for body in bodies]
    geoms = [model.geom(g).name for g in (geom1, geom2)]
    robot = [name.startswith("ur5e/") for name in names]
    if all(robot):
        if model.body_weldid[bodies[0]] == model.body_weldid[bodies[1]]:
            return "allowed_rigid_robot_internal"
        if all(name.startswith("ur5e/robotiq/") for name in names):
            suffixes = frozenset(name.rsplit("/", 1)[-1] for name in names)
            if suffixes in _ALLOWED_LINKAGE_PAIRS:
                return "allowed_gripper_linkage_internal"
        return "robot_self_collision"
    if any(robot):
        other = 1 if robot[0] else 0
        if names[other] in ("object_0", "object_1", "object_2"):
            return "robot_block_contact"
        if geoms[other] == "floor":
            return "robot_floor_contact"
        return "robot_other_contact"
    if "floor" in geoms and any(name in ("object_0", "object_1", "object_2") for name in names):
        return "allowed_block_floor"
    return "other_initial_contact"


def validate_start(env, requested_start_xyz):
    """Read the refreshed reset state without advancing or mutating simulation.

    Contact tests intentionally do not require a positive constraint force:
    overlapping objects at reset can still have inactive/zero-force contacts.
    MuJoCo's own collision masks/exclusions remain in force, so this cannot
    detect geometry intersections suppressed by the underlying robot model.
    """
    reasons = []
    try:
        requested = np.asarray(requested_start_xyz, dtype=float)
    except (TypeError, ValueError):
        requested = np.array([])
    requested_ok = requested.shape == (3,) and np.isfinite(requested).all()
    if not requested_ok:
        reasons.append("requested_start_must_have_three_finite_values")
    elif np.any(requested < SUPPORTED_START_LOW) or np.any(requested > SUPPORTED_START_HIGH):
        reasons.append("requested_start_outside_supported_workspace")

    actual = np.asarray(env.data.site_xpos[env._pinch_site_id], dtype=float)
    rotation = np.asarray(env.data.site_xmat[env._pinch_site_id], dtype=float).reshape(3, 3)
    finite = all(np.isfinite(value).all() for value in (
        env.data.qpos, env.data.qvel, env.data.ctrl, actual, rotation,
    )) and np.isfinite(env.data.time)
    if not finite:
        reasons.append("nonfinite_simulator_state")
    if float(env.data.time) != 0.0:
        reasons.append("not_a_reset_state_simulation_time_is_nonzero")

    position_error = None
    rotation_error = None
    if finite and requested_ok:
        position_error = float(np.linalg.norm(actual - requested))
        expected = env.fixed_effector_rotation.as_matrix()
        cosine = float(np.clip((np.trace(expected.T @ rotation) - 1.0) / 2.0, -1.0, 1.0))
        rotation_error = float(np.arccos(cosine))
        if position_error > POSITION_TOLERANCE_M:
            reasons.append("ik_position_error_exceeds_2mm")
        if rotation_error > ROTATION_TOLERANCE_RAD:
            reasons.append("ik_rotation_error_exceeds_0.03rad")
        # Separate the requested bounds from the achieved Cartesian pose.
        if np.any(actual < SUPPORTED_START_LOW - POSITION_TOLERANCE_M) or np.any(
            actual > SUPPORTED_START_HIGH + POSITION_TOLERANCE_M
        ):
            reasons.append("actual_start_outside_supported_workspace")

    joint_violations = []
    for joint_id in range(env.model.njnt):
        if not env.model.jnt_limited[joint_id]:
            continue
        kind = env.model.jnt_type[joint_id]
        if kind not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue  # This scene's robot contains only scalar hinge joints.
        address = int(env.model.jnt_qposadr[joint_id])
        value = float(env.data.qpos[address])
        low, high = env.model.jnt_range[joint_id]
        if not np.isfinite(value) or value < low - JOINT_LIMIT_NUMERICAL_TOL or value > high + JOINT_LIMIT_NUMERICAL_TOL:
            joint_violations.append({
                "joint": env.model.joint(joint_id).name,
                "position": value if np.isfinite(value) else None,
                "range": [float(low), float(high)],
            })
    if joint_violations:
        reasons.append("joint_limit_violation")

    safety = None
    if finite:
        try:
            safety = env.safety_report()
            if safety["unsafe"]:
                reasons.append("guards_unsafe_at_start")
        except (RuntimeError, ValueError) as exc:
            reasons.append("safety_check_error: " + str(exc))

    contacts = []
    for contact in env.data.contact[:env.data.ncon]:
        if not np.isfinite(contact.dist):
            reasons.append("nonfinite_contact_distance")
            continue
        # A positive-margin broad-phase candidate is not touching yet.
        if contact.dist > CONTACT_DISTANCE_NUMERICAL_TOL_M:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        kind = _contact_kind(env.model, g1, g2)
        allowed = kind.startswith("allowed_")
        contacts.append({
            "geom1": env.model.geom(g1).name,
            "geom2": env.model.geom(g2).name,
            "body1": env.model.body(int(env.model.geom_bodyid[g1])).name,
            "body2": env.model.body(int(env.model.geom_bodyid[g2])).name,
            "distance_m": float(contact.dist), "kind": kind, "allowed": allowed,
        })
        if not allowed and kind not in reasons:
            reasons.append(kind)

    return {
        "valid": not reasons, "reasons": reasons,
        "requested_start_xyz": requested.tolist() if requested_ok else None,
        "actual_start_xyz": actual.tolist() if np.isfinite(actual).all() else None,
        "sim_time_s": float(env.data.time) if np.isfinite(env.data.time) else None,
        "position_error_m": position_error, "rotation_error_rad": rotation_error,
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "rotation_tolerance_rad": ROTATION_TOLERANCE_RAD,
        "joint_limit_violations": joint_violations, "safety": safety,
        "contacts": contacts,
        "contact_scope": "reset contacts reported by the model collision masks",
        "path_reachability_guaranteed": False,
    }


def sample_valid_start(env, rng, max_attempts=50, low=DEFAULT_START_LOW, high=DEFAULT_START_HIGH):
    """Sample, reset and validate; leave env at the first accepted reset state.

    Each candidate consumes one rng.uniform draw; env.reset always uses seed 0
    to keep the objects/scene unchanged. On exhaustion, env remains at its last
    attempted reset and callers must not execute that invalid state.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator (e.g. default_rng(seed))")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, (int, np.integer)) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("low and high must each contain three finite coordinates")
    if np.any(low > high):
        raise ValueError("low must not exceed high on any axis")
    if np.any(low < SUPPORTED_START_LOW) or np.any(high > SUPPORTED_START_HIGH):
        raise ValueError("sampling bounds must lie inside the supported starting workspace")
    rejected = []
    for attempt in range(1, int(max_attempts) + 1):
        candidate = rng.uniform(low, high)
        env.arm_start = candidate.copy()
        try:
            env.reset(seed=0)
            validity = validate_start(env, candidate)
        except (RuntimeError, ValueError) as exc:
            validity = {"valid": False, "reasons": ["reset_or_validation_error: " + str(exc)]}
        if validity["valid"]:
            return {
                "requested_start_xyz": candidate.tolist(),
                "actual_start_xyz": validity["actual_start_xyz"],
                "attempts": attempt, "rejected": rejected, "validity": validity,
            }
        rejected.append({"start": candidate.tolist(), "reasons": validity["reasons"], "validity": validity})
    raise StartSamplingError(rejected, low, high)
