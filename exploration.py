"""SIMULATION ONLY: guided stochastic actions, never a real-robot controller.

Uses simulator truth to sample references, not to teleport objects or label
intended outcomes. No learned policy/WM/CBF. Environment task logic is unchanged.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from ogbench.manipspace import lie

from benchmark_plan import BENCH_PHASES
from collect_grasp_trajs import CollectedScene, capture_state
from counterfactual_plan import CF_PHASES

MODES = ("free", "grasp", "carry", "guard")
PHASES = {"SETTLE", "FREE", "APPROACH", "DESCEND", "CLOSE", "LIFT", "RETRACT",
          "CARRY", "HOVER", "GUARD_APPROACH", "GUARD_LOWER", "GUARD_PUSH",
          "GUARD_SEEK", "BRUSH_APPROACH", "BRUSH_LOWER", "BRUSH_SWEEP", "BRUSH_LIFT",
          "CARRY_GUARD", "RELEASE", "EVENT_HOLD", "DONE"}
# Phases of the danger-benchmark subclass (benchmark_exploration.py); the validator
# whitelist must stay equal to this set, so they are registered here as well.
PHASES |= set(BENCH_PHASES)
# Phases of counterfactual branch episodes (collect_counterfactual_branches.py, interaction_revision 4).
PHASES |= set(CF_PHASES)
INTERACTION_REVISION = 2
GOAL_LOW = np.array([0.395, -0.035, 0.15])
GOAL_HIGH = np.array([0.46, 0.035, 0.19])
GOAL_TOLERANCE_M = 0.015
GOAL_SPEED_MPS = 0.02
GOAL_HOLD_SECONDS = 1.0


class ExplorationScene(CollectedScene):
    """Adds separate goal-conditioned annotations at each physics substep."""

    def __init__(self, *args, **kwargs):
        self.hover_goal = np.array([0.425, 0, 0.17])
        self.hover_tolerance_m = GOAL_TOLERANCE_M
        self.hover_speed_limit_mps = GOAL_SPEED_MPS
        self.hover_required_seconds = GOAL_HOLD_SECONDS
        self.hover_report = {}
        self._hover_start = None
        self._contact_peak = np.zeros(2)
        self._green_contact_peak = np.zeros(2)
        super().__init__(*args, **kwargs)

    def initialize_episode(self):
        self._hover_start = None
        self._contact_peak = np.zeros(2)
        self._green_contact_peak = np.zeros(2)
        super().initialize_episode()

    def set_hover_goal(self, goal):
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("hover goal must be three finite coordinates")
        if self.data.time != 0:
            raise ValueError("Cannot change the task goal inside a trajectory")
        self.hover_goal = goal.copy()
        self._hover_start = None
        self.post_step()

    def step(self, action):
        self._contact_peak[:] = 0
        self._green_contact_peak[:] = 0
        return super().step(action)

    def post_step(self):
        super().post_step()
        velocity = np.zeros(6)
        body = self.model.body("object_0")
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                body.id, velocity, 0)
        speed = float(np.linalg.norm(velocity[3:]))
        error = float(np.linalg.norm(self.data.body("object_0").xipos - self.hover_goal))
        task, safety = self.task_report(), self.safety_report()
        in_goal = error <= self.hover_tolerance_m + 1e-9
        slow = speed <= self.hover_speed_limit_mps + 1e-9
        conditions = bool(task["grasped"] and in_goal and slow)
        if conditions and not task["episode_failed"]:
            if self._hover_start is None:
                self._hover_start = float(self.data.time)
            held = float(self.data.time) - self._hover_start
        else:
            self._hover_start, held = None, 0.0
        # Class 0 means instantaneous conditions, NOT one-second mission success.
        label = 1 if safety["unsafe"] else 0 if conditions else 2
        self.hover_report = {
            "hover_goal_xyz": self.hover_goal.copy(), "hover_error_m": error,
            "target_velocity_xyz_mps": velocity[3:].copy(), "target_speed_mps": speed,
            "hover_in_goal": bool(in_goal), "hover_slow": bool(slow),
            "hover_conditions_met": conditions, "hover_hold_seconds": held,
            "hover_success": bool(conditions and not task["episode_failed"] and
                                  held >= self.hover_required_seconds - 1e-9),
            "class_label": label,
        }
        forces = np.zeros(2)
        green_forces = np.zeros(2)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.efc_address < 0:
                continue
            names = [self.model.body(int(self.model.geom_bodyid[g])).name
                     for g in (contact.geom1, contact.geom2)]
            for j in (1, 2):
                if f"object_{j}" not in names:
                    continue
                other = names[1] if names[0] == f"object_{j}" else names[0]
                if other != "object_0" and not other.startswith("ur5e/"):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                forces[j - 1] += max(0.0, float(force[0]))
                if other == "object_0":
                    green_forces[j - 1] += max(0.0, float(force[0]))
        self.guard_contact_force_n = forces
        self._contact_peak = np.maximum(self._contact_peak, forces)
        self.green_guard_contact_force_n = green_forces
        self._green_contact_peak = np.maximum(self._green_contact_peak, green_forces)

    def exploration_state(self):
        state = capture_state(self)
        state.update(self.hover_report)
        state["guard_contact_force_n"] = self.guard_contact_force_n.copy()
        state["guard_contact_peak_last_step_n"] = self._contact_peak.copy()
        state["green_guard_contact_force_n"] = self.green_guard_contact_force_n.copy()
        state["green_guard_contact_peak_last_step_n"] = self._green_contact_peak.copy()
        for name, value in state.items():
            if name != "task_status" and not np.isfinite(value).all():
                raise RuntimeError(f"Non-finite exploration state: {name}")
        return state


def waypoint_reachable(env, xyz):
    """IK endpoint check on separate data, NOT a collision/path guarantee."""
    xyz = np.asarray(xyz, dtype=float)
    low, high = env._workspace_bounds
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or np.any(xyz < low) or np.any(xyz > high):
        return False
    pose = lie.SE3.from_rotation_and_translation(env.fixed_effector_rotation, xyz) @ env._T_pa
    joints = env._ik.solve(pos=pose.translation(), quat=pose.rotation().wxyz,
                           curr_qpos=env.data.qpos[env._arm_joint_ids])
    if not np.isfinite(joints).all():
        return False
    bounds = env.model.jnt_range[env._arm_joint_ids]
    if np.any(joints < bounds[:, 0] - 1e-7) or np.any(joints > bounds[:, 1] + 1e-7):
        return False
    probe = mujoco.MjData(env.model)
    probe.qpos[:] = env.data.qpos
    probe.qpos[env._arm_joint_ids] = joints
    mujoco.mj_forward(env.model, probe)
    rotation = probe.site_xmat[env._pinch_site_id].reshape(3, 3)
    expected = env.fixed_effector_rotation.as_matrix()
    return bool(np.linalg.norm(probe.site_xpos[env._pinch_site_id] - xyz) <= 0.002 and
                np.linalg.norm(rotation - expected) <= 0.04)


class ExplorationController:
    """One finite episode; construction never resets an already sampled scene."""

    def __init__(self, env, rng, mode, max_steps=600, max_attempts=2):
        if mode not in MODES or not isinstance(rng, np.random.Generator):
            raise ValueError("Expected an exploration mode and numpy Generator")
        if type(max_steps) is not int or max_steps <= 0 or type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("step and attempt limits must be positive integers")
        if env.data.time != 0:
            raise ValueError("Exploration must start at a fresh reset")
        self.env, self.rng, self.mode = env, rng, mode
        self.max_steps, self.max_attempts = max_steps, max_attempts
        self.steps, self.attempts, self.carry_segments = 0, 0, 0
        self.events, self.segments = [], []
        self.interactions = []
        self.interaction = None
        self.ever_lifted = False
        self.ever_hover_success = False
        self.lost_since = None
        self.phase, self.stop_reason = "SETTLE", None
        self.reference = self.actual.copy()
        self.waypoint = self.actual.copy()
        self.closed = False
        self.started = 0.0
        self.duration, self.speed = 0.6, 0.06
        self.tight_grasp = mode == "carry" or (mode == "guard" and rng.random() < 0.5)
        self.guarded_carry = mode == "guard" and self.tight_grasp
        self.goal_sampling_attempts = 0
        for _ in range(50):
            self.goal_sampling_attempts += 1
            goal = rng.uniform(GOAL_LOW, GOAL_HIGH)
            if waypoint_reachable(env, goal + [0, 0, 0.005]):
                env.set_hover_goal(goal)
                break
        else:
            raise RuntimeError("No IK-valid hover goal found")
        self._set("SETTLE", self.actual, closed=False, duration=0.6)

    @property
    def actual(self):
        return self.env.data.site_xpos[self.env._pinch_site_id].copy()

    @property
    def finished(self):
        return self.phase == "DONE"

    def _event(self, name, **details):
        self.events.append({"name": name, "time_s": float(self.env.data.time),
                            "step": self.steps, **details})

    def _set(self, phase, xyz=None, *, closed=None, duration=5.0):
        xyz = self.actual if xyz is None else np.asarray(xyz, dtype=float)
        if not waypoint_reachable(self.env, xyz):
            self._event("waypoint_ik_rejected")
            self.finish("waypoint_ik_rejected")
            return
        self.phase, self.waypoint = phase, xyz.copy()
        self.started, self.duration = float(self.env.data.time), float(duration)
        self.speed = float(self.rng.uniform(0.035, 0.07))
        if closed is not None:
            self.closed = bool(closed)
        self.segments.append({"phase": phase, "start_step": self.steps,
                              "time_s": self.started, "waypoint_xyz": xyz.tolist(),
                              "closed": self.closed, "speed_mps": self.speed,
                              "duration_limit_s": self.duration})

    def finish(self, reason):
        self._close_interaction(reason)
        self.phase, self.stop_reason = "DONE", reason

    def _settle_event(self, reason):
        self._close_interaction(reason)
        self._event(reason)
        self.event_reason = reason
        self.reference = self.actual
        self._set("EVENT_HOLD", self.actual, duration=1.0)

    def _hazard_acknowledged(self):
        """Subclasses may keep executing after a latched hazard; the explorer never does."""
        return False

    def _grasp(self):
        if self.attempts >= self.max_attempts:
            self.finish("grasp_attempts_exhausted")
            return
        self.attempts += 1
        target = self.env.data.body("object_0").xipos.copy()
        self.grasp_baseline_height = float(target[2])
        spread = np.array([0.005, 0.0015, 0.002]) if self.tight_grasp else np.array([0.03, 0.006, 0.006])
        self.grasp_offset = self.rng.uniform(-spread, spread)
        self.grasp_xyz = target + [0, 0, 0.005] + self.grasp_offset
        self.grasp_xyz[2] = max(0.025, self.grasp_xyz[2])
        self.close_lead_m = float(self.rng.uniform(0, 0.004))
        self.close_duration = float(self.rng.uniform(0.5, 1.0))
        self._set("APPROACH", [*self.grasp_xyz[:2], 0.19], closed=False)

    def _free(self):
        xyz = self.actual + self.rng.uniform([-0.025, -0.025, -0.025], [0.025, 0.025, 0.025])
        xyz = np.clip(xyz, [0.32, -0.1, 0.09], [0.5, 0.1, 0.25])
        self._set("FREE", xyz, closed=self.rng.random() < 0.35,
                  duration=float(self.rng.uniform(0.2, 0.5)))

    def _guard(self):
        self.guard_index = int(self.rng.integers(1, 3))
        self.guard_center = self.env.data.body(f"object_{self.guard_index}").xipos.copy()
        # Accessible outside faces, not the one-centimetre inner gap. The arm
        # yaw remains fixed; only the XYZ approach direction changes.
        outer = -1.0 if self.guard_index == 1 else 1.0
        normals = np.array([[-1, 0, 0], [1, 0, 0], [0, outer, 0],
                            [-1, outer, 0], [1, outer, 0]], dtype=float)
        normal = normals[int(self.rng.integers(len(normals)))]
        normal /= np.linalg.norm(normal)
        self.push_direction = -normal
        self.guard_height = float(self.rng.choice([0.035, 0.055, 0.08, 0.095]) +
                                  self.rng.uniform(-0.002, 0.002))
        stroke_bounds = [(0.002, 0.005), (0.008, 0.015), (0.02, 0.035)]
        self.push_stroke = float(self.rng.uniform(*stroke_bounds[int(self.rng.integers(3))]))
        xyz = self.guard_center + 0.085 * normal
        xyz[2] = 0.19
        self._start_interaction("empty_push", self.guard_index,
                                direction_xyz=self.push_direction.tolist(),
                                requested_height_m=self.guard_height,
                                requested_post_contact_travel_m=self.push_stroke)
        self._set("GUARD_APPROACH", xyz, closed=True)

    def _start_interaction(self, kind, guard_index, **details):
        self.interaction = {"kind": kind, "guard_index": guard_index,
                            "start_step": self.steps, "start_time_s": float(self.env.data.time),
                            "guard_initial_com_xyz": self.env.data.body(f"object_{guard_index}").xipos.tolist(),
                            "robot_or_green_contact_steps": 0, "green_contact_steps": 0,
                            "first_contact_step": None, "first_green_contact_step": None,
                            "max_contact_force_n": 0.0, "max_green_contact_force_n": 0.0,
                            **details}
        self.interactions.append(self.interaction)

    def _observe_interaction(self):
        """Record measured contacts, never infer them from requested waypoints."""
        row = self.interaction
        if row is None:
            return
        j = row["guard_index"] - 1
        force, green = float(self.env._contact_peak[j]), float(self.env._green_contact_peak[j])
        row["max_contact_force_n"] = max(row["max_contact_force_n"], force)
        row["max_green_contact_force_n"] = max(row["max_green_contact_force_n"], green)
        for value, count, first in ((force, "robot_or_green_contact_steps", "first_contact_step"),
                                     (green, "green_contact_steps", "first_green_contact_step")):
            if value > 1e-3:
                row[count] += 1
                if row[first] is None:
                    row[first] = self.steps
        if "contact_pinch_xyz" in row:
            row["actual_post_contact_travel_m"] = max(0.0, float(np.dot(
                self.actual - row["contact_pinch_xyz"], self.push_direction)))

    def _close_interaction(self, reason):
        row = self.interaction
        if row is None:
            return
        current = self.env.data.body(f"object_{row['guard_index']}").xipos
        row.update(stop_reason=reason, end_step=self.steps,
                   guard_final_com_xyz=current.tolist(),
                   actual_guard_shift_m=float(np.linalg.norm(current - row["guard_initial_com_xyz"])))
        self.interaction = None

    def _begin_push(self):
        self.reference = self.actual
        self.interaction["contact_pinch_xyz"] = self.actual.tolist()
        self.interaction["contact_phase"] = self.phase
        self.interaction["actual_post_contact_travel_m"] = 0.0
        self.interaction["tracking_reserve_m"] = 0.003
        self._event("guard_contact_push_started", guard_index=self.guard_index,
                    requested_travel_m=self.push_stroke)
        # A small command reserve compensates finite PD tracking error. Stop
        # using MEASURED travel below, not arrival at this farther waypoint.
        self._set("GUARD_PUSH", self.actual + self.push_direction * (self.push_stroke + 0.003), duration=3.0)
        self._interaction_speed()

    def _interaction_speed(self):
        if self.finished:
            return
        self.speed = float(self.rng.uniform(0.015, 0.03))
        self.segments[-1]["speed_mps"] = self.speed

    def _brush(self, index=None, direction=None, gap=None, height=None):
        """Lift-clear, approach an outer Y face, lower, sweep in X, then lift.

        Clearance is a nominal box-face gap, NOT a guaranteed physical gap:
        grasp error, rotation, gripper geometry and guard motion still matter.
        The benchmark controller passes index/direction/gap/height explicitly;
        with the defaults the random draws are exactly the original ones.
        """
        index = int(self.rng.integers(1, 3)) if index is None else int(index)
        guard = self.env.data.body(f"object_{index}").xipos.copy()
        outer = -1.0 if index == 1 else 1.0
        direction = float(self.rng.choice([-1.0, 1.0])) if direction is None else float(direction)
        gap = float(self.rng.uniform(-0.006, 0.008)) if gap is None else float(gap)
        height = float(self.rng.uniform(0.075, 0.12)) if height is None else float(height)
        half_span = float(self.rng.uniform(0.045, 0.06))
        half_y = float(self.env.model.geom(f"object_{index}").size[1] +
                       self.env.model.geom("object_0").size[1])
        self.brush_entry_com = guard + [-direction * half_span, outer * (half_y + gap), 0]
        self.brush_entry_com[2] = height
        self.brush_exit_com = self.brush_entry_com + [2 * direction * half_span, 0, 0]
        self._start_interaction("held_green_brush", index,
                                sweep_direction_xyz=[direction, 0.0, 0.0],
                                nominal_face_gap_m=gap, requested_green_com_height_m=height,
                                entry_green_com_xyz=self.brush_entry_com.tolist(),
                                exit_green_com_xyz=self.brush_exit_com.tolist())
        # First raise at the current XY; otherwise a low-to-high diagonal could
        # cross a guard before reaching the clear transfer height.
        self.brush_returning = False
        clear = self.actual.copy()
        clear[2] = 0.19 + self.grip_offset[2]
        self._set("BRUSH_LIFT", clear, closed=True, duration=5.0)

    def _carry(self):
        self.carry_segments += 1
        # First visit the fixed goal to obtain goal-near frames. Later movements
        # include guard interactions and random transport; the goal never changes.
        if self.guarded_carry or (self.carry_segments > 1 and self.rng.random() < 0.4):
            self._brush()
            return
        else:
            desired_com = (self.env.hover_goal.copy() if self.carry_segments % 2 else
                           self.rng.uniform(GOAL_LOW, GOAL_HIGH))
            phase = "CARRY"
        self._set(phase, desired_com + self.grip_offset, closed=True, duration=6.0)

    def _advance_phase(self):
        now = float(self.env.data.time)
        elapsed = now - self.started
        # Measured static IK/PD error on outside approaches can slightly exceed
        # 5 mm. Only these new interaction waypoints use 6 mm; grasp/goal rules
        # and the measured post-contact stroke stop are unchanged.
        arrival_tolerance = 0.006 if self.phase.startswith(("GUARD_", "BRUSH_")) else 0.005
        arrived = np.linalg.norm(self.actual - self.waypoint) <= arrival_tolerance
        timeout = elapsed >= self.duration - 1e-9
        task = self.env.task_report()
        if self.phase == "EVENT_HOLD":
            if timeout:
                self.finish(self.event_reason + "_settled")
            return
        if self.env.task_report()["episode_failed"] and not self._hazard_acknowledged():
            self._settle_event("guard_hazard")
            return
        if self.phase in ("CARRY", "CARRY_GUARD", "HOVER", "BRUSH_LIFT", "BRUSH_APPROACH",
                          "BRUSH_LOWER", "BRUSH_SWEEP"):
            if task["grasped"]:
                self.lost_since = None
            elif self.lost_since is None:
                self.lost_since = now
            elif now - self.lost_since >= 0.15:
                self._settle_event("lost_grasp")
                return
        if self.phase == "SETTLE" and timeout:
            if self.mode == "free":
                self._free()
            elif self.mode == "guard" and not self.guarded_carry:
                self._guard()
            else:
                self._grasp()
        elif self.phase == "FREE" and timeout:
            self._free()
        elif self.phase == "APPROACH" and (arrived or timeout):
            if timeout and not arrived:
                self.finish("approach_timeout")
            else:
                self._set("DESCEND", self.grasp_xyz, closed=False, duration=5.0)
        elif self.phase == "DESCEND":
            if self.actual[2] <= self.grasp_xyz[2] + self.close_lead_m:
                self.closed = True
            if arrived or timeout:
                self._set("CLOSE", self.actual, closed=True, duration=self.close_duration)
        elif self.phase == "CLOSE" and timeout:
            self._set("LIFT", [*self.actual[:2], 0.19], closed=True, duration=5.0)
        elif self.phase == "LIFT" and (arrived or timeout):
            lifted = task["grasped"] and task["target_height_m"] > self.grasp_baseline_height + 0.015
            if lifted:
                self.ever_lifted = True
                self._event("verified_lift")
                self.grip_offset = self.actual - self.env.data.body("object_0").xipos
                self._carry()
            else:
                self._event("missed_grasp")
                self._set("RETRACT", [*self.actual[:2], 0.19], closed=False, duration=1.0)
        elif self.phase == "RETRACT" and timeout:
            self._grasp()
        elif self.phase in ("CARRY", "CARRY_GUARD") and (arrived or timeout):
            self._set("HOVER", self.waypoint, closed=True, duration=float(self.rng.uniform(1.2, 2.0)))
        elif self.phase == "HOVER" and timeout:
            if self.carry_segments >= 2 and self.rng.random() < 0.3:
                self._event("intentional_release")
                self._set("RELEASE", self.actual, closed=False, duration=1.0)
            else:
                self._carry()
        elif self.phase == "RELEASE" and timeout:
            self.finish("release_observed")
        elif self.phase.startswith("BRUSH_"):
            if timeout and not arrived:
                self._settle_event("brush_waypoint_timeout")
            elif arrived:
                if self.phase == "BRUSH_LIFT":
                    if self.brush_returning:
                        self._close_interaction("brush_completed")
                        self._set("CARRY", self.env.hover_goal + self.grip_offset, closed=True, duration=6.0)
                    else:
                        xyz = self.brush_entry_com + self.grip_offset
                        xyz[2] = 0.19 + self.grip_offset[2]
                        self._set("BRUSH_APPROACH", xyz, closed=True, duration=6.0)
                elif self.phase == "BRUSH_APPROACH":
                    self._set("BRUSH_LOWER", self.brush_entry_com + self.grip_offset,
                              closed=True, duration=6.0)
                elif self.phase == "BRUSH_LOWER":
                    self._set("BRUSH_SWEEP", self.brush_exit_com + self.grip_offset,
                              closed=True, duration=8.0)
                    self._interaction_speed()
                else:
                    self.brush_returning = True
                    xyz = self.actual.copy()
                    xyz[2] = 0.19 + self.grip_offset[2]
                    self._set("BRUSH_LIFT", xyz, closed=True, duration=5.0)
        elif self.phase == "GUARD_APPROACH" and (arrived or timeout):
            if timeout and not arrived:
                self.finish("guard_approach_timeout")
            else:
                self._set("GUARD_LOWER", [*self.waypoint[:2], self.guard_height], duration=4.0)
        elif self.phase in ("GUARD_LOWER", "GUARD_SEEK"):
            if self.env._contact_peak[self.guard_index - 1] > 1e-3:
                self._begin_push()
            elif timeout and not arrived:
                self._settle_event("guard_waypoint_timeout")
            elif arrived:
                if self.phase == "GUARD_LOWER":
                    self._set("GUARD_SEEK", self.actual + self.push_direction * 0.08, duration=6.0)
                    self._interaction_speed()
                else:
                    self._settle_event("guard_no_contact")
        elif self.phase == "GUARD_PUSH":
            # Generic 5-mm waypoint tolerance would skip the shortest strokes.
            progress = float(np.dot(self.actual - self.interaction["contact_pinch_xyz"], self.push_direction))
            if progress >= self.push_stroke - 0.0005 or timeout:
                self._settle_event("guard_push_distance" if not timeout else "guard_push_timeout")

    def step(self):
        if self.finished:
            return None
        if self.steps >= self.max_steps:
            self.finish("time_budget")
            return None
        self._advance_phase()
        if self.finished:
            return None
        phase = self.phase
        delta = self.waypoint - self.reference
        length = float(np.linalg.norm(delta))
        limit = self.speed * self.env.control_timestep()
        if length > limit:
            delta *= limit / length
        candidate = self.reference + delta
        lead = candidate - self.actual
        if np.linalg.norm(lead) > 0.02:
            candidate = self.actual + lead * (0.02 / np.linalg.norm(lead))
        self.reference = np.clip(candidate, *self.env._workspace_bounds)
        physical = np.array([*(self.reference - self.actual), 0.0, 1.0 if self.closed else -1.0])
        action = self.env.normalize_action(physical)
        if not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise RuntimeError("Invalid exploration action")
        # Exceptions propagate to the recorder; never fabricate a successor.
        result = self.env.step(action)
        self.steps += 1
        self._observe_interaction()
        self.ever_hover_success |= self.env.hover_report["hover_success"]
        if result[2] or result[3]:
            self.finish("environment_terminated")
        elif self.steps >= self.max_steps:
            self.finish("time_budget")
        return phase

    def report(self):
        return {"phase": self.phase, "stop_reason": self.stop_reason, "mode": self.mode,
                "control_steps": self.steps, "attempts": self.attempts,
                "ever_lifted": self.ever_lifted, "ever_hover_success": self.ever_hover_success,
                "events": self.events, "segments": self.segments,
                "interaction_revision": INTERACTION_REVISION, "interactions": self.interactions,
                "goal_sampling_attempts": self.goal_sampling_attempts,
                "training_data_collected": False, "world_model_loaded": False,
                "uses_simulator_truth": True, "safety": self.env.safety_report()}
