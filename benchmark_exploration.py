"""SIMULATION ONLY: scripted danger-benchmark attempts on top of the guided explorer.

Consumes one intent from a benchmark_plan schedule per episode. Everything goes
through env.step with the exploration action semantics ([dx,dy,dz,0,grip],
fixed yaw); objects are never teleported. The controller only ATTEMPTS a
stratum; which stratum the episode reaches is measured afterwards with
wm_training.danger_strata. The four original exploration modes are untouched.
"""
from __future__ import annotations

import numpy as np

from benchmark_plan import (
    BENCH_INTERACTION_REVISION, BENCH_PHASES, NEEDS_GRASP, RECIPE_MODE, TIP_ANGLE_DEG,
    lean_vector, stop_reached, validate_intent,
)
from exploration import ExplorationController

TRANSFER_Z = 0.19                 # clear transfer height for the pinch centre
PARK_XY = np.array([0.34, 0.0])   # where the green block is set down before parked-green pushes
GUARD_TOP_Z = 0.10
PRESS_SEEK_BELOW_TOP_M = 0.02      # descend at most this far below the guard top looking for contact
PRESS_FORWARD_M, PRESS_DOWN_M = 0.10, 0.05   # shallow drag after contact; pre-check 1 showed deep presses pin the guard
# Pre-check 2: dragging toward the robot base (-x) under ~200 N of vertical load stalled the arm (friction);
# +x and outward presses worked with the 5 cm slope, so only -x gets a lighter one.
PRESS_DOWN_BY_LEAN = {"-x": 0.015}
PRESS_SLOWDOWN_DEG, PRESS_SLOW_SPEED_MPS = 8.0, 0.008   # creep toward the tilt target to limit overshoot
GREEN_HALF_HEIGHT_M = 0.05
GRIPPER_HALF_X_M, GRIPPER_HALF_Y_M = 0.03, 0.015   # closed-gripper half extents around the pinch site (fingers open along X)
RETREAT_DISTANCE_M = 0.12
ARRIVAL_TOLERANCE_M = 0.006
GRASP_LOSS_S = 0.15
GRASP_BUDGET_MIN_S = 24.0          # grasp/park prologues need 14-20 s before first contact
SLOW_SPEED_MPS = 0.015             # slowest interaction speed; sizes push/hit phase durations


class BenchmarkController(ExplorationController):
    """One finite benchmark attempt. Base phases (SETTLE, grasp, brush) are reused unchanged."""

    interaction_revision = BENCH_INTERACTION_REVISION

    def __init__(self, env, rng, mode, intent, max_steps=240, max_attempts=2):
        intent = validate_intent(dict(intent))
        if mode != RECIPE_MODE[intent["recipe"]]:
            raise ValueError("mode does not match the intent's recipe")
        self.intent, self.recipe = intent, intent["recipe"]
        needs_grasp = self.recipe in NEEDS_GRASP or intent["park_green"]
        super().__init__(env, rng, "carry" if needs_grasp else "free", max_steps=max_steps, max_attempts=max_attempts)
        self.mode = mode
        self.guard_index = intent["guard_index"]
        self.lean = None if intent["lean"] is None else lean_vector(intent["lean"], self.guard_index)
        self.after_event = intent["after_event"]
        self.stop = None if intent["stop"] is None else tuple(intent["stop"])
        self.params = self._sample_params()
        self.stop_rule = None if self.stop is None else (self.stop[0], self.params["stop_target"], self.params["stop_target"])
        self.end_action = "retreat" if self.recipe in ("touch",) else "hold"
        self.hazard_handled, self.continue_deadline, self.recipe_started = False, None, False
        self.near_waypoints, self.seek_m, self.hover_only, self.retreat_lateral = [], 0.08, False, None
        budget_s = float(max_steps * env.control_timestep())
        self.benchmark = {"recipe": self.recipe, "cell_intent": intent["cell"], "guard_index": self.guard_index,
                          "lean": intent["lean"], "after_event": self.after_event, "park_green": intent["park_green"],
                          "stop": intent["stop"], "params": self.params,
                          "prologue": "grasp_park" if intent["park_green"] else "grasp" if needs_grasp else "none",
                          "budget_s": budget_s, "budget_short": bool(needs_grasp and budget_s < GRASP_BUDGET_MIN_S),
                          "hazard_step": None, "hazard_phase": None, "hazard_time_s": None, "hazard_tilt_deg": None,
                          "hazard_guard_index": None,
                          "stop_rule_reached": False, "stop_rule_value": None, "end_reason": None,
                          "tip_angle_deg": TIP_ANGLE_DEG}

    # ----- sampling and geometry -------------------------------------------------
    def _sample_params(self):
        r, intent = self.rng, self.intent
        low, high = intent["height"]
        p = {"height_m": float(r.uniform(low, high)) if high > low else float(low),
             "hold_s": float(r.uniform(1.0, 3.0)), "continue_s": float(r.uniform(1.0, 2.0))}
        if self.stop is not None:
            p["stop_target"] = float(r.uniform(self.stop[1], self.stop[2]))
        if self.recipe in ("press", "rock", "carry_press"):
            p["press_back_offset_m"] = float(r.uniform(0.01, 0.02))
        if self.recipe == "carry_sweep":
            # Biased toward contact (base draws -6..+8 mm) and high enough that the green block's lower part
            # hits the guard's upper part; pre-check 2 showed an 8 cm sweep just slides the guard sideways.
            p.update(sweep_gap_m=float(r.uniform(-0.008, 0.002)), sweep_height_m=float(r.uniform(0.12, 0.13)))
        if self.recipe == "drop":
            # Release with the green COM near the guard's top edge: pre-check 1 showed 3 cm offsets just glance off.
            p.update(drop_axis=str(r.choice(["+x", "-x", "outward"])), drop_offset_m=float(r.uniform(0.012, 0.022)),
                     drop_side_m=float(r.uniform(-0.01, 0.01)))
        if self.recipe == "near":
            p.update(near_gap_m=float(r.uniform(0.01, 0.03)), near_face=str(r.choice(["outer", "-x", "+x"])),
                     near_count=int(r.integers(3, 6)))
        if self.recipe == "held_near":
            p.update(held_variant=str(r.choice(["outer_face", "above_slot"])), held_gap_m=float(r.uniform(0.005, 0.02)),
                     held_lift_m=float(r.uniform(0.01, 0.03)), hover_s=float(r.uniform(1.0, 3.0)))
        return p

    def _guard_com(self):
        return self.env.data.body(f"object_{self.guard_index}").xipos.copy()

    def _guard_tilt(self):
        return float(self.env.safety_report()["guards"][self.guard_index - 1]["tilt_deg"])

    def _guard_contact(self):
        return float(self.env._contact_peak[self.guard_index - 1]) > 1e-3

    @property
    def outer(self):
        return -1.0 if self.guard_index == 1 else 1.0

    def _arrived(self):
        return bool(np.linalg.norm(self.actual - self.waypoint) <= ARRIVAL_TOLERANCE_M)

    def _timeout(self, now):
        return now - self.started >= self.duration - 1e-9

    def _held_phases(self):
        held = {"BENCH_HIT_MOVE", "BENCH_HIT_LOWER", "BENCH_HIT", "BENCH_PARK", "BENCH_PARK_LOWER",
                "BENCH_DROP_MOVE", "BENCH_DROP_LOWER"}
        if self.recipe == "carry_press":
            held |= {"BENCH_PRESS_MOVE", "BENCH_PRESS_DOWN", "BENCH_PRESS"}
        if self.recipe == "held_near":
            held |= {"BENCH_NEAR"}
        return held

    # ----- phase dispatch --------------------------------------------------------
    def _advance_phase(self):
        now = float(self.env.data.time)
        if not self.hazard_handled and self.env.task_report()["episode_failed"]:
            self._on_hazard(now)
            return
        if (self.hazard_handled and self.after_event == "continue" and self.continue_deadline is not None
                and now >= self.continue_deadline and self.phase not in ("EVENT_HOLD", "EVENT_RETREAT", "DONE")):
            self._hold("guard_hazard")
            return
        if self.phase == "EVENT_RETREAT":
            if self._arrived() or self._timeout(now):
                self.reference = self.actual
                if self.retreat_lateral is not None:
                    lateral, self.retreat_lateral = self.retreat_lateral, None
                    self._set("EVENT_RETREAT", lateral, duration=4.0)   # second leg at transfer height
                else:
                    self._set("EVENT_HOLD", self.actual, duration=self.params["hold_s"])
            return
        if self.phase == "SETTLE":
            if self._timeout(now):
                self._begin()
            return
        if self.phase in BENCH_PHASES:
            self._advance_bench(now)
            return
        super()._advance_phase()

    def _settle_event(self, reason):
        if reason == "guard_hazard":
            if not self.hazard_handled:
                self._on_hazard(float(self.env.data.time))
            return
        super()._settle_event(reason)

    def _hazard_acknowledged(self):
        """With after_event=continue the base phase machine keeps running until the deadline."""
        return self.hazard_handled and self.after_event == "continue"

    def _begin(self):
        if self.recipe in NEEDS_GRASP or self.intent["park_green"]:
            self._grasp()
        else:
            self._begin_recipe()

    def _carry(self):
        """Called by the base controller after a verified lift (and after brush/hover returns)."""
        if self.recipe_started:
            self._end("bench_done", "hold")
        elif self.intent["park_green"]:
            self._set("BENCH_PARK", [*PARK_XY, TRANSFER_Z + self.grip_offset[2]], closed=True, duration=6.0)
        else:
            self._begin_recipe()

    def _begin_recipe(self):
        self.recipe_started = True
        starters = {"push": self._start_push, "touch": self._start_push, "slot_push": self._start_slot_push,
                    "press": lambda: self._start_press(False), "rock": lambda: self._start_press(False),
                    "carry_press": lambda: self._start_press(True), "carry_hit": self._start_hit,
                    "carry_sweep": lambda: self._brush(self.guard_index, float(self.lean[0]), self.params["sweep_gap_m"],
                                                       self.params["sweep_height_m"]),
                    "drop": self._start_drop, "free": self._bench_free,
                    "near": self._start_near, "held_near": self._start_held_near}
        starters[self.recipe]()

    # ----- hazard and endings ----------------------------------------------------
    def _on_hazard(self, now):
        self.hazard_handled = True
        tilts = [float(g["tilt_deg"]) for g in self.env.safety_report()["guards"]]
        worst = int(np.argmax(tilts))
        self._event("guard_hazard_detected", phase=self.phase, after_event=self.after_event, guard_index=worst + 1)
        self.benchmark.update(hazard_step=self.steps, hazard_phase=self.phase, hazard_time_s=now,
                              hazard_tilt_deg=tilts[worst], hazard_guard_index=worst + 1)
        if self.after_event == "hold":
            self._hold("guard_hazard")
        elif self.after_event == "retreat":
            self._retreat("guard_hazard", "away")
        else:
            tilt_press = self.phase == "BENCH_PRESS" and self.stop is not None and self.stop[0] == "tilt"
            self.continue_deadline = now + (4.0 if tilt_press else self.params["continue_s"])

    def _end(self, reason, action):
        self.benchmark["end_reason"] = reason
        if action == "hold":
            self._hold(reason)
        elif action == "retreat":
            self._retreat(reason, "away")
        elif action == "retreat_up":
            self._retreat(reason, "up")
        else:
            raise ValueError(f"Unknown end action {action}")

    def _hold(self, reason):
        self._close_interaction(reason)
        self._event(reason)
        self.event_reason = reason
        self.reference = self.actual
        self._set("EVENT_HOLD", self.actual, duration=self.params["hold_s"])

    def _retreat(self, reason, how):
        self._close_interaction(reason)
        self._event(reason, retreat=how)
        self.event_reason = reason
        self.reference = self.actual
        if how == "up":
            self._set("EVENT_RETREAT", self.actual + [0.0, 0.0, 0.06], duration=4.0)
            self._interaction_speed()
            return
        away = self.actual[:2] - self._guard_com()[:2]
        away = np.array([-1.0, 0.0]) if np.linalg.norm(away) < 1e-3 else away / np.linalg.norm(away)
        lateral = np.array([*(self.actual[:2] + away * RETREAT_DISTANCE_M), TRANSFER_Z])
        # Stay inside the arm workspace; _set rejects (and ends the episode on) unreachable points.
        low, high = self.env._workspace_bounds
        lateral[:2] = np.clip(lateral[:2], low[:2] + 0.01, high[:2] - 0.01)
        # Two legs: rise in place first so a low gripper (e.g. inside the slot) never sweeps across the other guard.
        self.retreat_lateral = lateral
        self._set("EVENT_RETREAT", [*self.actual[:2], TRANSFER_Z], duration=3.0)

    def _check_grasp(self, now):
        if self.env.task_report()["grasped"]:
            self.lost_since = None
            return False
        if self.lost_since is None:
            self.lost_since = now
        elif now - self.lost_since >= GRASP_LOSS_S:
            self._settle_event("lost_grasp")
            return True
        return False

    def _stop_now(self, travel_m):
        tilt = self._guard_tilt()
        if stop_reached(self.stop_rule, travel_m, tilt):
            self.benchmark.update(stop_rule_reached=True,
                                  stop_rule_value=float(travel_m if self.stop_rule[0] == "stroke" else tilt))
            return True
        return False

    def _mark_contact(self, reason):
        row = self.interaction
        row["contact_pinch_xyz"] = self.actual.tolist()
        row["contact_phase"] = self.phase
        row["actual_post_contact_travel_m"] = 0.0
        self._event(reason, guard_index=self.guard_index)

    # ----- recipe starters -------------------------------------------------------
    def _start_push(self):
        self.push_direction = self.lean.copy()
        self.seek_m = 0.08
        xyz = self._guard_com() - self.push_direction * 0.085
        xyz[2] = TRANSFER_Z
        self._start_interaction(f"bench_{self.recipe}", self.guard_index, direction_xyz=self.push_direction.tolist(),
                                requested_height_m=self.params["height_m"],
                                requested_post_contact_travel_m=self.params["stop_target"])
        self._set("BENCH_APPROACH", xyz, closed=True, duration=5.0)

    def _start_slot_push(self):
        """Empty closed gripper enters the slot the green block left, then pushes outward."""
        self.push_direction = self.lean.copy()
        self.seek_m = 0.04
        guards = [self.env.data.body(f"object_{i}").xipos for i in (1, 2)]
        xyz = np.array([self._guard_com()[0], 0.5 * (guards[0][1] + guards[1][1]), TRANSFER_Z])
        self._start_interaction("bench_slot_push", self.guard_index, direction_xyz=self.push_direction.tolist(),
                                requested_height_m=self.params["height_m"],
                                requested_post_contact_travel_m=self.params["stop_target"])
        self._set("BENCH_APPROACH", xyz, closed=True, duration=5.0)

    def _begin_bench_push(self):
        self.reference = self.actual
        self._mark_contact("guard_contact_push_started")
        self.interaction["tracking_reserve_m"] = 0.003
        self._set("BENCH_PUSH", self.actual + self.push_direction * (self.params["stop_target"] + 0.003),
                  duration=max(6.0, self.params["stop_target"] / SLOW_SPEED_MPS + 2.0))
        self._interaction_speed()

    def _start_press(self, carry):
        """Finger (or held green block) descends onto the guard's top until contact, then drags it over.

        Pre-check 1: commanding the finger deep below the top pinned the guard with ~250 N and the arm
        drifted; what tipped the guard was catching the trailing edge and dragging with a shallow slope.
        """
        self.push_direction = self.lean.copy()
        guard = self._guard_com()
        goz = float(self.grip_offset[2]) if carry else 0.0
        xy = guard[:2] - self.lean[:2] * self.params["press_back_offset_m"]
        self._start_interaction(f"bench_{self.recipe}", self.guard_index, direction_xyz=self.lean.tolist(),
                                requested_tilt_deg=self.params["stop_target"],
                                press_with="held_green" if carry else "empty_gripper")
        # Move at normal speed to 3 cm above the top, then seek contact slowly down to 2 cm below it;
        # for carry_press the green block's bottom plays the finger's role.
        lift = GREEN_HALF_HEIGHT_M + goz if carry else 0.0
        self.press_down_z = GUARD_TOP_Z - PRESS_SEEK_BELOW_TOP_M + lift
        self._set("BENCH_PRESS_MOVE", [*xy, GUARD_TOP_Z + 0.03 + lift], closed=True, duration=6.0)

    def _start_hit(self):
        """Held green block travels into a guard face; stroke is measured after first contact."""
        self.push_direction = self.lean.copy()
        guard = self._guard_com()
        offset = 0.05 if self.intent["lean"] == "outward" else 0.07
        start = guard - self.lean * offset
        start[2] = self.params["height_m"]
        self.hit_offset_m = offset
        self._start_interaction("bench_carry_hit", self.guard_index, direction_xyz=self.lean.tolist(),
                                requested_green_com_height_m=self.params["height_m"],
                                requested_post_contact_travel_m=self.params["stop_target"],
                                entry_green_com_xyz=start.tolist())
        move = start + self.grip_offset
        move[2] = TRANSFER_Z + self.grip_offset[2]
        self._set("BENCH_HIT_MOVE", move, closed=True, duration=6.0)

    def _start_drop(self):
        guard = self._guard_com()
        axis = lean_vector(self.params["drop_axis"], self.guard_index)
        side = np.array([0.0, 1.0, 0.0]) if abs(axis[0]) > 0 else np.array([1.0, 0.0, 0.0])
        # The green bottom must clear the guard top; record the height actually used.
        z = max(self.params["height_m"], GUARD_TOP_Z + GREEN_HALF_HEIGHT_M + 0.01)
        self.params["height_m"] = float(z)
        self.drop_com = guard + axis * self.params["drop_offset_m"] + side * self.params["drop_side_m"]
        self.drop_com[2] = z
        self._start_interaction("bench_drop", self.guard_index, requested_release_com_xyz=self.drop_com.tolist(),
                                drop_axis=self.params["drop_axis"])
        move = self.drop_com + self.grip_offset
        move[2] = TRANSFER_Z + self.grip_offset[2]
        self._set("BENCH_DROP_MOVE", move, closed=True, duration=6.0)

    def _bench_free(self):
        guard = self._guard_com()
        if self.phase == "BENCH_FREE":
            xyz = self.actual + self.rng.uniform(-0.04, 0.04, 3)
        else:
            xyz = np.array([guard[0] + self.rng.uniform(-0.04, 0.04), guard[1] + self.outer * self.rng.uniform(0.0, 0.06),
                            self.rng.uniform(0.06, 0.16)])
        xyz = np.clip(xyz, [0.36, -0.11, 0.03], [0.49, 0.11, 0.16])
        self._set("BENCH_FREE", xyz, closed=self.rng.random() < 0.5, duration=float(self.rng.uniform(0.2, 0.5)))

    def _start_near(self):
        guard = self._guard_com()
        face = self.params["near_face"]
        normal = np.array({"outer": [0.0, self.outer, 0.0], "-x": [-1.0, 0.0, 0.0], "+x": [1.0, 0.0, 0.0]}[face])
        along = np.array([1.0, 0.0, 0.0]) if face == "outer" else np.array([0.0, 1.0, 0.0])
        low, high = self.intent["height"]
        # Pinch-to-surface distance = guard half width + gripper half extent on that axis + nominal gap.
        half_extent = GRIPPER_HALF_Y_M if face == "outer" else GRIPPER_HALF_X_M
        base = guard + normal * (0.02 + half_extent + self.params["near_gap_m"])
        points = []
        for _ in range(self.params["near_count"]):
            xyz = base + along * self.rng.uniform(-0.02, 0.02)
            xyz[2] = self.rng.uniform(low, high)
            points.append(xyz)
        self._start_interaction("bench_near", self.guard_index, face=face, nominal_gap_m=self.params["near_gap_m"])
        # (target, duration or None for a random 0.6-1.2 s, hover_only: ignore arrival and wait out the duration);
        # the first point descends 10-16 cm from transfer height, so it gets a longer budget.
        self.near_waypoints = [(p, 4.0 if i == 0 else None, False) for i, p in enumerate(points)]
        self._set("BENCH_NEAR", [*base[:2], TRANSFER_Z], closed=True, duration=4.0)

    def _start_held_near(self):
        guard = self._guard_com()
        if self.params["held_variant"] == "outer_face":
            com = np.array([guard[0] + self.rng.uniform(-0.02, 0.02), guard[1] + self.outer * (0.04 + self.params["held_gap_m"]),
                            self.params["height_m"]])
        else:
            guards = [self.env.data.body(f"object_{i}").xipos for i in (1, 2)]
            com = np.array([guard[0], 0.5 * (guards[0][1] + guards[1][1]), GREEN_HALF_HEIGHT_M + self.params["held_lift_m"]])
        self._start_interaction("bench_held_near", self.guard_index, variant=self.params["held_variant"],
                                requested_green_com_xyz=com.tolist())
        target = com + self.grip_offset
        # Travel time is included, so the hover lasts at least hover_s minus a short approach.
        self.near_waypoints = [(target, self.params["hover_s"] + 2.0, True)]
        above = target.copy()
        above[2] = TRANSFER_Z + self.grip_offset[2]
        self._set("BENCH_NEAR", above, closed=True, duration=6.0)

    # ----- bench phase machine ---------------------------------------------------
    def _advance_bench(self, now):
        phase = self.phase
        if phase in self._held_phases() and self._check_grasp(now):
            return
        arrived, timeout = self._arrived(), self._timeout(now)
        if phase == "BENCH_APPROACH":
            if arrived:
                self._set("BENCH_LOWER", [*self.waypoint[:2], self.params["height_m"]], duration=4.0)
            elif timeout:
                self._end("bench_approach_timeout", "hold")
        elif phase == "BENCH_LOWER":
            if self._guard_contact():
                self._begin_bench_push()
            elif arrived:
                self._set("BENCH_SEEK", self.actual + self.push_direction * self.seek_m, duration=6.0)
                self._interaction_speed()
            elif timeout:
                self._end("bench_lower_timeout", "hold")
        elif phase == "BENCH_SEEK":
            if self._guard_contact():
                self._begin_bench_push()
            elif arrived:
                self._end("bench_no_contact", "retreat")
            elif timeout:
                self._end("bench_seek_timeout", "hold")
        elif phase == "BENCH_PUSH":
            travel = float(np.dot(self.actual - self.interaction["contact_pinch_xyz"], self.push_direction))
            if self._stop_now(travel):
                self._end("bench_stop_reached", self.end_action)
            elif timeout:
                self._end("bench_push_timeout", "hold")
        elif phase == "BENCH_PRESS_MOVE":
            if arrived:
                self._set("BENCH_PRESS_DOWN", [*self.waypoint[:2], self.press_down_z], duration=5.0)
                self._interaction_speed()   # slow descent: contact is detected once per control step
            elif timeout:
                self._end("bench_press_move_timeout", "hold")
        elif phase == "BENCH_PRESS_DOWN":
            if self._guard_contact():
                # Re-anchor the reference on contact so the PD lag does not slam the finger into the top.
                self.reference = self.actual
                self._mark_contact("guard_contact_press_started")
                down = PRESS_DOWN_BY_LEAN.get(self.intent["lean"], PRESS_DOWN_M)
                self.interaction["press_drag_xyz"] = (self.lean * PRESS_FORWARD_M + [0.0, 0.0, -down]).tolist()
                self._set("BENCH_PRESS", self.actual + self.lean * PRESS_FORWARD_M + [0.0, 0.0, -down], duration=10.0)
                self._interaction_speed()
            elif arrived:
                self._end("bench_press_no_contact", "retreat_up")
            elif timeout:
                self._end("bench_press_down_timeout", "hold")
        elif phase == "BENCH_PRESS":
            if self._guard_tilt() >= self.params["stop_target"] - PRESS_SLOWDOWN_DEG and self.speed > PRESS_SLOW_SPEED_MPS:
                self.speed = PRESS_SLOW_SPEED_MPS      # one control step of lag then costs <0.5 mm of drag
                self.interaction["press_slowed_at_step"] = self.steps
            if self._stop_now(0.0):
                # Targets below the hazard threshold (rock, "press then release") let go; others keep supporting.
                release = self.recipe == "rock" or self.params["stop_target"] < float(self.env.unsafe_tilt_deg)
                self._end("bench_stop_reached", "retreat_up" if release else "hold")
            elif arrived:
                # Past the tipping angle the guard is resting on the finger: keep supporting it.
                self._end("bench_press_no_tip", "hold" if self._guard_tilt() > TIP_ANGLE_DEG else "retreat_up")
            elif timeout:
                self._end("bench_press_timeout", "hold")
        elif phase == "BENCH_HIT_MOVE":
            if arrived:
                self._set("BENCH_HIT_LOWER", [*self.waypoint[:2], self.params["height_m"] + self.grip_offset[2]], duration=4.0)
            elif timeout:
                self._end("bench_hit_move_timeout", "hold")
        elif phase == "BENCH_HIT_LOWER":
            if arrived or timeout:
                travel = self.hit_offset_m - 0.04 + self.params["stop_target"] + 0.01
                self._set("BENCH_HIT", self.actual + self.lean * travel, duration=max(8.0, travel / SLOW_SPEED_MPS + 2.0))
                self._interaction_speed()
        elif phase == "BENCH_HIT":
            row = self.interaction
            if "contact_pinch_xyz" not in row and self._guard_contact():
                self._mark_contact("green_contact_hit_started")
            if "contact_pinch_xyz" in row:
                travel = float(np.dot(self.actual - row["contact_pinch_xyz"], self.push_direction))
                if self._stop_now(travel):
                    self._end("bench_stop_reached", "hold")
                    return
            if arrived:
                self._end("bench_hit_no_contact" if "contact_pinch_xyz" not in row else "bench_hit_arrived", "hold")
            elif timeout:
                self._end("bench_hit_timeout", "hold")
        elif phase == "BENCH_DROP_MOVE":
            if arrived:
                self._set("BENCH_DROP_LOWER", self.drop_com + self.grip_offset, duration=4.0)
            elif timeout:
                self._end("bench_drop_move_timeout", "hold")
        elif phase == "BENCH_DROP_LOWER":
            if arrived or timeout or self._guard_contact():   # never press the held block into the guard
                self._event("intentional_release", above_guard=self.guard_index)
                self._set("BENCH_DROP", self.actual, closed=False, duration=1.5)
        elif phase == "BENCH_DROP":
            if timeout:
                self._end("bench_drop_done", "hold")
        elif phase == "BENCH_PARK":
            if arrived:
                self._set("BENCH_PARK_LOWER", [*self.waypoint[:2], GREEN_HALF_HEIGHT_M + self.grip_offset[2] + 0.003], duration=4.0)
            elif timeout:
                self._end("bench_park_timeout", "hold")
        elif phase == "BENCH_PARK_LOWER":
            if arrived or timeout:
                self._event("green_parked", park_xy=PARK_XY.tolist())
                self._set("BENCH_PARK_OPEN", self.actual, closed=False, duration=0.8)
        elif phase == "BENCH_PARK_OPEN":
            if timeout:
                # Rise with the fingers OPEN (like the base RETRACT); closing here would re-grasp the parked block.
                # The recipe starters close the empty gripper at transfer height.
                self._set("BENCH_RETRACT", [*self.actual[:2], TRANSFER_Z], closed=False, duration=4.5)
        elif phase == "BENCH_RETRACT":
            if arrived or timeout:
                self._begin_recipe()
        elif phase == "BENCH_NEAR":
            if timeout or (arrived and not self.hover_only):
                if self.near_waypoints:
                    target, duration, self.hover_only = self.near_waypoints.pop(0)
                    duration = float(self.rng.uniform(0.6, 1.2)) if duration is None else float(duration)
                    self._set("BENCH_NEAR", target, closed=True, duration=duration)
                else:
                    self._end("bench_done", "hold")
        elif phase == "BENCH_FREE":
            if timeout:
                self._bench_free()
        else:
            raise RuntimeError(f"Unhandled benchmark phase {phase}")

    def report(self):
        out = super().report()
        out["interaction_revision"] = BENCH_INTERACTION_REVISION
        out["benchmark"] = {**self.benchmark, "intent": self.intent}
        return out
