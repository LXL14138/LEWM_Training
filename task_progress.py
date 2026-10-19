"""Simulation-time grasp task tracking, independent of MuJoCo and rendering."""

import math


class TaskProgress:
    """Current completion plus a reset-only latch for any observed guard hazard.

    This tracker needs a sequence of observations. Its output is NOT an
    instantaneous three-class label suitable for a stateless classifier.
    """

    def __init__(self, height_m=0.10, hold_seconds=1.0):
        if not math.isfinite(height_m) or height_m <= 0:
            raise ValueError("height_m must be finite and positive")
        if not math.isfinite(hold_seconds) or hold_seconds <= 0:
            raise ValueError("hold_seconds must be finite and positive")
        self.height_m = float(height_m)
        self.required_hold_seconds = float(hold_seconds)
        self.reset()

    def reset(self):
        self._hold_start = None
        self._last_time = None
        self._report = {
            "status": "IN_PROGRESS", "target_height_m": 0.0,
            "height_threshold_m": self.height_m, "height_ok": False,
            "grasped": False, "goal_conditions_met": False,
            "hold_seconds": 0.0, "required_hold_seconds": self.required_hold_seconds,
            "episode_failed": False, "task_success": False,
        }

    def update(self, sim_time, target_height_m, grasped, unsafe):
        if not math.isfinite(sim_time) or sim_time < 0:
            raise ValueError("sim_time must be finite and non-negative")
        if self._last_time is not None and sim_time < self._last_time:
            raise ValueError("sim_time cannot go backwards without reset")
        if not math.isfinite(target_height_m):
            raise ValueError("target_height_m must be finite")
        height_ok = target_height_m >= self.height_m - 1e-9
        self._last_time = float(sim_time)
        goal_conditions_met = bool(grasped and height_ok)
        failed = bool(self._report["episode_failed"] or unsafe)
        if goal_conditions_met and not failed:
            if self._hold_start is None:
                self._hold_start = float(sim_time)
            held = float(sim_time - self._hold_start)
        else:
            self._hold_start = None
            held = 0.0
        success = bool(not failed and goal_conditions_met and
                       held >= self.required_hold_seconds - 1e-9)
        status = "FAILED" if failed else (
            "SUCCESS" if success else "HOLDING" if goal_conditions_met else "IN_PROGRESS"
        )
        self._report.update(
            status=status, target_height_m=float(target_height_m),
            height_ok=bool(height_ok), grasped=bool(grasped),
            goal_conditions_met=goal_conditions_met, hold_seconds=held,
            episode_failed=failed, task_success=success,
        )
        return self.report()

    def report(self):
        return self._report.copy()
