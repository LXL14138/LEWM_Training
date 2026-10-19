"""Visual fixed-scene scripted grasp using simulator truth and env.step actions.

Not a learned policy, world model, planner or dataset collector. No object
teleporting or grasp constraints; task/safety thresholds belong to the scene.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from scene_config import IMAGE_HEIGHT, IMAGE_WIDTH
from three_block_scene import ThreeBlockScene


PANEL_HEIGHT = 297
PHASE_NAMES = {
    "SETTLE": "张开夹爪并稳定初态", "APPROACH": "移动到绿色块上方",
    "DESCEND": "下降到抓取高度", "CLOSE": "闭合并确认真实夹持",
    "LIFT": "抬升绿色块", "HOLD": "保持并等待完成判定",
    "PROBE_LIFT": "诊断试抬：保持闭合，不以双指判定阻止抬升",
    "PROBE_HOLD": "诊断试抬：在上方保持并观察绿色块",
    "PROBE_DONE": "试抬结束，查看绿色块实际高度和任务判定",
    "SUCCESS": "抓取成功，仿真已暂停", "FAILED": "本次失败，仿真已暂停",
}


class AutoGraspController:
    """Finite sequence with measured arrivals, contacts, hazards and timeouts.

    Waypoints use privileged simulator positions. Every motion is an ordinary
    normalized 5-D action [dx,dy,dz,0,dgripper], with four effective controls.
    Future world-model data collection needs its own complete transition logger.
    """

    def __init__(self, env, speed_mps=0.06, phase_timeout_s=15.0,
                 grasp_offset_x_m=0.0, close_timeout_s=None, probe_lift=False):
        if not np.isfinite(speed_mps) or not 0 < speed_mps <= 0.1:
            raise ValueError("speed_mps must be finite and in (0, 0.1] m/s")
        if not np.isfinite(phase_timeout_s) or phase_timeout_s <= 0:
            raise ValueError("phase_timeout_s must be finite and positive")
        if not np.isfinite(grasp_offset_x_m) or abs(grasp_offset_x_m) > 0.06:
            raise ValueError("grasp_offset_x_m must be finite and within +/-0.06 m")
        if close_timeout_s is not None and (not np.isfinite(close_timeout_s) or close_timeout_s <= 0):
            raise ValueError("close_timeout_s must be finite and positive, or None")
        if not isinstance(probe_lift, bool):
            raise ValueError("probe_lift must be a boolean")
        self.env = env
        self.speed_mps = float(speed_mps)
        self.phase_timeout_s = float(phase_timeout_s)
        self.grasp_offset_x_m = float(grasp_offset_x_m)
        self.close_timeout_s = None if close_timeout_s is None else float(close_timeout_s)
        self.probe_lift = probe_lift
        self.reset()

    @property
    def finished(self):
        return self.phase in ("SUCCESS", "FAILED", "PROBE_DONE")

    def reset(self):
        _, self.last_info = self.env.reset(seed=0)
        self.phase = "SETTLE"
        self.failure_reason = None
        self.phase_history = [self.phase]
        self._phase_started = float(self.env.data.time)
        self._stable_start = None
        self._attempt_started = float(self.env.data.time)
        self.last_action = np.zeros(5)
        self.control_steps = 0
        self.probe_samples = []
        target = self.env.data.body("object_0").xipos.copy()
        self.original_target_xyz = target.copy()
        # A deliberate aiming error, NOT a changed object pose or safety label.
        # Shift the entire approach/grasp/lift column along world X only; guards
        # lie along Y. This reduces, but does not guarantee absence of, contact.
        target[0] += self.grasp_offset_x_m
        self.grasp_target_xyz = target.copy()
        initial = self.env.data.site_xpos[self.env._pinch_site_id].copy()
        self.reference_xyz = initial.copy()
        # Pinch-site waypoints, not block COM heights. Place the pinch slightly
        # above the target's initial COM; resizing the target changes this height.
        grasp_z = float(target[2] + 0.005)
        lift_z = float(self.last_info["task"]["height_threshold_m"] + 0.05)
        overhead_z = max(float(initial[2]), lift_z,
                         max(block["size"][2] for block in self.env.blocks[1:]) + 0.07)
        self.waypoints = {
            "SETTLE": initial,
            "APPROACH": np.array([target[0], target[1], overhead_z]),
            "DESCEND": np.array([target[0], target[1], grasp_z]),
            "CLOSE": np.array([target[0], target[1], grasp_z]),
            "LIFT": np.array([target[0], target[1], lift_z]),
            "HOLD": np.array([target[0], target[1], lift_z]),
        }
        low, high = self.env._workspace_bounds
        for name, xyz in self.waypoints.items():
            if not np.isfinite(xyz).all() or np.any(xyz < low) or np.any(xyz > high):
                raise ValueError(f"{name} waypoint outside robot workspace; inspect scene configuration")
        if self.probe_lift:
            self.waypoints["PROBE_LIFT"] = self.waypoints["LIFT"].copy()
            self.waypoints["PROBE_HOLD"] = self.waypoints["HOLD"].copy()
        return self.last_info

    def _record_probe_sample(self):
        """Diagnostic 20-Hz measurements, not overrides of task/grasp labels."""
        task = self.last_info["task"]
        self.probe_samples.append({
            "time_s": float(self.env.data.time), "phase": self.phase,
            "target_com_z_m": float(task["target_height_m"]),
            "pinch_z_m": float(self.env.data.site_xpos[self.env._pinch_site_id][2]),
            "grasped": bool(task["grasped"]),
            "left_contact": bool(task["left_contact"]),
            "right_contact": bool(task["right_contact"]),
            "left_normal_force_n": float(task["left_normal_force_n"]),
            "right_normal_force_n": float(task["right_normal_force_n"]),
            "task_success": bool(task["task_success"]),
            "unsafe": bool(self.last_info["unsafe"]),
            "episode_failed": bool(self.last_info["episode_failed"]),
        })

    def _transition(self, phase, reason=None):
        self.phase = phase
        self.failure_reason = reason
        self.phase_history.append(phase)
        self._phase_started = float(self.env.data.time)
        self._stable_start = None

    def _stable_for(self, condition, seconds):
        if not condition:
            self._stable_start = None
            return False
        now = float(self.env.data.time)
        if self._stable_start is None:
            self._stable_start = now
        return now - self._stable_start >= seconds - 1e-9

    def step(self):
        if self.finished:
            return self.last_info  # Frozen result, not continued physical holding.
        current = self.env.get_step_info()
        if current["episode_failed"] or current["unsafe"]:
            self.last_info = current
            self._transition("FAILED", "Guard safety limit exceeded; press R to retry.")
            return current
        actual = self.env.data.site_xpos[self.env._pinch_site_id]
        reference_delta = self.waypoints[self.phase] - self.reference_xyz
        distance = float(np.linalg.norm(reference_delta))
        max_delta = self.speed_mps * self.env.control_timestep()
        if distance > max_delta:
            reference_delta = reference_delta * (max_delta / distance)
        candidate = self.reference_xyz + reference_delta
        lead = candidate - actual
        if np.linalg.norm(lead) > 0.02:
            candidate = actual + lead * (0.02 / np.linalg.norm(lead))
        self.reference_xyz = np.clip(candidate, *self.env._workspace_bounds)
        delta = self.reference_xyz - actual
        close = self.phase in ("CLOSE", "LIFT", "HOLD", "PROBE_LIFT", "PROBE_HOLD")
        # Limit reference motion, not actual-to-target error: re-anchoring each
        # 3-mm command to actual would accumulate gravity/tracking drift. Keep
        # the reference fixed at CLOSE/HOLD and correct error via normal actions.
        physical = np.array([*delta, 0.0, 1.0 if close else -1.0])
        self.last_action = self.env.normalize_action(physical)
        if not np.isfinite(self.last_action).all():
            self._transition("FAILED", "Invalid robot state; reset required.")
            return self.last_info
        try:
            _, _, terminated, truncated, self.last_info = self.env.step(self.last_action)
        except (RuntimeError, ValueError) as exc:
            if not np.isfinite(self.env.data.qpos).all() or not np.isfinite(self.env.data.qvel).all():
                raise RuntimeError("Invalid physics state; cannot safely render this attempt") from exc
            self._transition("FAILED", "Robot control error: " + str(exc))
            return self.last_info
        self.control_steps += 1
        if not np.isfinite(self.env.data.qpos).all() or not np.isfinite(self.env.data.qvel).all():
            raise RuntimeError("Non-finite simulator state; stopping automatic grasp")
        task = self.last_info["task"]
        if self.phase in ("PROBE_LIFT", "PROBE_HOLD"):
            self._record_probe_sample()
        if self.last_info["episode_failed"] or self.last_info["unsafe"]:
            self._transition("FAILED", "Guard safety limit exceeded; press R to retry.")
            return self.last_info
        if terminated or truncated:
            self._transition("FAILED", "Environment stopped before automatic sequence finished.")
            return self.last_info
        elapsed = float(self.env.data.time) - self._phase_started
        timeout = self.phase_timeout_s
        if self.phase == "CLOSE" and self.close_timeout_s is not None:
            timeout = min(timeout, self.close_timeout_s)
        if elapsed >= timeout:
            if self.phase == "CLOSE" and self.probe_lift:
                # This opt-in SIMULATION diagnostic tests whether the block
                # actually follows the fingers, even when the contact detector
                # says no. Guard hazards and physics errors were checked above.
                self._transition("PROBE_LIFT")
                self._record_probe_sample()
                return self.last_info
            reason = ("No stable two-finger grasp before timeout." if self.phase == "CLOSE"
                      else f"{self.phase} timed out; waypoint/task condition not reached.")
            self._transition("FAILED", reason)
            return self.last_info
        actual = self.env.data.site_xpos[self.env._pinch_site_id]
        arrived = (np.linalg.norm(actual - self.waypoints[self.phase]) <= 0.005 and
                   np.linalg.norm(self.reference_xyz - self.waypoints[self.phase]) <= 1e-6)
        if self.phase == "SETTLE":
            if elapsed >= 1.0 - 1e-9 and arrived:
                self._transition("APPROACH")
        elif self.phase == "APPROACH":
            if self._stable_for(arrived, 0.20):
                self._transition("DESCEND")
        elif self.phase == "DESCEND":
            if self._stable_for(arrived, 0.20):
                self._transition("CLOSE")
        elif self.phase == "CLOSE":
            if not self.probe_lift and self._stable_for(task["grasped"], 0.30):
                self._transition("LIFT")
        elif self.phase == "PROBE_LIFT":
            if self._stable_for(arrived, 0.20):
                self._transition("PROBE_HOLD")
        elif self.phase == "PROBE_HOLD":
            if elapsed >= 1.0 - 1e-9:
                # Neutral diagnostic outcome: reaching the arm waypoint does
                # not establish that the block was grasped or the task complete.
                self._transition("PROBE_DONE")
        elif self.phase == "LIFT":
            if not task["grasped"]:
                self._transition("FAILED", "Lost two-finger grasp while lifting.")
            elif self._stable_for(arrived, 0.20):
                self._transition("HOLD")
        elif self.phase == "HOLD":
            if not task["grasped"]:
                self._transition("FAILED", "Lost two-finger grasp during hold.")
            elif task["task_success"] and elapsed >= task["required_hold_seconds"] - 1e-9:
                # Also give the visible HOLD stage a full dwell at the final
                # waypoint; env's timer may already start during the ascent.
                self._transition("SUCCESS")
        return self.last_info

    def report(self):
        probe = {"enabled": self.probe_lift, "sampling": "control-step endpoints (20 Hz)",
                 "samples": [dict(row) for row in self.probe_samples]}
        if self.probe_samples:
            initial, final = self.probe_samples[0], self.probe_samples[-1]
            peak = max(row["target_com_z_m"] for row in self.probe_samples)
            probe.update(
                initial_target_com_z_m=initial["target_com_z_m"],
                peak_target_com_z_m=peak, final_target_com_z_m=final["target_com_z_m"],
                peak_target_rise_m=peak - initial["target_com_z_m"],
                initial_pinch_z_m=initial["pinch_z_m"], final_pinch_z_m=final["pinch_z_m"],
                ever_dual_contact=any(row["grasped"] for row in self.probe_samples),
                ever_task_success=any(row["task_success"] for row in self.probe_samples),
            )
        return {
            "phase": self.phase, "phase_history": self.phase_history.copy(),
            "failure_reason": self.failure_reason,
            "sim_time_s": float(self.env.data.time) - self._attempt_started,
            "control_steps": self.control_steps, "speed_mps": self.speed_mps,
            "grasp_offset_x_m": self.grasp_offset_x_m,
            "grasp_target_xyz_m": self.grasp_target_xyz.tolist(),
            "original_target_xyz_m": self.original_target_xyz.tolist(),
            "close_timeout_s": self.close_timeout_s,
            "probe_lift": probe,
            "result_frozen": self.finished, "task": self.env.task_report(),
            "safety": self.env.safety_report(), "world_model_loaded": False,
            "shadows_enabled": self.env.shadows_enabled,
            "uses_simulator_object_position": True, "training_data_collected": False,
        }


class AutoGraspInput:
    def __init__(self):
        self.paused = False
        self.reset_requested = False
        self.quit_requested = False
        self.camera_index = 0

    def handle(self, event):
        if event.type == pygame.QUIT:
            self.quit_requested = True
        elif event.type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWMINIMIZED):
            self.paused = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.quit_requested = True
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_r:
                self.reset_requested = True
                self.paused = False
            elif event.key == pygame.K_TAB:
                self.camera_index = 1 - self.camera_index


def draw_auto_window(screen, font, controller, inputs):
    env = controller.env
    frame = env.render(camera=("front_pixels", "side_pixels")[inputs.camera_index])
    screen.fill((20, 24, 32))
    screen.blit(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), (0, 0))
    task = env.task_report()
    guards = env.safety_report()["guards"]
    xyz = env.data.site_xpos[env._pinch_site_id]
    light, yellow, green, red = (225, 231, 240), (245, 201, 116), (110, 235, 160), (255, 100, 100)
    phase_color = red if controller.phase == "FAILED" else green if controller.phase == "SUCCESS" else yellow
    state = "FROZEN RESULT" if controller.finished else "PAUSED" if inputs.paused else "RUNNING"
    lines = [
        (f"{'PROBE LIFT' if controller.probe_lift else 'AUTO GRASP'} | X aim offset: {100 * controller.grasp_offset_x_m:+.1f} cm | Scripted", light),
        ("Space: pause/resume    R: restart    Tab: view    Esc: exit", light),
        (f"Stage: {controller.phase}", phase_color),
        (f"Simulation: {state}   Time: {env.data.time:.2f} s", yellow),
        (f"Green COM: {100 * task['target_height_m']:.2f} cm / >= {100 * task['height_threshold_m']:g} cm", light),
        (f"Grasp: {'YES' if task['grasped'] else 'NO'} (L:{int(task['left_contact'])} R:{int(task['right_contact'])})   Hold: {task['hold_seconds']:.2f}/{task['required_hold_seconds']:g} s", light),
        (f"Left guard: {guards[0]['tilt_deg']:.2f} deg   Shift: {100 * guards[0]['displacement_xy_m']:.2f} cm", red if guards[0]["unsafe"] else green),
        (f"Right guard: {guards[1]['tilt_deg']:.2f} deg   Shift: {100 * guards[1]['displacement_xy_m']:.2f} cm", red if guards[1]["unsafe"] else green),
        (env.safety_rule_text(), yellow),
        (f"Pinch XYZ: {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f} m | Fixed yaw: {env.arm_fixed_yaw_deg:g}", light),
        (controller.failure_reason or ("Probe finished; arm lifting does NOT imply task success." if controller.phase == "PROBE_DONE"
          else "Success checked in physics; result frozen for inspection." if controller.finished
          else "Script uses simulator positions; no learned model or planner."), phase_color),
        ("No training data saved. Focus loss pauses; Space resumes.", light),
    ]
    for row, (text, color) in enumerate(lines):
        if font.size(text)[0] > IMAGE_WIDTH - 24:
            while font.size(text + "...")[0] > IMAGE_WIDTH - 24:
                text = text[:-1]
            text += "..."  # Complete diagnostic remains available in terminal/report.
        screen.blit(font.render(text, True, color), (12, IMAGE_HEIGHT + 10 + 23 * row))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run one bounded attempt without a visible window")
    parser.add_argument("--speed", type=float, default=0.06, help="Cartesian command limit in m/s, at most 0.1")
    parser.add_argument("--phase-timeout", type=float, default=15.0, help="Maximum simulation seconds per phase")
    parser.add_argument("--grasp-offset-x", type=float, default=0.0, help="Aim offset along world X in metres (+/-0.06 max); objects stay fixed")
    parser.add_argument("--close-timeout", type=float, help="Optional shorter closing wait in simulation seconds; does not override safety labels")
    parser.add_argument("--probe-lift", action="store_true", help="Simulation diagnostic: after CLOSE wait, lift even without two-pad grasp; retain safety and task checks")
    parser.add_argument("--snapshot-dir", type=Path, help="Headless only: new directory for result PNG and JSON, not training data")
    args = parser.parse_args()
    if args.snapshot_dir and not args.headless:
        parser.error("--snapshot-dir requires --headless")
    if not np.isfinite(args.speed) or not 0 < args.speed <= 0.1:
        parser.error("--speed must be finite and in (0, 0.1]")
    if not np.isfinite(args.phase_timeout) or args.phase_timeout <= 0:
        parser.error("--phase-timeout must be finite and positive")
    if not np.isfinite(args.grasp_offset_x) or abs(args.grasp_offset_x) > 0.06:
        parser.error("--grasp-offset-x must be finite and within +/-0.06 m")
    if args.close_timeout is not None and (not np.isfinite(args.close_timeout) or args.close_timeout <= 0):
        parser.error("--close-timeout must be finite and positive")
    if args.snapshot_dir:
        args.snapshot_dir.mkdir(parents=True, exist_ok=False)
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = ThreeBlockScene()
    try:
        controller = AutoGraspController(env, args.speed, args.phase_timeout,
                                         args.grasp_offset_x, args.close_timeout, args.probe_lift)
        inputs = AutoGraspInput()
        if not args.headless or args.snapshot_dir:
            pygame.display.init()
            pygame.font.init()
            screen = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT + PANEL_HEIGHT))
            pygame.display.set_caption("Three blocks - Automatic grasp")
            font = pygame.font.Font(None, 22)
        clock = pygame.time.Clock()
        print("固定场景自动抓取：自动启动。空格暂停/继续，R重新开始，Tab切换视角，Esc退出。", flush=True)
        print("使用模拟器物体位置，不是 world model 或 RRT；本次不采集训练数据。", flush=True)
        print(f"抓取瞄准点沿世界 X 偏移 {100 * controller.grasp_offset_x_m:+g} cm，"
              "方块位置与安全/完成规则不变；偏移不预设成功或失败标签。", flush=True)
        print("失去窗口焦点会暂停；返回画面后按空格继续。成功/失败后冻结仿真供查看。", flush=True)
        if controller.probe_lift:
            print("诊断模式：闭合等待结束后试抬，并保持 1 秒；双指判定为否也继续观察。红柱危险仍按原控制步流程停止。", flush=True)
            print("PROBE_DONE 只表示试抬结束，不代表抓取成功；请观察 Green COM 和真实任务判定。", flush=True)
        previous_phase = None
        while not inputs.quit_requested:
            if not args.headless:
                for event in pygame.event.get():
                    inputs.handle(event)
            if inputs.quit_requested:
                break
            if inputs.reset_requested:
                inputs.reset_requested = False
                controller.reset()
                previous_phase = None
            if not inputs.paused and not controller.finished:
                controller.step()
            if controller.phase != previous_phase:
                print(f"[{env.data.time:.2f}s] {PHASE_NAMES[controller.phase]}", flush=True)
                if controller.failure_reason:
                    print("原因：" + controller.failure_reason, flush=True)
                if controller.phase == "PROBE_DONE":
                    probe = controller.report()["probe_lift"]
                    print(f"试抬时绿色块质心：起始 {100 * probe['initial_target_com_z_m']:.2f} cm，"
                          f"最高 {100 * probe['peak_target_com_z_m']:.2f} cm，"
                          f"最后 {100 * probe['final_target_com_z_m']:.2f} cm。"
                          f"当前任务成功判定：{controller.last_info['task']['task_success']}", flush=True)
                previous_phase = controller.phase
            if not args.headless:
                draw_auto_window(screen, font, controller, inputs)
                clock.tick(round(1 / env.control_timestep()))
            elif controller.finished:
                if args.snapshot_dir:
                    draw_auto_window(screen, font, controller, inputs)
                    pygame.image.save(screen, str(args.snapshot_dir / "result.png"))
                    inputs.camera_index = 1
                    draw_auto_window(screen, font, controller, inputs)
                    pygame.image.save(screen, str(args.snapshot_dir / "side.png"))
                    (args.snapshot_dir / "report.json").write_text(
                        json.dumps(controller.report(), indent=2) + "\n", encoding="utf-8"
                    )
                print(json.dumps(controller.report(), indent=2), flush=True)
                return 0 if controller.phase in ("SUCCESS", "PROBE_DONE") else 1
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
