"""Editable Cube scene with guard-tilt and horizontal COM safety checks.

Uses the installed stable-worldmodel/OGBench robot and physics without editing
site-packages. The custom hooks intentionally remove original random resets,
goal markers and success logic. Our grasp/hold task uses simulator truth,
not a learned model or classifier.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
from ogbench.manipspace import lie
from ogbench.manipspace.envs.env import CustomMuJoCoEnv
from PIL import Image
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

from scene_config import (
    ARM_FIXED_YAW_DEG, ARM_START_XYZ, ARM_START_MIN_Z_M, BLOCKS, IMAGE_HEIGHT, IMAGE_WIDTH, UNSAFE_TILT_DEG,
    UNSAFE_DISPLACEMENT_M, GUARD_DISPLACEMENT_CHECK_ENABLED, RENDER_SHADOWS,
    TARGET_COMPLETION_HEIGHT_M, TARGET_HOLD_SECONDS, GRASP_MIN_NORMAL_FORCE_N,
    FRONT_CAMERA_POSITION, FRONT_CAMERA_LOOK_AT, FRONT_CAMERA_FOVY_DEG,
)
from task_progress import TaskProgress


# Prevent floating-point noise at exactly 30 degrees from changing the label.
# This is numerical tolerance only, not a physical safety margin or hysteresis.
TILT_NUMERICAL_TOL_DEG = 1e-7
DISPLACEMENT_NUMERICAL_TOL_M = 1e-9


class ThreeBlockScene(CubeEnv):
    """Three movable boxes with guard safety and continuous grasp completion."""

    def __init__(self, blocks=None, unsafe_tilt_deg=UNSAFE_TILT_DEG,
                 arm_fixed_yaw_deg=ARM_FIXED_YAW_DEG,
                 unsafe_displacement_m=UNSAFE_DISPLACEMENT_M,
                 target_completion_height_m=TARGET_COMPLETION_HEIGHT_M,
                 target_hold_seconds=TARGET_HOLD_SECONDS,
                 displacement_check_enabled=GUARD_DISPLACEMENT_CHECK_ENABLED,
                 shadows_enabled=RENDER_SHADOWS):
        self.blocks = copy.deepcopy(BLOCKS if blocks is None else blocks)
        self.arm_start = np.asarray(ARM_START_XYZ, dtype=float)
        self.front_camera_position = np.asarray(FRONT_CAMERA_POSITION, dtype=float)
        self.front_camera_look_at = np.asarray(FRONT_CAMERA_LOOK_AT, dtype=float)
        self.front_camera_fovy_deg = float(FRONT_CAMERA_FOVY_DEG)
        self.arm_fixed_yaw_deg = float(arm_fixed_yaw_deg)
        self.unsafe_tilt_deg = float(unsafe_tilt_deg)
        self.unsafe_displacement_m = float(unsafe_displacement_m)
        self.displacement_check_enabled = displacement_check_enabled
        self.shadows_enabled = shadows_enabled
        self._task_progress = TaskProgress(target_completion_height_m, target_hold_seconds)
        self._validate_config()
        super().__init__(
            env_type="triple",
            ob_type="states",
            permute_blocks=False,
            mode="task",
            terminate_at_goal=False,
            visualize_info=False,
            pixel_transparent_arm=False,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
        )

    def _validate_config(self):
        self._front_camera_axes()
        for name in ("displacement_check_enabled", "shadows_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if not np.isfinite(GRASP_MIN_NORMAL_FORCE_N) or GRASP_MIN_NORMAL_FORCE_N <= 0:
            raise ValueError("GRASP_MIN_NORMAL_FORCE_N must be finite and positive")
        if not np.isfinite(self.unsafe_displacement_m) or self.unsafe_displacement_m <= 0:
            raise ValueError("unsafe_displacement_m must be finite and positive")
        if not np.isfinite(self.arm_fixed_yaw_deg):
            raise ValueError("arm_fixed_yaw_deg must be finite")
        if not np.isfinite(self.unsafe_tilt_deg) or not 0 < self.unsafe_tilt_deg < 180:
            raise ValueError("unsafe_tilt_deg must be finite and between 0 and 180 degrees.")
        if len(self.blocks) != 3:
            raise ValueError("This scene requires exactly 3 blocks: target + 2 guards.")
        if self.arm_start.shape != (3,) or not np.isfinite(self.arm_start).all():
            raise ValueError("ARM_START_XYZ must contain 3 finite values.")
        if np.any(self.arm_start < [0.25, -0.35, ARM_START_MIN_Z_M]) or np.any(
            self.arm_start > [0.60, 0.35, 0.35]
        ):
            raise ValueError("ARM_START_XYZ is outside the supported starting workspace.")
        for block in self.blocks:
            for key, dim in (("xy", 2), ("size", 3), ("color", 3)):
                value = np.asarray(block[key], dtype=float)
                if value.shape != (dim,) or not np.isfinite(value).all():
                    raise ValueError(f"{block['name']}: invalid {key}.")
                block[key] = value.tolist()
            if np.any(np.asarray(block["size"]) <= 0):
                raise ValueError("All block dimensions must be positive.")
            if np.any(np.asarray(block["color"]) < 0) or np.any(
                np.asarray(block["color"]) > 1
            ):
                raise ValueError("RGB color values must be between 0 and 1.")
            half_xy = np.asarray(block["size"][:2]) / 2
            xy = np.asarray(block["xy"])
            if np.any(xy - half_xy < [0.25, -0.35]) or np.any(
                xy + half_xy > [0.60, 0.35]
            ):
                raise ValueError(f"{block['name']}: block lies outside the scene workspace.")
        for i, left in enumerate(self.blocks):
            for right in self.blocks[i + 1 :]:
                separation = np.abs(np.asarray(left["xy"]) - right["xy"])
                half_sum = (
                    np.asarray(left["size"][:2]) + right["size"][:2]
                ) / 2
                if np.all(separation < half_sum + 0.001):
                    raise ValueError("Blocks overlap or have less than a 1 mm gap.")

    def _front_camera_axes(self):
        """Fixed optical axes from configured world points, not object tracking."""
        for value in (self.front_camera_position, self.front_camera_look_at):
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Front camera position/look-at must contain 3 finite values")
        if not np.isfinite(self.front_camera_fovy_deg) or not 1 <= self.front_camera_fovy_deg < 179:
            raise ValueError("Front camera field of view must be finite and in [1, 179)")
        forward = self.front_camera_look_at - self.front_camera_position
        if np.linalg.norm(forward) < 1e-8:
            raise ValueError("Front camera position and look-at must be distinct")
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        if np.linalg.norm(right) < 1e-8:
            raise ValueError("Front camera direction must not be parallel to world vertical")
        right /= np.linalg.norm(right)
        return np.concatenate([right, np.cross(-forward, right)])

    @property
    def fixed_effector_rotation(self):
        # World-Z yaw first, with the pinch/approach axis still facing down.
        return lie.SO3.from_z_radians(np.deg2rad(self.arm_fixed_yaw_deg)) @ self._effector_down_rotation

    @property
    def action_space(self):
        # Keep the upstream 5-D shape for compatibility, but the yaw slot is
        # explicitly zero-only. Sampling this space cannot request rotation.
        space = super().action_space
        space.low[3] = 0.0
        space.high[3] = 0.0
        return space

    def set_control(self, action):
        """Relative XYZ/gripper actions with one fixed absolute orientation.

        Preserve [dx, dy, dz, yaw, dgripper] shape, but reject nonzero yaw
        instead of silently discarding it in future action-labelled datasets.
        This changes only our scene subclass, not the installed OGBench env.
        """
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("action must contain 5 finite normalized values")
        if np.any(np.abs(action) > 1):
            raise ValueError("normalized actions must be between -1 and 1")
        if action[3] != 0:
            raise ValueError("Yaw control is disabled; action[3] must be zero")
        physical_action = self.unnormalize_action(action)
        target_xyz = np.clip(
            self.data.site_xpos[self._pinch_site_id] + physical_action[:3],
            *self._workspace_bounds,
        )
        current_gripper = np.clip(self.data.qpos[self._gripper_opening_joint_id] / 0.8, 0, 1)
        target_gripper = np.clip(current_gripper + physical_action[4], 0, 1)
        self._target_effector_pose = lie.SE3.from_rotation_and_translation(
            self.fixed_effector_rotation, target_xyz
        )
        attach_pose = self._target_effector_pose @ self._T_pa
        joints = self._ik.solve(
            pos=attach_pose.translation(), quat=attach_pose.rotation().wxyz,
            curr_qpos=self.data.qpos[self._arm_joint_ids].copy(),
        )
        if not np.isfinite(joints).all():
            raise RuntimeError("IK returned invalid joint targets")
        self.data.ctrl[self._arm_actuator_ids] = joints
        self.data.ctrl[self._gripper_actuator_ids] = 255.0 * target_gripper

    def set_tasks(self):
        # Called inside the parent constructor. Configuration already exists.
        positions = np.asarray(
            [[*block["xy"], block["size"][2] / 2] for block in self.blocks]
        )
        self.task_infos = [{
            "task_name": "grasp_target_without_disturbing_guards",
            "init_xyzs": positions,
            "goal_xyzs": positions.copy(),
        }]

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Edit scene_config.py instead of passing random reset options.")
        self._validate_config()
        # Deliberately bypass upstream task selection/domain randomization. Keep
        # its normal build/compile/reset pipeline and seed handling.
        self.cur_task_id = 1
        self.cur_task_info = self.task_infos[0]
        return CustomMuJoCoEnv.reset(self, seed=seed)

    def modify_mjcf_model(self, model):
        # Disable casting at the light source as well, covering the native
        # passive viewer in addition to our camera renderer. Keep illumination.
        if not self.shadows_enabled:
            for light in model.find_all("light"):
                if light.castshadow is not False:
                    light.castshadow = False
                    self.mark_dirty()
        # Set dimensions BEFORE compilation, so MuJoCo recomputes inertia/mass
        # from the inherited density. Never resize compiled collision geoms only.
        for i, block in enumerate(self.blocks):
            half_size = np.asarray(block["size"]) / 2
            for body_name in (f"object_{i}", f"object_target_{i}"):
                for geom in model.find("body", body_name).find_all("geom"):
                    if geom.size is None or not np.array_equal(geom.size, half_size):
                        geom.size = half_size
                        self.mark_dirty()
                    alpha = 0.0 if body_name.startswith("object_target_") else 1.0
                    rgba = np.asarray([*block["color"], alpha])
                    if geom.rgba is None or not np.array_equal(geom.rgba, rgba):
                        geom.rgba = rgba
                        self.mark_dirty()
        camera = model.find("camera", "front_pixels")
        if camera is None:
            raise RuntimeError("Fixed front_pixels camera was not found")
        for key, value in (("pos", self.front_camera_position),
                           ("xyaxes", self._front_camera_axes()),
                           ("fovy", self.front_camera_fovy_deg)):
            if not np.array_equal(getattr(camera, key), value):
                setattr(camera, key, value)
                self.mark_dirty()
        return model

    def render(self, camera="front_pixels", depth=False, segmentation=False,
               scene_option=None, scene_callback=None):
        """Apply one shadow policy to preview, dataset, and wrist cameras."""
        def configure_scene(scene):
            if scene_callback is not None:
                scene_callback(scene)
            scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.shadows_enabled

        return super().render(camera=camera, depth=depth, segmentation=segmentation,
                              scene_option=scene_option, scene_callback=configure_scene)

    def initialize_episode(self):
        self._data.qvel[:] = 0
        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_forward(self._model, self._data)
        # Start at a legal downward-facing end-effector pose. This avoids the
        # first zero action moving an out-of-workspace home pose unexpectedly.
        pose = lie.SE3.from_rotation_and_translation(
            self.fixed_effector_rotation, self.arm_start
        ) @ self._T_pa
        joints = self._ik.solve(
            pos=pose.translation(),
            quat=pose.rotation().wxyz,
            curr_qpos=self._home_qpos.copy(),
        )
        self._data.qpos[self._arm_joint_ids] = joints
        self._data.ctrl[self._arm_actuator_ids] = joints
        self._preview_arm_target = joints.copy()
        self._data.ctrl[self._gripper_actuator_ids] = 0
        for i, block in enumerate(self.blocks):
            joint = self._data.joint(f"object_joint_{i}")
            joint.qpos[:3] = [*block["xy"], block["size"][2] / 2]
            joint.qpos[3:] = [1, 0, 0, 0]
            joint.qvel[:] = 0
            # Targets remain hidden and are not interpreted as task goals.
            target_id = self._cube_target_mocap_ids[i]
            self._data.mocap_pos[target_id] = [0, 0, -1]
        self._cur_goal_ob = None
        self._cur_goal_rendered = None
        mujoco.mj_forward(self._model, self._data)
        # Capture each physical body's COM at reset, before the first physics
        # step. Copy values: keeping a view into data would move the baseline.
        self._guard_initial_com_xy = {
            index: self.data.body(f"object_{index}").xipos[:2].copy()
            for index in (1, 2)
        }
        # Cache physical colliders only, excluding visual meshes on pad bodies.
        self._grasp_geom_ids = {}
        for role, body_name in (
            ("target", "object_0"),
            ("left", "ur5e/robotiq/left_pad"),
            ("right", "ur5e/robotiq/right_pad"),
        ):
            body_id = self.model.body(body_name).id
            self._grasp_geom_ids[role] = {
                i for i in range(self.model.ngeom)
                if self.model.geom_bodyid[i] == body_id and
                (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
            }
            if not self._grasp_geom_ids[role]:
                raise RuntimeError(f"No physical colliders found for {body_name}")
        self._task_progress.reset()
        self.pre_step()
        self.post_step()

    def post_step(self):
        # Refresh poses after integration so body.xmat matches the current qpos,
        # rather than the pre-integration pose left in mj_step's derived data.
        mujoco.mj_forward(self._model, self._data)
        self._grasp_measurement = self.grasp_report()
        task = self._task_progress.update(
            float(self.data.time), self._grasp_measurement["target_height_m"],
            self._grasp_measurement["grasped"], self.safety_report()["unsafe"],
        )
        self._success = task["task_success"]

    def grasp_report(self):
        """Both pads must bear force against the target in this sampled state.

        Multiple contact points on a side are summed. A closed command, contact
        with the floor/guards, and zero-force/inactive contacts do not qualify.
        Dual contact plus sustained lift is a practical detector, not a proof
        of force closure. Read after post_step / mj_forward refreshes physics.
        """
        forces = {"left": 0.0, "right": 0.0}
        force = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.efc_address < 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in self._grasp_geom_ids["target"]:
                other = g2
            elif g2 in self._grasp_geom_ids["target"]:
                other = g1
            else:
                continue
            for side in forces:
                if other in self._grasp_geom_ids[side]:
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    if not np.isfinite(force[0]):
                        raise RuntimeError("Non-finite target contact force")
                    forces[side] += max(0.0, float(force[0]))
        left = forces["left"] > GRASP_MIN_NORMAL_FORCE_N
        right = forces["right"] > GRASP_MIN_NORMAL_FORCE_N
        return {
            "target_height_m": float(self.data.body("object_0").xipos[2]),
            "left_contact": bool(left), "right_contact": bool(right),
            "left_normal_force_n": forces["left"], "right_normal_force_n": forces["right"],
            "min_normal_force_n": GRASP_MIN_NORMAL_FORCE_N,
            "grasped": bool(left and right),
        }

    def task_report(self):
        """Snapshot of last physics check; rendering/reading never advances time."""
        return {**self._task_progress.report(), **self._grasp_measurement}

    def task_status_text(self):
        task = self.task_report()
        label = {"IN_PROGRESS": "未完成", "HOLDING": "正在保持",
                 "SUCCESS": "成功", "FAILED": "本次失败，按 R 重置"}[task["status"]]
        return (
            f"任务：{label} | 绿色块质心高度：{100 * task['target_height_m']:.2f} cm "
            f"（要求≥{100 * task['height_threshold_m']:g} cm） | "
            f"双指夹持：{'是' if task['grasped'] else '否'} | "
            f"连续保持：{task['hold_seconds']:.2f}/{task['required_hold_seconds']:g} 秒（仿真时间）"
        )

    def advance_physics(self):
        """Shared control step: monitor every 2-ms physics substep, not just 50 ms.

        Controls stay fixed during these substeps as in the original nstep call.
        This catches hazards/contact interruptions between control observations;
        it is still discrete simulation monitoring, not a continuous guarantee.
        """
        self.pre_step()
        self.post_step()  # Also observe the starting state (e.g. test fixtures).
        for _ in range(self._n_steps):
            mujoco.mj_step(self.model, self.data)
            self.post_step()
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def step(self, action):
        # Our task is explicitly evaluated after physics, overriding the original
        # Cube goal logic/timing. Keep Gym's five-result interface and zero reward.
        if self._reset_next_step:
            return self.reset()
        self.set_control(action)
        self.advance_physics()
        return (self.compute_observation(), self.compute_reward(),
                self.terminate_episode(), self.truncate_episode(), self.get_step_info())

    def safety_report(self):
        """Read current simulator poses; no classifier or episode-history latch.

        The body's local +Z is the pillar's height direction. R maps body axes
        into the world, so R[:, 2] is this direction in world coordinates.
        Its dot product with world-up [0, 0, 1] is R[2, 2]. Pure yaw leaves
        this unchanged. Do NOT take abs(): upside-down (180 deg) is unsafe too.

        Horizontal displacement uses the physical body's world COM (xipos),
        relative to its own reset COM. Tilt is always checked. Displacement
        contributes to unsafe only when displacement_check_enabled is true.
        Otherwise it is a diagnostic measurement only. This is current-state
        labelling, not a permanent episode-failure latch or a safety filter.

        reset/step/step_preview refresh poses. If manually editing qpos in a
        test, call mujoco.mj_forward(model, data) before requesting this report.
        """
        guards = []
        for index in (1, 2):  # Roles follow indices, not color recognition.
            body = self.data.body(f"object_{index}")
            rotation = body.xmat.reshape(3, 3)
            if not np.isfinite(rotation).all():
                raise RuntimeError(f"Invalid simulator orientation for guard {index}")
            cos_tilt = float(np.clip(rotation[2, 2], -1.0, 1.0))
            tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
            com_xy = body.xipos[:2]
            initial_com_xy = self._guard_initial_com_xy[index]
            if not np.isfinite(com_xy).all() or not np.isfinite(initial_com_xy).all():
                raise RuntimeError(f"Invalid simulator COM position for guard {index}")
            displacement_xy_m = float(np.linalg.norm(com_xy - initial_com_xy))
            unsafe_tilt = tilt_deg > self.unsafe_tilt_deg + TILT_NUMERICAL_TOL_DEG
            unsafe_displacement = (
                self.displacement_check_enabled and
                displacement_xy_m > self.unsafe_displacement_m + DISPLACEMENT_NUMERICAL_TOL_M
            )
            reasons = []
            if unsafe_tilt:
                reasons.append("tilt")
            if unsafe_displacement:
                reasons.append("displacement")
            guards.append({
                "index": index,
                "name": self.blocks[index]["name"],
                "tilt_deg": tilt_deg,
                "initial_com_xy_m": initial_com_xy.tolist(),
                "com_xy_m": com_xy.tolist(),
                "displacement_xy_m": displacement_xy_m,
                "unsafe_tilt": unsafe_tilt,
                "unsafe_displacement": unsafe_displacement,
                "unsafe_reasons": reasons,
                "unsafe": unsafe_tilt or unsafe_displacement,
            })
        return {
            "safety_scope": ("guard_tilt_or_displacement" if self.displacement_check_enabled
                             else "guard_tilt_only"),
            "displacement_check_enabled": self.displacement_check_enabled,
            "displacement_monitor_only": not self.displacement_check_enabled,
            "unsafe_tilt_deg": self.unsafe_tilt_deg,
            "unsafe_displacement_m": self.unsafe_displacement_m,
            "guards": guards,
            "unsafe": any(guard["unsafe"] for guard in guards),
        }

    def safety_status_text(self):
        safety = self.safety_report()
        left, right = safety["guards"]
        rule = "红柱倾角与位移规则" if self.displacement_check_enabled else "仅红柱倾角规则"
        state = "危险 (unsafe)" if safety["unsafe"] else f"安全（{rule}）"
        reasons = []
        for guard in safety["guards"]:
            side = "左红柱" if guard["index"] == 1 else "右红柱"
            if guard["unsafe_tilt"]:
                reasons.append(f"{side}倾角超限")
            if guard["unsafe_displacement"]:
                reasons.append(f"{side}位移超限")
        reason_text = "；原因：" + "、".join(reasons) if reasons else ""
        shift_rule = (f" 或 水平位移>{100 * self.unsafe_displacement_m:g} cm"
                      if self.displacement_check_enabled else "；COM 位移判定已关闭（仅记录）")
        return (
            f"左红柱倾角：{left['tilt_deg']:.2f}°，水平位移：{100 * left['displacement_xy_m']:.2f} cm | "
            f"右红柱倾角：{right['tilt_deg']:.2f}°，水平位移：{100 * right['displacement_xy_m']:.2f} cm | "
            f"当前状态：{state} | 阈值：倾角>{self.unsafe_tilt_deg:g}°{shift_rule}{reason_text}"
        )

    def safety_rule_text(self):
        rule = f"Unsafe: guard tilt >{self.unsafe_tilt_deg:g} deg"
        return rule + (f" OR shift >{100 * self.unsafe_displacement_m:g} cm"
                       if self.displacement_check_enabled else " | COM check OFF")

    def compute_reward(self):
        return 0.0

    def step_preview(self):
        """Advance physics while holding fixed joints, without an action policy.

        The step(action) interface has a zero-only yaw slot and fixed heading.
        Repeated relative zero actions would re-anchor at the current pose and
        accumulate gravity-induced drift, so the preview uses a fixed target.
        """
        self._data.ctrl[self._arm_actuator_ids] = self._preview_arm_target
        self._data.ctrl[self._gripper_actuator_ids] = 0
        self.advance_physics()

    def get_reset_info(self):
        info = super().get_reset_info()
        info.update(preview_only=False, task_defined=True)
        info.update(arm_fixed_yaw_deg=self.arm_fixed_yaw_deg, yaw_control_enabled=False)
        info["safety"] = self.safety_report()
        info["unsafe"] = info["safety"]["unsafe"]
        info["task"] = self.task_report()
        info["episode_failed"] = info["task"]["episode_failed"]
        info["success"] = info["task"]["task_success"]
        return info

    def get_step_info(self):
        info = super().get_step_info()
        info.update(preview_only=False, task_defined=True)
        info.update(arm_fixed_yaw_deg=self.arm_fixed_yaw_deg, yaw_control_enabled=False)
        info["safety"] = self.safety_report()
        info["unsafe"] = info["safety"]["unsafe"]
        info["task"] = self.task_report()
        info["episode_failed"] = info["task"]["episode_failed"]
        info["success"] = info["task"]["task_success"]
        return info

    def block_report(self):
        rows = []
        for i, block in enumerate(self.blocks):
            joint = self.model.joint(f"object_joint_{i}")
            geom = self.model.geom(f"object_{i}")
            rows.append({
                "index": i,
                "name": block["name"],
                "position_xyz": self.data.joint(f"object_joint_{i}").qpos[:3].tolist(),
                "size_xyz": (2 * geom.size).tolist(),
                "rgba": geom.rgba.tolist(),
                "free_joint": int(joint.type[0]) == int(mujoco.mjtJoint.mjJNT_FREE),
                "collision_enabled": bool(geom.contype[0] or geom.conaffinity[0]),
                "mass_kg": float(self.model.body(f"object_{i}").mass[0]),
            })
        return rows

    def close(self):
        self.close_passive_viewer()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Check physics without a GUI")
    parser.add_argument("--steps", type=int, default=40, help="Headless physics steps")
    parser.add_argument("--snapshot-dir", type=Path, help="New directory for PNG and scene report")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must not be negative")
    env = ThreeBlockScene()
    try:
        env.reset(seed=0)
        initial_positions = np.asarray([row["position_xyz"] for row in env.block_report()])
        initial_effector = env.data.site_xpos[env._pinch_site_id].copy()
        if args.headless:
            for _ in range(args.steps):
                env.step_preview()
            if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
                raise RuntimeError("Non-finite simulator state")
        if args.snapshot_dir:
            args.snapshot_dir.mkdir(parents=True, exist_ok=False)
            Image.fromarray(env.render()).save(args.snapshot_dir / "front.png")
            Image.fromarray(env.render(camera="side_pixels")).save(args.snapshot_dir / "side.png")
        report = {
            "preview_only": False,
            "task_defined": True,
            "world_model_loaded": False,
            "arm_fixed_yaw_deg": env.arm_fixed_yaw_deg,
            "yaw_control_enabled": False,
            "shadows_enabled": env.shadows_enabled,
            "headless_steps": args.steps if args.headless else 0,
            "blocks": env.block_report(),
            "safety": env.safety_report(),
            "task": env.task_report(),
            "max_block_displacement_m": float(np.max(np.linalg.norm(
                np.asarray([r["position_xyz"] for r in env.block_report()]) - initial_positions,
                axis=1,
            ))),
            "effector_displacement_m": float(np.linalg.norm(
                env.data.site_xpos[env._pinch_site_id] - initial_effector
            )),
        }
        print(json.dumps(report, indent=2))
        print(env.safety_status_text(), flush=True)
        print(env.task_status_text(), flush=True)
        if args.snapshot_dir:
            (args.snapshot_dir / "scene_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        if not args.headless:
            reset_requested = threading.Event()
            print("Static arm preview with task checks. Edit scene_config.py and restart to apply changes.")
            print("Drag/scroll to change the view; R resets this layout; close the window to exit.")
            print(env.safety_rule_text() + "; status prints every second.")
            next_status_time = time.monotonic() + 1.0
            last_unsafe = env.safety_report()["unsafe"]
            with env.passive_viewer(
                key_callback=lambda key: reset_requested.set() if key in (82, 114) else None
            ) as viewer:
                while viewer.is_running():
                    start = time.monotonic()
                    if reset_requested.is_set():
                        reset_requested.clear()
                        env.reset(seed=0)
                    env.step_preview()
                    viewer.sync()
                    unsafe = env.safety_report()["unsafe"]
                    now = time.monotonic()
                    if now >= next_status_time or unsafe != last_unsafe:
                        print(env.safety_status_text(), flush=True)
                        print(env.task_status_text(), flush=True)
                        next_status_time = now + 1.0
                        last_unsafe = unsafe
                    time.sleep(max(0.0, env.control_timestep() - (time.monotonic() - start)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
