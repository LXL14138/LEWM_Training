"""Small synthetic on-disk fixtures; no MuJoCo, world model, or training needed."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from inspect_grasp_dataset import (
    BOOL_FIELDS, export_snapshots, main, replay_episode, validate_dataset, validate_episode,
)


class DatasetFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.episode = self.root / "episode_0000"
        self.episode.mkdir()
        (self.episode / "images").mkdir()
        self.metadata, self.arrays = self.make_fixture()
        self.write_episode()
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def make_fixture(actions=2):
        states = actions + 1
        positions = np.array([[0.425, 0, 0.02], [0.425, -0.09, 0.05], [0.425, 0.09, 0.05]])
        low = np.array([-0.05, -0.05, -0.05, -0.3, -1.0])
        high = -low
        metadata = {
            "schema_version": 1, "status": "complete", "num_actions": actions,
            "num_states": states, "control_dt": 0.05, "physics_dt": 0.002,
            "image": {"width": 16, "height": 12, "camera": "front_pixels", "overlay": False},
            "action_low": low.tolist(), "action_high": high.tolist(),
            "fixed_yaw_deg": 90.0, "unsafe_tilt_deg": 30.0,
            "unsafe_displacement_m": 0.01, "target_height_threshold_m": 0.1,
            "required_hold_seconds": 1.0, "min_normal_force_n": 0.001,
            "guard_initial_com_xy_m": positions[1:, :2].tolist(),
            "initial_block_positions_xyz": positions.tolist(),
            "start_sample": {"seed": 10, "accepted": True, "xyz": [0.3, 0, 0.25]},
            "outcome": {"phase": "FAILED", "failure_reason": "Synthetic timeout fixture"},
        }
        arrays = {
            "time": np.arange(states, dtype=np.float64) * 0.05,
            "qpos": np.zeros((states, 35)), "qvel": np.zeros((states, 32)), "ctrl": np.zeros((states, 7)),
            "object_com_xyz": np.tile(positions, (states, 1, 1)),
            "object_quat_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (states, 3, 1)),
            "pinch_xyz": np.tile([0.30, 0.0, 0.25], (states, 1)),
            "actions": np.tile([0.2, -0.1, 0.0, 0.0, -1.0], (actions, 1)),
            "action_phase": np.full(actions, "SETTLE", dtype="U12"),
            "hold_seconds": np.zeros(states), "target_height_m": np.full(states, 0.02),
            "guard_tilt_deg": np.zeros((states, 2)),
            "guard_displacement_xy_m": np.zeros((states, 2)),
            "finger_normal_force_n": np.zeros((states, 2)),
        }
        arrays["physical_actions"] = low + (arrays["actions"] + 1) * (high - low) / 2
        arrays.update({field: np.zeros(states, dtype=bool) for field in BOOL_FIELDS})
        return metadata, arrays

    def write_episode(self, frames=True):
        (self.episode / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        np.savez_compressed(self.episode / "trajectory.npz", **self.arrays)
        if frames:
            for index in range(self.metadata["num_states"]):
                Image.new("RGB", (16, 12), (index % 256, 20, 30)).save(self.episode / "images" / f"{index:06d}.png")

    def write_manifest(self, status="complete", entries=None, requested=1):
        if entries is None:
            entries = [{"id": 0, "path": self.episode.name, "status": "complete",
                        "phase": self.metadata["outcome"]["phase"],
                        "num_actions": self.metadata["num_actions"], "num_states": self.metadata["num_states"]}]
        manifest = {"schema_version": 1, "status": status, "requested_episodes": requested, "episodes": entries}
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def assertInvalid(self, substring):
        result = validate_episode(self.episode)
        self.assertFalse(result["valid"], result)
        self.assertIn(substring, " ".join(result["errors"]))

    def set_grasp_height(self):
        self.arrays["object_com_xyz"][:, 0, 2] = 0.11
        self.arrays["target_height_m"][:] = 0.11
        self.arrays["finger_normal_force_n"][:] = 2.0
        for key in ("height_ok", "left_contact", "right_contact", "grasped", "goal_conditions_met"):
            self.arrays[key][:] = True

    def test_normal_failed_episode_is_valid(self):
        result = validate_episode(self.episode)
        self.assertTrue(result["valid"], result)
        self.assertEqual((result["num_actions"], result["num_states"]), (2, 3))
        self.assertEqual(result["outcome"], "FAILED")

    def test_successful_episode(self):
        self.metadata, self.arrays = self.make_fixture(actions=22)
        self.set_grasp_height()
        self.arrays["hold_seconds"][:] = self.arrays["time"]
        self.arrays["task_success"][:] = self.arrays["time"] >= 1.0 - 1e-9
        self.metadata["outcome"] = {"phase": "SUCCESS", "failure_reason": None}
        self.write_episode()
        self.assertTrue(validate_episode(self.episode)["valid"])

    def test_zero_actions_has_one_state_and_image(self):
        self.metadata, self.arrays = self.make_fixture(actions=0)
        # Existing fixtures remain as extra frames until put in a fresh episode.
        self.episode = self.root / "episode_0001"
        (self.episode / "images").mkdir(parents=True)
        self.write_episode()
        self.assertTrue(validate_episode(self.episode)["valid"])

    def test_count_mismatch(self):
        self.metadata["num_states"] = 4
        self.write_episode()
        self.assertInvalid("num_states must equal")

    def test_wrong_array_shape(self):
        self.arrays["qpos"] = self.arrays["qpos"][:-1]
        self.write_episode()
        self.assertInvalid("qpos")

    def test_missing_array(self):
        del self.arrays["actions"]
        self.write_episode()
        self.assertInvalid("missing trajectory arrays")

    def test_time_must_be_float64(self):
        self.arrays["time"] = self.arrays["time"].astype("float32")
        self.write_episode()
        self.assertInvalid("time must use float64")

    def test_time_intervals(self):
        self.arrays["time"][2] = 0.11
        self.write_episode()
        self.assertInvalid("time intervals")

    def test_time_reset_not_cross_episode(self):
        self.arrays["time"] += 10.0
        self.write_episode()
        self.assertInvalid("time zero")

    def test_nan_state(self):
        self.arrays["qvel"][0, 0] = np.nan
        self.write_episode()
        self.assertInvalid("finite real")

    def test_no_pickle_even_extra_array(self):
        self.arrays["extra_payload"] = np.array([{"bad": True}], dtype=object)
        self.write_episode()
        self.assertInvalid("allow_pickle=False")

    def test_extra_numeric_array_must_be_finite(self):
        self.arrays["physics_state"] = np.full((3, 10), np.nan)
        self.write_episode()
        self.assertInvalid("finite real")

    def test_optional_rotation_must_be_rotation(self):
        self.arrays["pinch_rotation"] = np.tile(np.eye(3) * 2, (3, 1, 1))
        self.write_episode()
        self.assertInvalid("orthonormality")

    def test_optional_task_status_matches_labels(self):
        self.arrays["task_status"] = np.full(3, "SUCCESS", dtype="U12")
        self.write_episode()
        self.assertInvalid("task_status")

    def test_optional_transition_terminal_arrays_align_with_actions(self):
        self.arrays["terminated"] = np.zeros(2, dtype=bool)
        self.arrays["truncated"] = np.zeros(2, dtype=bool)
        self.write_episode()
        self.assertTrue(validate_episode(self.episode)["valid"])
        self.arrays["truncated"] = np.zeros(3, dtype=bool)
        self.write_episode()
        self.assertInvalid("boolean transition array")

    def test_bool_not_integer_labels(self):
        self.arrays["unsafe"] = self.arrays["unsafe"].astype(int)
        self.write_episode()
        self.assertInvalid("boolean array")

    def test_wrong_yaw_action(self):
        self.arrays["actions"][0, 3] = 0.3
        self.write_episode()
        self.assertInvalid("normalized yaw")

    def test_out_of_range_action(self):
        self.arrays["actions"][0, 0] = 1.5
        self.write_episode()
        self.assertInvalid("[-1, 1]")

    def test_wrong_physical_scaling(self):
        self.arrays["physical_actions"][0, 0] = 0.03
        self.write_episode()
        self.assertInvalid("physical action scaling")

    def test_quaternion_normalization(self):
        self.arrays["object_quat_wxyz"][0, 0, 0] = 0.7
        self.write_episode()
        self.assertInvalid("quaternion unit norm")

    def test_tilt_measurement_checked_against_quaternion(self):
        self.arrays["guard_tilt_deg"][0, 0] = 15.0
        self.write_episode()
        self.assertInvalid("guard tilt from quaternion")

    def test_displacement_measurement_checked_against_com(self):
        self.arrays["guard_displacement_xy_m"][0, 0] = 0.009
        self.write_episode()
        self.assertInvalid("guard COM displacement")

    def test_unsafe_label_inconsistent(self):
        self.arrays["unsafe"][1] = True
        self.write_episode()
        self.assertInvalid("unsafe does not match")

    def use_schema_two(self, shift_enabled=False):
        self.metadata.update(schema_version=2, displacement_check_enabled=shift_enabled,
                             safety_scope="guard_tilt_or_displacement" if shift_enabled else "guard_tilt_only")
        self.metadata["image"]["shadows_enabled"] = False

    def test_schema_two_shift_is_recorded_but_not_unsafe_when_disabled(self):
        self.use_schema_two()
        self.arrays["object_com_xyz"][1, 1, 0] += 0.02
        self.arrays["guard_displacement_xy_m"][1, 0] = 0.02
        self.write_episode()
        result = validate_episode(self.episode)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["unsafe_states"], 0)
        self.metadata["displacement_check_enabled"] = True
        self.metadata["safety_scope"] = "guard_tilt_or_displacement"
        self.write_episode()
        self.assertInvalid("unsafe does not match")

    def test_schema_two_tilt_still_requires_hazard_label(self):
        self.use_schema_two()
        half = np.deg2rad(40) / 2
        self.arrays["object_quat_wxyz"][1, 1] = [np.cos(half), np.sin(half), 0, 0]
        self.arrays["guard_tilt_deg"][1, 0] = 40
        self.write_episode()
        self.assertInvalid("unsafe does not match")
        self.arrays["unsafe"][1] = True
        self.arrays["episode_failed"][1:] = True
        self.write_episode()
        result = validate_episode(self.episode)
        self.assertTrue(result["valid"], result)

    def test_schema_two_requires_explicit_boolean_settings(self):
        self.use_schema_two()
        for bad in (None, 0, "false"):
            self.metadata["displacement_check_enabled"] = bad
            self.write_episode()
            self.assertInvalid("displacement_check_enabled must be a boolean")
        self.metadata.pop("displacement_check_enabled")
        self.write_episode()
        self.assertInvalid("displacement_check_enabled must be a boolean")
        self.metadata["displacement_check_enabled"] = False
        self.metadata["image"].pop("shadows_enabled")
        self.write_episode()
        self.assertInvalid("image.shadows_enabled")

    def test_legacy_schema_cannot_silently_disable_displacement(self):
        self.metadata["displacement_check_enabled"] = False
        self.write_episode()
        self.assertInvalid("schema 1 requires displacement safety")

    def test_schema_two_manifest_supported(self):
        self.use_schema_two()
        self.write_episode()
        manifest = self.write_manifest()
        manifest["schema_version"] = 2
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = validate_dataset(self.root)
        self.assertTrue(result["valid"], result)

    def test_valid_hazard_then_current_safe_still_failed(self):
        self.arrays["object_com_xyz"][1, 1, 0] += 0.02
        self.arrays["guard_displacement_xy_m"][1, 0] = 0.02
        self.arrays["unsafe"][1] = True
        self.arrays["episode_failed"][1:] = True
        self.write_episode()
        result = validate_episode(self.episode)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["unsafe_states"], 1)

    def test_episode_failure_cannot_clear(self):
        self.arrays["episode_failed"][1] = True
        self.write_episode()
        self.assertInvalid("cannot clear")

    def test_unsaved_substep_hazard_latch_is_allowed(self):
        self.arrays["episode_failed"][1:] = True
        self.write_episode()
        result = validate_episode(self.episode)
        self.assertTrue(result["valid"], result)
        self.assertTrue(any("2-ms" in warning for warning in result["warnings"]))

    def test_true_grasp_requires_two_force_contacts(self):
        self.arrays["grasped"][1] = True
        self.write_episode()
        self.assertInvalid("grasped does not match")

    def test_contact_force_threshold(self):
        self.arrays["finger_normal_force_n"][1, 0] = 0.1
        self.write_episode()
        self.assertInvalid("left_contact")

    def test_target_height_from_com(self):
        self.arrays["target_height_m"][1] = 0.12
        self.write_episode()
        self.assertInvalid("target COM height")

    def test_hold_can_restart_between_control_observations(self):
        self.set_grasp_height()
        self.arrays["hold_seconds"][:] = [0, 0.05, 0.026]
        self.write_episode()
        self.assertTrue(validate_episode(self.episode)["valid"])

    def test_hold_cannot_increase_faster_than_time(self):
        self.set_grasp_height()
        self.arrays["hold_seconds"][:] = [0, 0.07, 0.10]
        self.write_episode()
        self.assertInvalid("increase faster")

    def test_hold_clears_when_failed(self):
        self.set_grasp_height()
        self.arrays["hold_seconds"][:] = [0, 0.05, 0.1]
        self.arrays["episode_failed"][2] = True
        self.write_episode()
        self.assertInvalid("hold timer must reset")

    def test_success_cannot_be_declared_early(self):
        self.arrays["task_success"][2] = True
        self.write_episode()
        self.assertInvalid("task_success")

    def test_success_outcome_needs_final_success(self):
        self.metadata["outcome"]["phase"] = "SUCCESS"
        self.write_episode()
        self.assertInvalid("SUCCESS outcome requires")

    def test_missing_image(self):
        (self.episode / "images" / "000001.png").unlink()
        self.assertInvalid("frame set mismatch")

    def test_extra_frame(self):
        Image.new("RGB", (16, 12)).save(self.episode / "images" / "000003.png")
        self.assertInvalid("extra=")

    def test_wrong_image_size(self):
        Image.new("RGB", (20, 12)).save(self.episode / "images" / "000001.png")
        self.assertInvalid("16x12 RGB PNG")

    def test_bad_png(self):
        (self.episode / "images" / "000001.png").write_bytes(b"not an image")
        self.assertInvalid("cannot identify image")

    def test_overlay_flag(self):
        self.metadata["image"]["overlay"] = True
        self.write_episode()
        self.assertInvalid("overlay=false")

    def test_interrupted_episode_invalid(self):
        self.metadata["status"] = "interrupted"
        self.write_episode()
        self.assertInvalid("not complete")

    def test_complete_dataset(self):
        result = validate_dataset(self.root)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["total_states"], 3)

    def test_independent_episodes_each_start_at_zero(self):
        old_episode = self.episode
        self.episode = self.root / "episode_0001"
        (self.episode / "images").mkdir(parents=True)
        self.write_episode()
        entries = [{"id": index, "path": path.name, "status": "complete", "phase": "FAILED",
                    "num_actions": 2, "num_states": 3} for index, path in enumerate([old_episode, self.episode])]
        self.write_manifest(entries=entries, requested=2)
        result = validate_dataset(self.root)
        self.assertTrue(result["valid"], result)
        self.assertEqual((result["total_actions"], result["total_states"]), (4, 6))

    def test_dataset_incomplete(self):
        self.write_manifest(status="collecting")
        result = validate_dataset(self.root)
        self.assertFalse(result["valid"])
        self.assertIn("dataset is not complete", " ".join(result["errors"]))

    def test_manifest_path_traversal(self):
        entry = {"id": 0, "path": "../elsewhere", "status": "complete"}
        self.write_manifest(entries=[entry])
        result = validate_dataset(self.root)
        self.assertFalse(result["valid"])
        self.assertIn("child directory", " ".join(result["errors"]))

    def test_manifest_duplicate_episode(self):
        manifest = self.write_manifest()
        self.write_manifest(entries=manifest["episodes"] * 2, requested=2)
        result = validate_dataset(self.root)
        self.assertFalse(result["valid"])
        self.assertIn("duplicate", " ".join(result["errors"]))

    def test_manifest_counts_disagree(self):
        manifest = self.write_manifest()
        manifest["episodes"][0]["num_actions"] = 99
        self.write_manifest(entries=manifest["episodes"])
        result = validate_dataset(self.root)
        self.assertFalse(result["valid"])
        self.assertIn("disagrees", " ".join(result["errors"]))

    def test_unlisted_episode_directory(self):
        (self.root / "episode_0099").mkdir()
        result = validate_dataset(self.root)
        self.assertFalse(result["valid"])
        self.assertIn("absent from manifest", " ".join(result["errors"]))

    def test_snapshot_export_exact_copies_and_no_overwrite(self):
        output = self.root / "snapshots"
        files = export_snapshots(self.episode, output)
        self.assertEqual(Path(files["middle"]).read_bytes(), (self.episode / "images" / "000001.png").read_bytes())
        with self.assertRaises(FileExistsError):
            export_snapshots(self.episode, output)

    def test_cli_read_only_validation(self):
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root)]), 0)
        after = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_cli_partial_returns_failure(self):
        self.write_manifest(status="interrupted")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root)]), 1)

    def test_cli_report_contains_validation_result(self):
        output = self.root / "validation.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root), "--report", str(output)]), 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["valid"])
        self.assertEqual(report["total_actions"], 2)
        self.assertEqual(report["total_states"], 3)

    def test_cli_report_refuses_to_overwrite(self):
        output = self.root / "validation.json"
        output.write_text("existing report", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                main([str(self.root), "--report", str(output)])
        self.assertNotEqual(failure.exception.code, 0)
        self.assertEqual(output.read_text(encoding="utf-8"), "existing report")

    def test_cli_invalid_data_still_exports_report_and_returns_failure(self):
        output = self.root / "invalid_validation.json"
        self.write_manifest(status="interrupted")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root), "--report", str(output)]), 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(report["valid"])
        self.assertTrue(report["errors"])

    def test_cli_snapshot_does_not_require_pygame(self):
        output = self.root / "new_snapshots"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--episode", str(self.episode), "--snapshot-dir", str(output)]), 0)
        self.assertTrue((output / "last.png").is_file())

    def test_replay_controls_dummy_window(self):
        # Exercise real PNG decode, surface drawing and the event handling path,
        # but never open a visible window or claim a physical-keyboard test.
        with mock.patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}):
            import pygame
            events = [
                [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)],
                [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)],
                [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)],
                [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)],
                [pygame.event.Event(pygame.WINDOWFOCUSLOST)],
                [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)],
            ]
            with mock.patch.object(pygame.event, "get", side_effect=events) as get_events:
                replay_episode(self.episode)
            self.assertEqual(get_events.call_count, 6)
            self.assertFalse(pygame.get_init())


if __name__ == "__main__":
    unittest.main()
