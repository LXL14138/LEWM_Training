"""Synthetic-fixture tests only; never read or change the real 5000 episodes."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image
import torch

from wm_training.dataset import EpisodeWindows
from wm_training.prepare_data import (
    action_statistics, contained_path, letterbox, make_split, prepare, read_json,
    sha256, validate_arrays,
)


def manifest_fixture(count=20):
    modes = ["carry", "free", "grasp", "guard"]
    return {"status": "complete", "schema_version": 3, "dataset_kind": "guided_exploration_v1",
            "requested_episodes": count, "class_names": {"0": "instant hover", "1": "unsafe", "2": "other safe"},
            "episodes": [{"id": i, "path": f"episode_{i:04d}", "status": "complete",
                          "num_actions": 5, "num_states": 6, "mode": modes[i % 4]} for i in range(count)]}


def episode_fixture(episode_id):
    metadata = {"schema_version": 3, "dataset_kind": "guided_exploration_v1", "status": "complete",
                "image": {"width": 16, "height": 12, "camera": "front_pixels", "overlay": False},
                "control_dt": 0.05, "physics_dt": 0.002, "num_actions": 5, "num_states": 6,
                "state_alignment": "state[i] -- actions[i] --> state[i+1]",
                "action_order": ["dx", "dy", "dz", "yaw_zero_only", "dgripper"],
                "action_low": [-0.05, -0.05, -0.05, -0.3, -1],
                "action_high": [0.05, 0.05, 0.05, 0.3, 1], "fixed_yaw_deg": 90,
                "blocks": [], "unsafe_tilt_deg": 30, "displacement_check_enabled": False,
                "safety_scope": "guard_tilt_only", "hover_goal_xyz": [0.42, 0, 0.17]}
    action = np.zeros((5, 5))
    action[:, 0] = np.arange(5) * 0.1
    action[:, 1] = episode_id * 0.01
    action[:, 2] = 0.1  # additional constant channel must also be safe
    action[:, 4] = [-1, -1, 1, 1, 0]
    arrays = {"actions": action, "time": np.arange(6) * 0.05,
              "unsafe": np.array([0, 0, 0, 0, 1, 0], dtype=bool),
              "hover_conditions_met": np.array([0, 0, 1, 1, 1, 0], dtype=bool),
              "hover_success": np.zeros(6, dtype=bool),
              "episode_failed": np.array([0, 0, 0, 0, 1, 1], dtype=bool),
              "class_label": np.array([2, 2, 0, 0, 1, 2]),
              "hover_goal_xyz": np.tile([0.42, 0, 0.17], (6, 1))}
    return metadata, arrays


def write_fixture(root):
    root.mkdir()
    manifest = manifest_fixture()
    (root / "manifest.json").write_text(json.dumps(manifest))
    split = make_split(manifest, 17)
    # Only selected pilot files exist, proving that conversion doesn't read the
    # reserved test or unused training images/NPZ at all.
    for i in split["train"][:4] + split["val"][:1]:
        episode = root / f"episode_{i:04d}"
        images = episode / "images"
        images.mkdir(parents=True)
        metadata, arrays = episode_fixture(i)
        (episode / "metadata.json").write_text(json.dumps(metadata))
        np.savez_compressed(episode / "trajectory.npz", **arrays)
        for t in range(6):
            Image.new("RGB", (16, 12), (t * 30, i * 7 % 256, 20)).save(images / f"{t:06d}.png")
    return manifest


class SplitTests(unittest.TestCase):
    def test_full_size_and_independence(self):
        manifest = manifest_fixture(5000)
        split = make_split(manifest, 17)
        self.assertEqual({k: len(v) for k, v in split.items()}, {"train": 4500, "val": 250, "test": 250})
        self.assertEqual(len(set(sum(split.values(), []))), 5000)
        self.assertEqual(split, make_split(manifest, 17))
        self.assertNotEqual(split, make_split(manifest, 18))
        manifest["episodes"].reverse()
        self.assertEqual(split, make_split(manifest, 17))
        for name, count in (("train", 80), ("val", 20)):
            modes = [i % 4 for i in split[name][:count]]
            self.assertEqual([modes.count(i) for i in range(4)], [count // 4] * 4)

    def test_duplicates_and_incomplete_rejected(self):
        manifest = manifest_fixture()
        manifest["episodes"][1]["id"] = 0
        with self.assertRaises(ValueError):
            make_split(manifest)
        manifest = manifest_fixture()
        manifest["status"] = "interrupted"
        with self.assertRaises(ValueError):
            make_split(manifest)


class ContractTests(unittest.TestCase):
    def test_action_mapping(self):
        meta, arrays = episode_fixture(0)
        result = validate_arrays(arrays, meta, manifest_fixture()["episodes"][0])
        np.testing.assert_allclose(result, arrays["actions"][:, [0, 1, 2, 4]])

    def test_bad_transitions_rejected(self):
        for mutation in ("yaw", "time", "label", "goal", "nonfinite"):
            with self.subTest(mutation=mutation):
                meta, arrays = episode_fixture(0)
                if mutation == "yaw":
                    arrays["actions"][2, 3] = 0.1
                elif mutation == "time":
                    arrays["time"][3] += 0.01
                elif mutation == "label":
                    arrays["class_label"][4] = 0
                elif mutation == "goal":
                    arrays["hover_goal_xyz"][3, 0] += 0.01
                else:
                    arrays["actions"][1, 1] = np.nan
                with self.assertRaises(ValueError):
                    validate_arrays(arrays, meta, manifest_fixture()["episodes"][0])

    def test_zero_std_has_finite_unit_scale(self):
        data = np.ones((8, 4))
        stats = action_statistics(data.sum(0), (data**2).sum(0), len(data))
        self.assertEqual(stats["scale"], [1, 1, 1, 1])
        self.assertEqual(stats["constant_channels"], [0, 1, 2, 3])

    def test_letterbox_preserves_full_width(self):
        image = Image.new("RGB", (40, 20), (200, 0, 0))
        out = letterbox(image, 28)
        self.assertEqual(out.shape, (28, 28, 3))
        self.assertEqual(out[14, 0, 0], 200)
        self.assertEqual(out[14, -1, 0], 200)
        self.assertNotEqual(out[0, 14, 0], 200)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                contained_path(tmp, "../outside")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source, self.cache = root / "raw", root / "cache"
        self.manifest = write_fixture(self.source)

    def convert(self):
        with redirect_stdout(io.StringIO()):
            return prepare(self.source, self.cache, train_episodes=4, val_episodes=1, seed=17, image_size=28)

    def test_read_only_and_train_only_statistics(self):
        before = {str(p.relative_to(self.source)): sha256(p) for p in self.source.rglob("*") if p.is_file()}
        index = self.convert()
        after = {str(p.relative_to(self.source)): sha256(p) for p in self.source.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        actions = np.concatenate([episode_fixture(i)[1]["actions"][:, [0, 1, 2, 4]].astype(np.float32)
                                  for i in index["selected_ids"]["train"]]).astype(np.float64)
        np.testing.assert_allclose(index["normalization"]["mean"], actions.mean(0))
        np.testing.assert_allclose(index["normalization"]["std"], actions.std(0, ddof=1), atol=1e-7)
        self.assertEqual(len(list((self.cache / "episodes").glob("*.h5"))), 5)
        self.assertTrue(index["reserved_test_not_read"])

    def test_windows_do_not_cross_boundary_and_keep_final_frame(self):
        index = self.convert()
        data = EpisodeWindows(self.cache, "train", history_size=3)
        self.addCleanup(data.close)
        self.assertEqual(len(data), 12)  # 4 episodes, each 5 - 3 + 1 windows
        self.assertEqual(data.locate(2), (0, 2))
        self.assertEqual(data.locate(3), (1, 0))
        sample = data[2]
        self.assertEqual(set(sample), {"pixels", "action"})
        self.assertEqual(tuple(sample["pixels"].shape), (4, 3, 28, 28))
        self.assertEqual(tuple(sample["action"].shape), (3, 4))
        self.assertTrue(torch.isfinite(sample["action"]).all())
        red = (sample["pixels"][:, 0, 14, 14] * index["pixel_std"][0] + index["pixel_mean"][0]) * 255
        np.testing.assert_allclose(red.numpy(), [60, 90, 120, 150], atol=1e-4)
        with self.assertRaises(IndexError):
            data[len(data)]

    def test_pickle_and_spawn_dataloader(self):
        self.convert()
        data = EpisodeWindows(self.cache, "train", 3)
        self.addCleanup(data.close)
        expected = data[0]
        restored = pickle.loads(pickle.dumps(data))
        self.addCleanup(restored.close)
        torch.testing.assert_close(expected["action"], restored[0]["action"])
        loader = torch.utils.data.DataLoader(data, batch_size=2, num_workers=1,
                                             multiprocessing_context="spawn", persistent_workers=False)
        batches = list(loader)
        self.assertEqual(len(batches), 6)
        torch.testing.assert_close(batches[0]["pixels"][0], expected["pixels"])

    def test_overwrite_is_rejected(self):
        self.convert()
        before = sha256(self.cache / "index.json")
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual(before, sha256(self.cache / "index.json"))

    def test_incomplete_cache_is_not_published(self):
        i = make_split(self.manifest, 17)["train"][0]
        # The fixture file is intentionally corrupt, never a real trajectory.
        path = self.source / f"episode_{i:04d}" / "images/000002.png"
        path.write_bytes(b"not a PNG")
        with self.assertRaises(OSError):
            self.convert()
        self.assertFalse((self.cache / "index.json").exists())
        self.assertTrue((self.cache / "INCOMPLETE.json").exists())

    def test_label_fields_remain_diagnostics(self):
        index = self.convert()
        with h5py.File(self.cache / index["episodes"]["train"][0]["path"], "r") as handle:
            self.assertIn("class_label", handle["diagnostics"])
            self.assertIn("hover_goal_xyz", handle["diagnostics"])
            self.assertTrue(handle.attrs["complete"])
        data = EpisodeWindows(self.cache, "val")
        self.addCleanup(data.close)
        self.assertNotIn("class_label", data[0])
        self.assertNotIn("hover_goal_xyz", data[0])


if __name__ == "__main__":
    unittest.main()
