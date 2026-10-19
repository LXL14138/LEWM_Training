"""Parallel/resume tests on tiny temporary fixtures, not the real dataset."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image

from wm_training.dataset import EpisodeWindows
from wm_training.prepare_data import prepare, sha256
from wm_training import prepare_full as full
from wm_training.tests.test_data import episode_fixture, write_fixture


class ParallelPrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "raw"
        self.output = self.root / "parallel"
        self.manifest = write_fixture(self.source)

    def run_prepare(self, output=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return full.prepare_parallel(self.source, output or self.output, seed=17, image_size=28,
                                         train_episodes=4, val_episodes=1, **kwargs)

    def interrupt_after_one(self):
        real = full.work_episode
        calls = 0
        def limited(job):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("test interruption")
            return real(job)
        with patch.object(full, "work_episode", limited):
            with self.assertRaises(KeyboardInterrupt):
                self.run_prepare(workers=1)

    def test_parallel_content_matches_serial_and_pilot(self):
        with redirect_stdout(io.StringIO()):
            pilot = prepare(self.source, self.root / "pilot", train_episodes=4, val_episodes=1,
                            seed=17, image_size=28)
        serial = self.run_prepare(self.root / "serial", workers=1)
        parallel = self.run_prepare(workers=2)
        self.assertEqual(serial["normalization"], parallel["normalization"])
        self.assertEqual(pilot["normalization"], parallel["normalization"])
        self.assertEqual(parallel["execution"]["origins"], {"converted": 5})
        for split in ("train", "val"):
            self.assertEqual(serial["selected_ids"][split], parallel["selected_ids"][split])
            for row in parallel["episodes"][split]:
                with h5py.File(self.root / "pilot" / row["path"], "r") as a, \
                        h5py.File(self.output / row["path"], "r") as b:
                    for key in ("pixels", "action"):
                        np.testing.assert_array_equal(a[key][:], b[key][:])
                    self.assertEqual(a.attrs["pixels_sha256"], b.attrs["pixels_sha256"])
        reader = EpisodeWindows(self.output, "train", 3)
        self.addCleanup(reader.close)
        self.assertEqual(len(reader), 12)

    def test_resume_preserves_completed_files_and_ignores_partials(self):
        self.interrupt_after_one()
        completed = list((self.output / "episodes").glob("*.h5"))
        self.assertEqual(len(completed), 1)
        self.assertFalse((self.output / "index.json").exists())
        before_hash, before_mtime = sha256(completed[0]), completed[0].stat().st_mtime_ns
        leftover = self.output / "partial" / "unfinished.h5"
        leftover.write_bytes(b"not a complete HDF5")
        result = self.run_prepare(workers=2, resume=True)
        self.assertEqual(result["execution"]["origins"], {"verified-existing": 1, "converted": 4})
        self.assertEqual(sha256(completed[0]), before_hash)
        self.assertEqual(completed[0].stat().st_mtime_ns, before_mtime)
        self.assertEqual(leftover.read_bytes(), b"not a complete HDF5")
        self.assertEqual(result["normalization"]["num_actions"], 20)
        with patch.object(full, "work_episode", side_effect=AssertionError("must not start")):
            again = self.run_prepare(workers=1, resume=True)
        self.assertEqual(again, result)

    def test_copy_reuse_keeps_original_cache_unchanged(self):
        pilot_root = self.root / "pilot"
        with redirect_stdout(io.StringIO()):
            pilot = prepare(self.source, pilot_root, train_episodes=4, val_episodes=1,
                            seed=17, image_size=28)
        before = {str(p.relative_to(pilot_root)): sha256(p) for p in pilot_root.rglob("*") if p.is_file()}
        result = self.run_prepare(workers=2, reuse_cache=pilot_root)
        after = {str(p.relative_to(pilot_root)): sha256(p) for p in pilot_root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result["execution"]["origins"], {"reused-pilot": 5})
        self.assertEqual(pilot["normalization"], result["normalization"])
        row = result["episodes"]["train"][0]
        with h5py.File(self.output / row["path"], "r+") as handle:
            handle["pixels"][0, 0, 0, 0] = 0
        self.assertEqual(sha256(pilot_root / row["path"]), before[row["path"]])

    def test_bad_cached_pixels_are_rejected_not_overwritten(self):
        self.interrupt_after_one()
        cached = next((self.output / "episodes").glob("*.h5"))
        with h5py.File(cached, "r+") as handle:
            old = int(handle["pixels"][0, 0, 0, 0])
            handle["pixels"][0, 0, 0, 0] = (old + 1) % 256
        before = sha256(cached)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_prepare(workers=1, resume=True)
        self.assertEqual(sha256(cached), before)
        self.assertFalse((self.output / "index.json").exists())

    def test_changed_resume_plan_and_implicit_overwrite_rejected(self):
        self.interrupt_after_one()
        with self.assertRaises(FileExistsError):
            self.run_prepare(workers=1)
        with self.assertRaisesRegex(ValueError, "plan/source/code"):
            with redirect_stdout(io.StringIO()):
                full.prepare_parallel(self.source, self.output, workers=1, resume=True,
                                      seed=17, image_size=42, train_episodes=4, val_episodes=1)

    def test_same_directory_cannot_have_two_writers(self):
        self.output.mkdir()
        with full.run_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "Another converter"):
                with full.run_lock(self.output):
                    self.fail("second writer acquired lock")
        with full.run_lock(self.output):
            pass  # leftover lock file is harmless after the owner releases it

    def test_full_split_test_is_converted_but_not_fitted_or_trained(self):
        # Populate all 20 synthetic episodes, including reserved test/unused train.
        for row in self.manifest["episodes"]:
            episode = self.source / row["path"]
            if episode.exists():
                continue
            (episode / "images").mkdir(parents=True)
            meta, arrays = episode_fixture(row["id"])
            (episode / "metadata.json").write_text(json.dumps(meta))
            np.savez_compressed(episode / "trajectory.npz", **arrays)
            for i in range(6):
                Image.new("RGB", (16, 12), (i * 30, row["id"] * 7 % 256, 20)).save(
                    episode / "images" / f"{i:06d}.png")
        result = self.run_prepare(workers=2, full=True)
        self.assertEqual({k: len(v) for k, v in result["episodes"].items()}, {"train": 18, "val": 1, "test": 1})
        self.assertFalse(result["reserved_test_not_read"])
        self.assertFalse(result["test_used_for_training"])
        self.assertEqual(result["normalization"]["num_actions"], 18 * 5)
        actions = np.concatenate([episode_fixture(i)[1]["actions"][:, [0, 1, 2, 4]].astype(np.float32)
                                  for i in result["selected_ids"]["train"]]).astype(np.float64)
        np.testing.assert_allclose(result["normalization"]["mean"], actions.mean(0), atol=1e-8)
        with self.assertRaises(ValueError):
            EpisodeWindows(self.output, "test")

    def test_disk_full_guard_never_changes_raw_data(self):
        before = {str(p.relative_to(self.source)): sha256(p) for p in self.source.rglob("*") if p.is_file()}
        usage = type("Disk", (), {"free": 0})()
        with patch.object(full.shutil, "disk_usage", return_value=usage):
            with self.assertRaises(OSError):
                self.run_prepare(workers=1)
        after = {str(p.relative_to(self.source)): sha256(p) for p in self.source.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.output / "index.json").exists())


if __name__ == "__main__":
    unittest.main()
