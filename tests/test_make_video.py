"""make_video.py on synthetic data, no simulator: a raw episode, a counterfactual dataset with --grid, an anchor npz
(grid and single rollout) and a training-cache h5 episode, all written to a temporary directory with tiny frames.
The checks: the GIF / PNG files exist, the GIF frame counts, the refusal to overwrite and the error on an unknown path."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image

import make_video

W, H = 64, 48                                   # raw frame size of the synthetic episodes


def frames(n, w=W, h=H):
    """n frames of a flat background with a red block that moves one pixel per state (no two frames are identical)."""
    out = np.zeros((n, h, w, 3), np.uint8)
    out[...] = (90, 120, 160)
    for i in range(n):
        out[i, 10:30, 4 + i:14 + i] = (200, 40, 40)
    return out


def write_episode(folder, n, tilt=None, counterfactual=None):
    """The raw episode layout: images/%06d.png, trajectory.npz (time, guard_tilt_deg, unsafe, grasped), metadata.json."""
    folder.mkdir(parents=True)
    (folder / "images").mkdir()
    for i, frame in enumerate(frames(n)):
        Image.fromarray(frame).save(folder / "images" / f"{i:06d}.png")
    tilt = np.zeros((n, 2)) if tilt is None else np.asarray(tilt, float)
    np.savez_compressed(folder / "trajectory.npz", time=3.0 + 0.05 * np.arange(n), guard_tilt_deg=tilt,
                        unsafe=tilt.max(axis=1) > 30, grasped=np.arange(n) % 2 == 0, actions=np.zeros((n - 1, 5)))
    meta = {"status": "complete", "num_states": n, "controller": {"mode": "free"}}
    if counterfactual:
        meta["controller"]["counterfactual"] = counterfactual
        meta["outcome"] = {"counterfactual": counterfactual}
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def run(*argv):
    """make_video.main with its printed lines captured -> (return code, output)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = make_video.main([str(a) for a in argv])
    return rc, out.getvalue()


def gif_frames(path):
    with Image.open(path) as image:
        return image.n_frames, image.size


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "videos"

    def tearDown(self):
        self.tmp.cleanup()

    def test_episode_gif_and_strip(self):
        tilt = np.zeros((13, 2))
        tilt[8:, 1] = 35.0                                               # the second pillar falls at state 8
        write_episode(self.root / "exploration_x" / "episode_0003", 13, tilt)
        rc, text = run(self.root / "exploration_x" / "episode_0003", "--out", self.out)
        self.assertEqual(rc, 0, text)
        gif, strip = self.out / "exploration_x__episode_0003.gif", self.out / "exploration_x__episode_0003_strip.png"
        self.assertTrue(gif.is_file() and strip.is_file(), text)
        n, size = gif_frames(gif)
        self.assertEqual(n, 13)                                          # 13 states <= 120: every state
        self.assertEqual(size[0], W)                                     # 64 px wide frames are not scaled
        self.assertGreater(size[1], H)                                   # the label bar sits above the frame
        with Image.open(strip) as sheet:
            self.assertEqual(sheet.size, (4 * size[0], 2 * size[1]))     # 8 tiles, 4 per row
        self.assertIn("wrote", text)
        self.assertIn("2 files", text)
        rc, text = run(self.root / "exploration_x" / "episode_0003", "--out", self.out / "b", "--every", "2")
        self.assertEqual(rc, 0, text)
        self.assertEqual(gif_frames(self.out / "b" / "exploration_x__episode_0003.gif")[0], 7)   # 0, 2, ..., 12

    def test_refuses_to_overwrite_unless_forced(self):
        write_episode(self.root / "d" / "episode_0000", 5)
        self.assertEqual(run(self.root / "d" / "episode_0000", "--out", self.out)[0], 0)
        rc, text = run(self.root / "d" / "episode_0000", "--out", self.out)
        self.assertEqual(rc, 1)
        self.assertIn("exists", text)
        self.assertIn("--force", text)
        self.assertEqual(run(self.root / "d" / "episode_0000", "--out", self.out, "--force")[0], 0)

    def test_unknown_path_fails_with_one_line(self):
        (self.root / "notes.txt").write_text("x", encoding="utf-8")
        (self.root / "empty").mkdir()
        for path in (self.root / "notes.txt", self.root / "empty", self.root / "missing"):
            rc, text = run(path, "--out", self.out)
            self.assertEqual(rc, 2)
            self.assertEqual(len(text.strip().splitlines()), 1, text)
            self.assertTrue(text.startswith("FAIL make_video.py:"), text)
        self.assertFalse(self.out.exists())

    def test_dataset_limit_and_counterfactual_grid(self):
        raw = self.root / "exploration_y"
        for i in range(4):
            write_episode(raw / f"episode_{i:04d}", 6)
        (raw / "manifest.json").write_text(json.dumps({"status": "complete", "interaction_revision": 2, "episodes": [
            {"id": i, "path": f"episode_{i:04d}", "status": "complete"} for i in range(4)]}), encoding="utf-8")
        rc, text = run(raw, "--out", self.out)
        self.assertEqual(rc, 0, text)
        self.assertEqual(len(list(self.out.glob("exploration_y__episode_*.gif"))), 3)     # default --limit 3
        self.assertEqual(run(raw, "--out", self.out / "one", "--limit", "1")[0], 0)
        self.assertEqual(len(list((self.out / "one").glob("*.gif"))), 1)
        rc, text = run(raw, "--out", self.out / "g", "--grid")
        self.assertEqual(rc, 2)                                           # --grid needs a counterfactual dataset
        self.assertIn("counterfactual", text)

        cf = self.root / "cf_branches_z"
        rows, n = [], 0
        for branch, lead, length in (("push_back", 4, 13), ("hold", 4, 10), ("hold", 8, 9)):
            rec = {"branch": branch, "lead": lead, "prefix_frames": 5, "restore_frame": 20 - lead,
                   "source_dataset": 0, "source_episode_id": 5, "onset_frame": 20}
            write_episode(cf / f"episode_{n:04d}", length, counterfactual=rec)
            rows.append({"id": n, "path": f"episode_{n:04d}", "status": "complete", "branch": branch, "lead": lead})
            n += 1
        (cf / "manifest.json").write_text(json.dumps({"status": "complete", "interaction_revision": 4,
                                                      "counterfactual_branches": {}, "episodes": rows}), encoding="utf-8")
        rc, text = run(cf, "--out", self.out, "--grid", "--all")
        self.assertEqual(rc, 0, text)
        lead4, lead8 = self.out / "cf_branches_z__src0-0005_lead04_grid.gif", self.out / "cf_branches_z__src0-0005_lead08_grid.gif"
        self.assertTrue(lead4.is_file() and lead8.is_file(), text)
        self.assertEqual(gif_frames(lead4), (13, (2 * W, gif_frames(lead8)[1][1])))   # hold | push_back, longest branch
        self.assertEqual(gif_frames(lead8)[0], 9)
        self.assertFalse(list(self.out.glob("cf_branches_z__*.png")))                   # grids have no contact sheet
        rc, text = run(cf, "--out", self.out / "single", "--limit", "1")                 # without --grid: one episode
        self.assertEqual(rc, 0, text)
        self.assertTrue((self.out / "single" / "cf_branches_z__episode_0000_strip.png").is_file())


class CacheFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "videos"

    def tearDown(self):
        self.tmp.cleanup()

    def test_anchor_grid_and_single_rollout(self):
        R, S, size = 5, 6, 32
        pixels = np.stack([frames(S, size, size) for _ in range(R)])            # (R, S, 32, 32, 3)
        tilt = np.zeros((R, S, 2))
        tilt[2, 3:, 0] = 40.0
        branch = self.root / "rrt_branches_w"
        branch.mkdir()
        np.savez_compressed(branch / "anchor_0007.npz", pixels=pixels, guard_tilt_deg=tilt, unsafe=tilt.max(-1) > 30,
                            grasped=np.ones((R, S), bool), class_label=np.full((R, S), 2, np.int8),
                            families=np.array(["recorded", "hold", "edge", "edge", "guard_push"]), anchor_id=np.int64(7))
        rc, text = run(branch / "anchor_0007.npz", "--out", self.out)
        self.assertEqual(rc, 0, text)
        n, (w, h) = gif_frames(self.out / "rrt_branches_w__anchor_0007.gif")
        self.assertEqual((n, w), (S, 4 * size))                                  # 4 panels per row, 2 rows
        self.assertGreater(h, 2 * size)
        rc, text = run(branch / "anchor_0007.npz", "--out", self.out, "--rollout", "2")
        self.assertEqual(rc, 0, text)
        self.assertEqual(gif_frames(self.out / "rrt_branches_w__anchor_0007_r2.gif")[0], S)
        self.assertTrue((self.out / "rrt_branches_w__anchor_0007_r2_strip.png").is_file())
        self.assertEqual(run(branch / "anchor_0007.npz", "--out", self.out, "--rollout", "9")[0], 2)
        # the dataset directory of anchors: manifest + anchor_*.npz -> grids, --limit applies
        (branch / "manifest.json").write_text(json.dumps({"format": "rrt_branches_v1", "anchors": [{"id": 7}]}), encoding="utf-8")
        rc, text = run(branch, "--out", self.out / "ds")
        self.assertEqual(rc, 0, text)
        self.assertEqual(gif_frames(self.out / "ds" / "rrt_branches_w__anchor_0007.gif")[0], S)

    def test_h5_episode_with_and_without_diagnostics(self):
        cache = self.root / "wm_full_v" / "episodes"
        cache.mkdir(parents=True)
        pixels = frames(7, 32, 32)
        with h5py.File(cache / "episode_0000012.h5", "w") as f:
            f.create_dataset("pixels", data=pixels)
            f.create_dataset("action", data=np.zeros((6, 4), np.float32))
            diag = f.create_group("diagnostics")
            tilt = np.zeros((7, 2))
            tilt[4:, 0] = 33.0
            diag.create_dataset("guard_tilt_deg", data=tilt)
            diag.create_dataset("unsafe", data=tilt.max(1) > 30)
            diag.create_dataset("time", data=0.05 * np.arange(7))
        with h5py.File(cache / "bare.h5", "w") as f:
            f.create_dataset("pixels", data=pixels)
        rc, text = run(cache / "episode_0000012.h5", "--out", self.out)
        self.assertEqual(rc, 0, text)
        self.assertEqual(gif_frames(self.out / "wm_full_v__episode_0000012.gif")[0], 7)   # named after the cache, not 'episodes'
        self.assertTrue((self.out / "wm_full_v__episode_0000012_strip.png").is_file())
        rc, text = run(cache / "bare.h5", "--out", self.out)
        self.assertEqual(rc, 0, text)
        self.assertEqual(gif_frames(self.out / "wm_full_v__bare.gif")[0], 7)


if __name__ == "__main__":
    unittest.main()
