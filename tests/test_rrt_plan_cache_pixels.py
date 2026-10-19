"""rrt_branches_plan.CachePixelBlock (2026-09-28, stage-2 collection): cache_pixels.npz is written block by block.

The planner used to hold every anchor's 11 cache frames in memory before np.savez_compressed (5340 anchors = 8.2 GiB for the
stage-2 training plan, above the 7 GB job limit). It now stores CachePixelBlock references that numpy reads one at a time
through __array__. These tests show that the written file is BYTE-identical to the one written from the arrays, that the
block is re-checked at write time (completeness, source sha256) and that only one block is read at a time.
Synthetic h5 caches in a temporary directory; needs h5py (training venv), skipped elsewhere.
"""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import random_branches_plan as rb
import rrt_branches_plan as rp

try:
    import h5py
except ImportError:                                         # simulation venv
    h5py = None

FRAMES = 30


def write_cache_episode(path, seed, sha="npz-sha", complete=True):
    pixels = np.random.default_rng(seed).integers(0, 256, size=(FRAMES, 224, 224, 3), dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f.attrs["complete"] = complete
        f.attrs["source_npz_sha256"] = sha
        f.create_dataset("pixels", data=pixels, chunks=(1, 224, 224, 3), compression="lzf")
    return pixels


@unittest.skipIf(h5py is None, "needs h5py (training venv)")
class CachePixelBlockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.episodes = {i: write_cache_episode(self.root / f"episode_{i}.h5", seed=i, sha=f"sha{i}") for i in range(3)}
        # (episode, anchor frame) in plan order, one episode used twice like two anchors of one trajectory
        self.anchors = [(0, 12), (1, 20), (0, 29), (2, 15)]

    def tearDown(self):
        self.tmp.cleanup()

    def refs(self):
        return {f"a{i:04d}": rp.CachePixelBlock(self.root / f"episode_{e}.h5", frame, f"sha{e}") for i, (e, frame) in enumerate(self.anchors)}

    def eager(self):
        return {f"a{i:04d}": rb.read_cache_pixels(self.root / f"episode_{e}.h5", frame, f"sha{e}")[0] for i, (e, frame) in enumerate(self.anchors)}

    def test_npz_bytes_identical_to_the_eager_write(self):
        lazy, eager = io.BytesIO(), io.BytesIO()
        np.savez_compressed(lazy, **self.refs())
        np.savez_compressed(eager, **self.eager())
        self.assertEqual(lazy.getvalue(), eager.getvalue())
        lazy.seek(0)
        with np.load(lazy, allow_pickle=False) as z:
            self.assertEqual(list(z.keys()), [f"a{i:04d}" for i in range(len(self.anchors))])
            for i, (e, frame) in enumerate(self.anchors):
                block = z[f"a{i:04d}"]
                self.assertEqual((block.shape, block.dtype), ((rp.PREFIX + 1, 224, 224, 3), np.uint8))
                np.testing.assert_array_equal(block, self.episodes[e][frame - rp.PREFIX:frame + 1])

    def test_block_is_rechecked_at_write_time(self):
        refs = self.refs()
        with h5py.File(self.root / "episode_1.h5", "r+") as f:  # the cache episode changed after planning
            f.attrs["source_npz_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "different trajectory.npz"):
            np.savez_compressed(io.BytesIO(), **refs)
        with h5py.File(self.root / "episode_1.h5", "r+") as f:
            f.attrs["source_npz_sha256"] = "sha1"
            f.attrs["complete"] = False
        with self.assertRaisesRegex(ValueError, "not complete"):
            np.savez_compressed(io.BytesIO(), **refs)

    def test_one_block_at_a_time(self):
        live, peak = [0], [0]
        real = rb.read_cache_pixels

        class Tracked(np.ndarray):
            def __del__(self):
                live[0] -= 1

        def tracked(*args):
            block, n = real(*args)
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            return block.view(Tracked), n

        with patch.object(rb, "read_cache_pixels", tracked):
            np.savez_compressed(io.BytesIO(), **self.refs())
        self.assertEqual(peak[0], 1)

    def test_array_protocol(self):
        ref = rp.CachePixelBlock(self.root / "episode_2.h5", 15, "sha2")
        np.testing.assert_array_equal(np.asarray(ref), self.episodes[2][5:16])
        self.assertEqual(np.asarray(ref, dtype=np.float32).dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
