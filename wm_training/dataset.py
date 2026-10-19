"""Lazy read-only episode windows. No simulator, labels, goals or truth inputs."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from wm_training.prepare_data import FORMAT, contained_path, read_json


class EpisodeWindows(Dataset):
    """Four consecutive images + three aligned actions when history_size=3.

    Each item lies entirely in one episode. Pixels are encoded independently;
    only the first three encodings/actions feed the causal predictor. The last
    image is a training target, never an input to that predictor.
    """

    def __init__(self, cache, split, history_size=3, stride=1, max_open_files=4):
        self.root = Path(cache).expanduser().resolve()
        if split not in ("train", "val"):
            raise ValueError("This pilot reader deliberately does not expose test episodes")
        if history_size < 1 or stride < 1 or max_open_files < 1:
            raise ValueError("history_size, stride and max_open_files must be positive")
        self.index = read_json(self.root / "index.json")
        index = self.index
        if index.get("format") != FORMAT or index.get("status") != "complete":
            raise ValueError("Incomplete/unrecognized cache")
        if index.get("frameskip") != 1 or index["source_action_columns"] != [0, 1, 2, 4]:
            raise ValueError("Unsupported action or time convention")
        pools = {k: set(v) for k, v in index["split_ids"].items()}
        if (pools["train"] & pools["val"] or pools["train"] & pools["test"] or
                pools["val"] & pools["test"]):
            raise ValueError("Episode leakage in split manifest")
        seen = set()
        for name in ("train", "val"):
            ids = [r["id"] for r in index["episodes"][name]]
            if (len(ids) != len(set(ids)) or set(ids) & seen or
                    set(ids) != set(index["selected_ids"][name]) or not set(ids) <= pools[name]):
                raise ValueError("Incorrect selected episode split")
            seen.update(ids)
        stats = index["normalization"]
        if stats["fit_split"] != "train":
            raise ValueError("Action normalization must be fitted to TRAIN only")
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.scale = np.asarray(stats["scale"], dtype=np.float32)
        if (self.mean.shape != (4,) or self.scale.shape != (4,) or
                not np.isfinite([self.mean, self.scale]).all() or np.any(self.scale <= 0)):
            raise ValueError("Invalid action statistics")
        self.pixel_mean = torch.tensor(index["pixel_mean"]).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(index["pixel_std"]).view(1, 3, 1, 1)
        self.history_size, self.stride = history_size, stride
        self.max_open_files = max_open_files
        self.rows = index["episodes"][split]
        self.ends = []
        count = 0
        for row in self.rows:
            if row["num_states"] != row["num_actions"] + 1 or row["num_actions"] < history_size:
                raise ValueError(f"Invalid/too short episode: {row['id']}")
            path = contained_path(self.root, row["path"])
            with h5py.File(path, "r") as handle:
                self._check_file(handle, row)
            count += (row["num_actions"] - history_size) // stride + 1
            self.ends.append(count)
        if not count:
            raise ValueError(f"No valid windows in {split}")
        self._handles = OrderedDict()
        self._pid = os.getpid()

    def _check_file(self, handle, row):
        size = self.index["image_size"]
        if (handle.attrs.get("format") != FORMAT or not handle.attrs.get("complete", False) or
                handle.attrs.get("episode_id") != row["id"] or
                handle["pixels"].shape != (row["num_states"], size, size, 3) or
                handle["pixels"].dtype != np.dtype("uint8") or
                handle["action"].shape != (row["num_actions"], 4)):
            raise ValueError(f"Corrupt/incomplete cache episode: {row['id']}")

    def __len__(self):
        return self.ends[-1]

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        previous = self.ends[episode - 1] if episode else 0
        return episode, (index - previous) * self.stride

    def _open(self, episode):
        if os.getpid() != self._pid:
            self.close()
            self._pid = os.getpid()
        if episode not in self._handles:
            if len(self._handles) >= self.max_open_files:
                _, handle = self._handles.popitem(last=False)
                handle.close()
            row = self.rows[episode]
            handle = h5py.File(contained_path(self.root, row["path"]), "r")
            try:
                self._check_file(handle, row)
            except BaseException:
                handle.close()
                raise
            self._handles[episode] = handle
        self._handles.move_to_end(episode)
        return self._handles[episode]

    def __getitem__(self, index):
        episode, start = self.locate(index)
        stop = start + self.history_size
        handle = self._open(episode)
        rgb = np.asarray(handle["pixels"][start:stop + 1])
        action = np.asarray(handle["action"][start:stop], dtype=np.float32)
        action = (action - self.mean) / self.scale
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite normalized action")
        pixels = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255)
        pixels = (pixels - self.pixel_mean) / self.pixel_std
        # Deliberately exclude labels, target positions, simulator qpos/qvel, etc.
        return {"pixels": pixels.contiguous(), "action": torch.from_numpy(action)}

    def close(self):
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        self._handles = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_pid"] = None
        return state

    def __del__(self):
        self.close()
