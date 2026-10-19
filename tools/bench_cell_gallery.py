"""Copy a few recorded frames per stratum for visual inspection. numpy only, read-only on the datasets.

For every stratum in a benchmark_split.json, picks up to --per-cell episodes
with a fixed seed and copies the ORIGINAL PNGs (no re-rendering, no overlays)
of the frames: five before onset, the onset, ten after onset and the peak-tilt
frame (event episodes), or the first frame, the peak-tilt frame and the last
frame (negatives). Writes them under a NEW directory as
<stratum>/<dataset>_ep<id>_<tag>_f<frame>.png with an index.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wm_training.danger_strata import load_episode_arrays, read_json  # noqa: E402


def frame_tags(arrays, onset):
    n = len(arrays["unsafe"])
    tilt = arrays["guard_tilt_deg"].max(1)
    peak = int(np.argmax(tilt))
    if onset is None:
        return [("first", 0), ("peak", peak), ("last", n - 1)]
    return [("onset-5", max(0, onset - 5)), ("onset", onset), ("onset+10", min(n - 1, onset + 10)), ("peak", peak)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True, help="benchmark_split.json")
    p.add_argument("--per-cell", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True, help="NEW directory")
    a = p.parse_args(argv)
    if a.per_cell < 1:
        raise ValueError("per-cell must be positive")
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    split = read_json(a.split)
    datasets = [Path(d["path"]) for d in split["datasets"]]
    for d in datasets:
        if output.is_relative_to(d.resolve()):
            raise ValueError("Output must be outside the datasets")
    rng = np.random.default_rng(a.seed)
    episodes = {(e["dataset"], e["episode_id"]): e for e in split["episodes"]}
    output.mkdir(parents=True, exist_ok=False)
    lines = [f"# 每层抽样帧（split {a.split.name}，seed {a.seed}，每层最多 {a.per_cell} 条）\n",
             "原始 PNG 直接复制，没有重新渲染或叠加标注。文件名：数据集编号_ep编号_帧标签_f帧号。\n"]
    copied = 0
    for name in sorted(split["strata"]):
        members = split["strata"][name]["dev"] + split["strata"][name]["test"]
        if not members:
            continue
        order = rng.permutation(len(members))[:a.per_cell]
        folder = output / name.replace("|", "__")
        folder.mkdir()
        lines.append(f"\n## {name}（共 {len(members)} 条，抽 {len(order)} 条）\n")
        for i in order:
            ref = members[i]
            info = episodes[(ref["dataset"], ref["episode_id"])]
            episode_dir = datasets[ref["dataset"]] / info["path"]
            arrays = load_episode_arrays(episode_dir)
            tags = frame_tags(arrays, info.get("onset_frame"))
            names = []
            for tag, frame in tags:
                src = episode_dir / "images" / f"{frame:06d}.png"
                dst = folder / f"{ref['dataset']}_ep{ref['episode_id']:04d}_{tag}_f{frame:06d}.png"
                shutil.copyfile(src, dst)
                names.append(dst.name)
                copied += 1
            lines.append(f"- ep{ref['episode_id']:04d}（{info['recipe']}，柱 {info['guard_index']}，{info['after_event']}，"
                         f"峰值 {info.get('peak_tilt_deg', info.get('max_tilt_deg', 0)):.1f}°）：" + "，".join(names))
    (output / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "gallery.json").open("x", encoding="utf-8") as stream:
        json.dump({"split": str(a.split.resolve()), "seed": a.seed, "per_cell": a.per_cell, "frames_copied": copied}, stream, indent=2)
    print(f"Complete: {output / 'index.md'} ({copied} frames copied)")


if __name__ == "__main__":
    main()
