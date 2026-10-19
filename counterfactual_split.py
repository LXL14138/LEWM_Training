"""Split file (danger_bench_split_v1 layout) for counterfactual BRANCH datasets, by MEASURED outcome. numpy only.

wm_training.prepare_bench converts raw datasets through a split file, and benchmark_split.py refuses
branch datasets (they carry no benchmark intent). This helper reads one or more datasets written by
collect_counterfactual_branches.py, validates every branch record against its arrays, measures the
outcome of each branch with the WP0 ruler (wm_training.danger_strata via benchmark_measure.measured_cell)
and writes a NEW directory with benchmark_split.json + REPORT_ZH.md:

* part: every branch episode goes to the part of its SOURCE event (``test`` = the fine-tune training
  half, ``dev`` = the validation half); nothing is shuffled, no seed is drawn;
* stratum: the measured ``direction|severity`` cell when the branch crossed 30 degrees, otherwise
  ``no_event|<branch>`` (the branch avoided the fall).

Datasets are never modified; images are never read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from benchmark_measure import measured_cell
from benchmark_split import SPLIT_FORMAT
from counterfactual_plan import BRANCHES, CF_FORMAT, CF_INTERACTION_REVISION, validate_branch_record
from wm_training.danger_strata import episode_features, load_episode_arrays, native, read_json, sha256

# Must equal wm_training.prepare_bench.BRANCH_MEASURE_FORMAT (prepare_bench reads the per-episode records of this split
# into the cache index); kept as a literal here so this script stays numpy-only (the simulator venv has no h5py).
MEASURE_FORMAT = "counterfactual_branch_measure_v1"


def load_branch_datasets(paths, allow_interrupted=False):
    """Datasets and every complete branch episode with its measured outcome; strict about the records."""
    datasets, episodes = [], []
    for index, path in enumerate(paths):
        path = Path(path).resolve()
        manifest = read_json(path / "manifest.json")
        acceptable = ("complete", "interrupted") if allow_interrupted else ("complete",)
        record = manifest.get("counterfactual_branches")
        if (manifest.get("status") not in acceptable or manifest.get("interaction_revision", 1) < CF_INTERACTION_REVISION
                or not isinstance(record, dict) or record.get("format") != CF_FORMAT or not record.get("sha256")):
            raise ValueError(f"{path}: needs a complete counterfactual branch dataset (interaction_revision 4 with a "
                             f"counterfactual_branches record; interrupted only with --allow-interrupted)")
        rows = [r for r in manifest["episodes"] if r.get("status") == "complete"]
        if len(rows) != manifest["requested_episodes"] and not allow_interrupted:
            raise ValueError(f"{path}: {len(rows)} complete episodes of {manifest['requested_episodes']} requested")
        if not rows:
            raise ValueError(f"{path}: no complete episodes")
        datasets.append({"index": index, "path": str(path), "manifest_sha256": sha256(path / "manifest.json"),
                         "schedule_sha256": record["sha256"], "provenance": "counterfactual_branches", "seed": manifest.get("seed"),
                         "episodes": len(rows), "status": manifest.get("status"), "requested_episodes": manifest.get("requested_episodes"),
                         "source_datasets": record.get("source_datasets"), "leads": record.get("leads"), "branches": record.get("branches"),
                         "hold_steps": record.get("hold_steps"), "prefix_frames": record.get("prefix_frames")})
        for row in rows:
            folder = path / row["path"]
            meta = read_json(folder / "metadata.json")
            controller = meta.get("controller", {})
            if meta.get("status") != "complete" or controller.get("interaction_revision") != CF_INTERACTION_REVISION:
                raise ValueError(f"{folder}: not a complete counterfactual branch episode")
            cf = validate_branch_record(meta["outcome"]["counterfactual"], controller.get("counterfactual"))
            features = episode_features(load_episode_arrays(folder), float(meta["control_dt"]))
            if features["n"] != meta["num_states"] or features["n"] != cf["prefix_frames"] + cf["hold_steps"] + 1:
                raise ValueError(f"{folder}: state count disagrees with the branch record")
            cell, detail = measured_cell(features)
            stratum = cell if cell != "no_event" else f"no_event|{cf['branch']}"
            episodes.append({"dataset": index, "episode_id": int(row["id"]), "path": row["path"], "stratum": stratum,
                             "measured_cell": cell, "part": cf["source_part"], "branch": cf["branch"], "lead": cf["lead"],
                             "restore_frame": cf["restore_frame"], "prefix_frames": cf["prefix_frames"], "hold_steps": cf["hold_steps"],
                             "active_steps": cf["active_steps"], "source_dataset": cf["source_dataset"],
                             "source_episode_id": cf["source_episode_id"], "source_path": cf["source_path"],
                             "source_stratum": cf["source_stratum"], "source_onset_frame": cf["onset_frame"],
                             "frames": int(features["n"]), "onset_frame": detail.get("onset_frame"),
                             "reached_unsafe": cf["reached_unsafe"], "first_unsafe_step": cf["first_unsafe_step"],
                             "max_tilt_deg": cf["max_tilt_deg"], "replay_exact": cf["replay_exact"],
                             "stop_reason": meta["outcome"]["stop_reason"],
                             **{k: v for k, v in detail.items() if k != "onset_frame"}})
    seen = Counter((e["dataset"], e["episode_id"]) for e in episodes)
    if any(v > 1 for v in seen.values()):
        raise ValueError("Duplicate episode ids inside a dataset")
    return datasets, episodes


def split_by_source_part(episodes):
    """{stratum: {dev: [...], test: [...]}} from each branch episode's SOURCE part; nothing is drawn."""
    strata = defaultdict(lambda: {"dev": [], "test": []})
    for e in sorted(episodes, key=lambda e: (e["dataset"], e["episode_id"])):
        if e["part"] not in ("dev", "test"):
            raise ValueError(f"Unknown source part {e['part']}")
        strata[e["stratum"]][e["part"]].append({"dataset": e["dataset"], "episode_id": e["episode_id"]})
    return {name: strata[name] for name in sorted(strata)}


def outcome_table(episodes):
    """reached_unsafe fraction per branch x lead (what the branches actually did)."""
    table = {}
    for branch in BRANCHES:
        rows = [e for e in episodes if e["branch"] == branch]
        if not rows:
            continue
        table[branch] = {}
        for lead in sorted({e["lead"] for e in rows}):
            sel = [e for e in rows if e["lead"] == lead]
            table[branch][str(lead)] = {"episodes": len(sel), "reached_unsafe": int(sum(e["reached_unsafe"] for e in sel)),
                                        "fraction_unsafe": float(np.mean([e["reached_unsafe"] for e in sel])),
                                        "max_tilt_deg_median": float(np.median([e["max_tilt_deg"] for e in sel]))}
    return table


def write_report(output, result):
    lines = ["# 反事实分支数据集切分（按分支实测结果分层）\n",
             f"- 数据集：{'; '.join(d['path'] for d in result['datasets'])}（共 {len(result['episodes'])} 条分支轨迹）",
             "- 每条分支轨迹进入其**来源事件**所在的部分（test = 微调训练半、dev = 验证半），不洗牌、不抽样。",
             "- 层 = 分支实测结果：越过 30° 时为 `方向|严重程度`（`wm_training.danger_strata`），没越过为 `no_event|<分支>`。",
             f"- 来源数据集只含训练材料（见各 manifest 的 `counterfactual_branches.source_datasets`）；没有用任何基准轨迹。\n",
             "## 每层数量\n", "| 层 | dev | test |", "|---|---|---|"]
    for name, part in result["strata"].items():
        cell = name.replace("|", "\\|")          # stratum names contain '|', which would split a Markdown cell
        lines.append(f"| {cell} | {len(part['dev'])} | {len(part['test'])} |")
    lines += ["", "## 分支 × 提前量：越过 30° 的比例\n", "| 分支 | 提前量 k | 条数 | 越过 30° | 比例 | 最大倾角中位(°) |", "|---|---|---|---|---|---|"]
    for branch, leads in result["outcome_by_branch_lead"].items():
        for lead, cell in leads.items():
            lines.append(f"| {branch} | {lead} | {cell['episodes']} | {cell['reached_unsafe']} | {cell['fraction_unsafe']:.2f} | {cell['max_tilt_deg_median']:.1f} |")
    inexact = [e for e in result["episodes"] if not e["replay_exact"]]
    lines += ["", f"- 前缀重放与记录不完全一致的轨迹：{len(inexact)}。", f"- 用时 {result['elapsed_seconds']:.1f} s。"]
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, action="append", required=True, help="counterfactual branch dataset directory; repeatable")
    p.add_argument("--allow-interrupted", action="store_true", help="accept an interrupted generation and use its complete episodes")
    p.add_argument("--output", type=Path, required=True, help="NEW directory")
    a = p.parse_args(argv)
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    for d in a.dataset:
        if output.is_relative_to(d.resolve()):
            raise ValueError("Output must be outside the datasets")
    started = time.perf_counter()
    datasets, episodes = load_branch_datasets(a.dataset, a.allow_interrupted)
    strata = split_by_source_part(episodes)
    result = {"format": SPLIT_FORMAT, "measure_format": MEASURE_FORMAT, "seed": None, "dev_fraction": None,
              "allow_interrupted": a.allow_interrupted, "quota_per_cell": None, "datasets": datasets, "strata": strata,
              "quota": [], "composition": {name: {"branch": dict(Counter(e["branch"] for e in episodes if e["stratum"] == name).most_common()),
                                                  "lead": dict(Counter(str(e["lead"]) for e in episodes if e["stratum"] == name).most_common())}
                                           for name in strata},
              "outcome_by_branch_lead": outcome_table(episodes), "episodes": episodes,
              "rule": "stratum = measured direction|severity of the branch (wm_training.danger_strata) or no_event|<branch>; "
                      "part = the source event's part of the training split (no shuffle); every episode is training material",
              "test_pool_read": False, "images_read": False, "elapsed_seconds": time.perf_counter() - started}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "benchmark_split.json").open("x", encoding="utf-8") as stream:
        json.dump(native(result), stream, ensure_ascii=False, indent=1, allow_nan=False)
        stream.write("\n")
    write_report(output, result)
    for branch, leads in result["outcome_by_branch_lead"].items():
        print(branch, " ".join(f"k{lead}:{cell['reached_unsafe']}/{cell['episodes']}" for lead, cell in leads.items()))
    print(f"Complete: {output / 'benchmark_split.json'} ({len(episodes)} branch episodes, {len(strata)} strata)")


if __name__ == "__main__":
    main()
