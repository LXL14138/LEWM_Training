"""Intended-vs-measured cell table for a danger-benchmark dataset. numpy only, read-only.

Reads each episode's scheduled intent and controller record from metadata.json
and the recorded geometry from trajectory.npz, then classifies the reached
stratum with wm_training.danger_strata (the WP0 ruler). Writes a NEW directory
with precheck_report.json and REPORT_ZH.md. Never reads images.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_measure import measured_cell  # noqa: E402  (shared with benchmark_split.py)
from benchmark_plan import ALL_CELLS  # noqa: E402
from wm_training.danger_strata import episode_features, load_episode_arrays, native, read_json, sha256  # noqa: E402


def analyse(dataset):
    dataset = Path(dataset).resolve()
    manifest = read_json(dataset / "manifest.json")
    if manifest.get("interaction_revision", 1) < 3 or not manifest.get("benchmark_schedule"):
        raise ValueError("Not a danger-benchmark dataset (needs interaction_revision 3 and a schedule)")
    rows, by_intent, by_recipe = [], defaultdict(Counter), defaultdict(Counter)
    for entry in manifest["episodes"]:
        if entry.get("status") != "complete":
            continue
        folder = dataset / entry["path"]
        meta = read_json(folder / "metadata.json")
        intent = meta["controller"]["benchmark_intent"]
        record = meta["outcome"]["benchmark"]
        features = episode_features(load_episode_arrays(folder), float(meta["control_dt"]))
        cell, detail = measured_cell(features)
        row = {"episode_id": entry["id"], "intent_cell": intent["cell"], "recipe": intent["recipe"], "guard_index": intent["guard_index"],
               "lean": intent["lean"], "after_event": intent["after_event"], "measured_cell": cell, **detail,
               "hazard_step": record["hazard_step"], "hazard_phase": record["hazard_phase"],
               "hazard_guard_index": record.get("hazard_guard_index"), "budget_short": record.get("budget_short"),
               "stop_rule_reached": record["stop_rule_reached"], "end_reason": record["end_reason"],
               "stop_reason": meta["outcome"]["stop_reason"], "frames": int(features["n"]),
               "ever_lifted": meta["outcome"].get("ever_lifted")}
        rows.append(row)
        by_intent[intent["cell"]][cell] += 1
        by_recipe[intent["recipe"]]["attempts"] += 1
        # "hazard" is measured (any unsafe frame); a hazard latched on the very last action has no hazard_step.
        by_recipe[intent["recipe"]]["hazard"] += features["event"] is not None
        by_recipe[intent["recipe"]]["hazard_recorded"] += record["hazard_step"] is not None
        by_recipe[intent["recipe"]]["budget_short"] += bool(record.get("budget_short"))
        by_recipe[intent["recipe"]]["stop_rule_reached"] += bool(record["stop_rule_reached"])
        by_recipe[intent["recipe"]][f"end:{record['end_reason'] or meta['outcome']['stop_reason']}"] += 1
    if not rows:
        raise ValueError("No complete episodes")
    hit = {}
    for cell in ALL_CELLS:
        attempts = sum(by_intent[cell].values())
        if not attempts:
            continue
        if cell.startswith("negative|"):
            wanted = sum(n for c, n in by_intent[cell].items() if c == "no_event")
        else:
            wanted = by_intent[cell].get(cell, 0)
        hit[cell] = {"attempts": attempts, "hit": wanted, "rate": wanted / attempts,
                     "measured": dict(sorted(by_intent[cell].items(), key=lambda kv: -kv[1]))}
    return {"dataset": str(dataset), "manifest_status": manifest.get("status"), "manifest_sha256": sha256(dataset / "manifest.json"),
            "schedule": manifest["benchmark_schedule"], "episodes_complete": len(rows),
            "episodes_requested": manifest.get("requested_episodes"), "hit_by_intent_cell": hit,
            "by_recipe": {k: dict(v) for k, v in by_recipe.items()}, "rows": rows,
            "note": "cells are measured with wm_training.danger_strata; intents are only attempts"}


def write_report(output, result):
    lines = ["# 危险测试集预检：意图格子 vs 测得格子\n",
             f"- 数据集：`{result['dataset']}`（manifest 状态 {result['manifest_status']}；完整轨迹 {result['episodes_complete']}/{result['episodes_requested']}）",
             f"- schedule：seed {result['schedule']['seed']}，sha256 `{result['schedule']['sha256'][:12]}…`",
             "- 测得格子用 WP0 的 `danger_strata` 口径：方向 = 主导方向；S1 = 峰值 ≥85°；S2/S3 = 斜靠（峰值 <60° 且连续超限 ≥10 帧）按峰值 ≥40 / <40 分；`transient` = 超限但既没倒平也没持续；`no_event` = 全程没超限。\n",
             "## 每个意图格子的命中率\n", "| 意图格子 | 尝试 | 命中 | 命中率 | 测得分布 |", "|---|---|---|---|---|"]
    for cell, h in result["hit_by_intent_cell"].items():
        dist = "；".join(f"{k}×{v}" for k, v in h["measured"].items())
        lines.append(f"| {cell} | {h['attempts']} | {h['hit']} | {100 * h['rate']:.0f}% | {dist} |")
    lines += ["\n## 每种配方\n", "| 配方 | 尝试 | 测得危险 | 控制器记到危险 | 预算不足 | 停止规则达成 | 结束原因 |", "|---|---|---|---|---|---|---|"]
    for recipe, v in result["by_recipe"].items():
        ends = "；".join(f"{k[4:]}×{n}" for k, n in v.items() if k.startswith("end:"))
        lines.append(f"| {recipe} | {v['attempts']} | {v['hazard']} | {v['hazard_recorded']} | {v['budget_short']} | {v['stop_rule_reached']} | {ends} |")
    lines += ["\n## 逐条\n", "| 轨迹 | 意图 | 配方 | 柱 | 方向 | 事件后 | 测得 | 峰值/最大倾角 | 危险步 | 结束 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        peak = r.get("peak_tilt_deg", r.get("max_tilt_deg"))
        lines.append(f"| {r['episode_id']:04d} | {r['intent_cell']} | {r['recipe']} | {r['guard_index']} | {r['lean']} | {r['after_event']} | "
                     f"{r['measured_cell']} | {peak:.1f}° | {r['hazard_step']} | {r['end_reason'] or r['stop_reason']} |")
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True, help="NEW directory")
    a = p.parse_args(argv)
    output = a.output.resolve()
    if output.exists() or output.is_relative_to(a.dataset.resolve()):
        raise FileExistsError("Use a NEW output directory outside the dataset")
    result = analyse(a.dataset)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "precheck_report.json").open("x", encoding="utf-8") as stream:
        json.dump(native(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
    write_report(output, result)
    for cell, h in result["hit_by_intent_cell"].items():
        print(f"{cell:<32} {h['hit']}/{h['attempts']}  measured={h['measured']}")
    print(f"Complete: {output / 'REPORT_ZH.md'}")


if __name__ == "__main__":
    main()
