"""Fixed bench-dev / bench-test split of danger-benchmark episodes, by MEASURED stratum. numpy only.

Reads one or more complete benchmark datasets (interaction_revision 3), audits
every episode's stored labels, classifies each episode with the WP0 ruler
(benchmark_measure), then shuffles each stratum with a fixed seed and puts the
first dev_fraction into bench-dev and the rest into bench-test. Writes a NEW
directory with benchmark_split.json and REPORT_ZH.md. Datasets are never
modified. The split never looks at images, scores or models.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from benchmark_measure import MEASURE_FORMAT, load_benchmark_episode
from benchmark_plan import ALL_CELLS, HARD_CELLS, NEGATIVE_CELLS
from wm_training.danger_strata import native, read_json, sha256

SPLIT_FORMAT = "danger_bench_split_v1"
QUOTA = {"dev": 15, "test": 30}   # per hard cell and per negative class (design draft, approved 2026-09-23)


def load_datasets(paths, allow_interrupted=False):
    """allow_interrupted: accept a collection that stopped early and use only its complete episodes
    (for TRAINING material; the benchmark itself must come from complete datasets)."""
    datasets, episodes = [], []
    for index, path in enumerate(paths):
        path = Path(path).resolve()
        manifest = read_json(path / "manifest.json")
        acceptable = ("complete", "interrupted") if allow_interrupted else ("complete",)
        if (manifest.get("status") not in acceptable or manifest.get("interaction_revision", 1) < 3
                or not manifest.get("benchmark_schedule")):
            raise ValueError(f"{path}: needs a complete benchmark dataset (interaction_revision 3 with a schedule)")
        rows = [r for r in manifest["episodes"] if r.get("status") == "complete"]
        if len(rows) != manifest["requested_episodes"] and not allow_interrupted:
            raise ValueError(f"{path}: {len(rows)} complete episodes of {manifest['requested_episodes']} requested")
        if not rows:
            raise ValueError(f"{path}: no complete episodes")
        datasets.append({"index": index, "path": str(path), "manifest_sha256": sha256(path / "manifest.json"),
                         "schedule_sha256": manifest["benchmark_schedule"]["sha256"], "seed": manifest.get("seed"),
                         "episodes": len(rows), "status": manifest.get("status"),
                         "requested_episodes": manifest.get("requested_episodes")})
        for row in rows:
            info = load_benchmark_episode(path / row["path"])
            episodes.append({"dataset": index, "episode_id": int(row["id"]), "path": row["path"],
                             "stratum": info["stratum"], "measured_cell": info["measured_cell"],
                             "intent_cell": info["intent"]["cell"], "recipe": info["intent"]["recipe"],
                             "guard_index": info["intent"]["guard_index"], "lean": info["intent"]["lean"],
                             "after_event": info["intent"]["after_event"], "frames": info["frames"],
                             "hazard_step": info["record"]["hazard_step"], "stop_reason": info["stop_reason"],
                             **{k: v for k, v in info["detail"].items()}})
    seen = Counter((e["dataset"], e["episode_id"]) for e in episodes)
    if any(v > 1 for v in seen.values()):
        raise ValueError("Duplicate episode ids inside a dataset")
    return datasets, episodes


def split_episodes(episodes, seed, dev_fraction):
    """Per stratum: fixed-seed shuffle, first round(n * dev_fraction) to dev (at least one of each when n >= 2)."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be inside (0, 1)")
    rng = np.random.default_rng(seed)
    by_stratum = defaultdict(list)
    for e in episodes:
        by_stratum[e["stratum"]].append(e)
    strata = {}
    for name in sorted(by_stratum):
        members = sorted(by_stratum[name], key=lambda e: (e["dataset"], e["episode_id"]))
        order = rng.permutation(len(members))
        n_dev = int(round(len(members) * dev_fraction))
        if len(members) >= 2:
            n_dev = min(max(n_dev, 1), len(members) - 1)
        chosen = [members[i] for i in order]
        strata[name] = {"dev": [{"dataset": e["dataset"], "episode_id": e["episode_id"]} for e in chosen[:n_dev]],
                        "test": [{"dataset": e["dataset"], "episode_id": e["episode_id"]} for e in chosen[n_dev:]]}
    return strata


def quota_table(strata):
    rows = []
    for cell in list(HARD_CELLS) + list(NEGATIVE_CELLS) + [c for c in ALL_CELLS if c not in HARD_CELLS and c not in NEGATIVE_CELLS]:
        part = strata.get(cell, {"dev": [], "test": []})
        required = cell in HARD_CELLS or cell in NEGATIVE_CELLS
        rows.append({"cell": cell, "dev": len(part["dev"]), "test": len(part["test"]), "required": required,
                     "met": (len(part["dev"]) >= QUOTA["dev"] and len(part["test"]) >= QUOTA["test"]) if required else None})
    extra = sorted(set(strata) - set(ALL_CELLS))
    for name in extra:
        rows.append({"cell": name, "dev": len(strata[name]["dev"]), "test": len(strata[name]["test"]), "required": False, "met": None})
    return rows


def write_report(output, result):
    lines = ["# 危险测试集切分：bench-dev / bench-test\n",
             f"- 数据集：{'; '.join(d['path'] for d in result['datasets'])}（共 {sum(d['episodes'] for d in result['datasets'])} 条）",
             f"- seed {result['seed']}，dev 比例 {result['dev_fraction']:.4f}；层 = 测得格子（事件）或反例意图类（无事件）；每层内固定 seed 洗牌后前 round(n×比例) 条进 dev。",
             f"- 配额：硬配额格与反例类各需 dev ≥{QUOTA['dev']}、test ≥{QUOTA['test']}。未达标的格子在表里标出。",
             "- 三轮预检的数据不在这里，也不应该加进来。\n",
             "## 每层数量\n", "| 层 | dev | test | 要求 | 达标 |", "|---|---|---|---|---|"]
    for r in result["quota"]:
        mark = "" if r["met"] is None else ("是" if r["met"] else "**否**")
        lines.append(f"| {r['cell']} | {r['dev']} | {r['test']} | {'是' if r['required'] else '否'} | {mark} |")
    short = [r["cell"] for r in result["quota"] if r["met"] is False]
    lines.append("\n未达标：" + ("无" if not short else "、".join(short)) + "\n")
    lines += ["## 事件层的配方构成（支撑物 / 机理，供分层报告参考）\n", "| 层 | 配方分布 | 事件后动作分布 |", "|---|---|---|"]
    for name, comp in result["composition"].items():
        lines.append(f"| {name} | {'；'.join(f'{k}×{v}' for k, v in comp['recipe'].items())} | {'；'.join(f'{k}×{v}' for k, v in comp['after_event'].items())} |")
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, action="append", required=True, help="benchmark dataset directory; repeatable")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dev-fraction", type=float, default=1 / 3)
    p.add_argument("--allow-interrupted", action="store_true",
                   help="accept interrupted collections and use their complete episodes (training material only)")
    p.add_argument("--output", type=Path, required=True, help="NEW directory")
    a = p.parse_args(argv)
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    for d in a.dataset:
        if output.is_relative_to(d.resolve()):
            raise ValueError("Output must be outside the datasets")
    started = time.perf_counter()
    datasets, episodes = load_datasets(a.dataset, a.allow_interrupted)
    strata = split_episodes(episodes, a.seed, a.dev_fraction)
    composition = {}
    for name, part in strata.items():
        members = [e for e in episodes if e["stratum"] == name]
        composition[name] = {"recipe": dict(Counter(e["recipe"] for e in members).most_common()),
                             "after_event": dict(Counter(e["after_event"] for e in members).most_common())}
    result = {"format": SPLIT_FORMAT, "measure_format": MEASURE_FORMAT, "seed": a.seed, "dev_fraction": a.dev_fraction,
              "allow_interrupted": a.allow_interrupted,
              "quota_per_cell": QUOTA, "datasets": datasets, "strata": strata, "quota": quota_table(strata),
              "composition": composition, "episodes": episodes,
              "rule": "stratum = measured direction|severity for events (wm_training.danger_strata), intent class for negatives; "
                      "fixed-seed shuffle within stratum; bench-dev is for threshold/hyper-parameter choice only, bench-test for reporting",
              "test_pool_read": False, "images_read": False, "elapsed_seconds": time.perf_counter() - started}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "benchmark_split.json").open("x", encoding="utf-8") as stream:
        json.dump(native(result), stream, ensure_ascii=False, indent=1, allow_nan=False)
        stream.write("\n")
    write_report(output, result)
    for r in result["quota"]:
        if r["required"]:
            print(f"{r['cell']:<32} dev {r['dev']:>3} test {r['test']:>3}  {'ok' if r['met'] else 'SHORT'}")
    print(f"Complete: {output / 'benchmark_split.json'} ({len(episodes)} episodes, {len(strata)} strata)")


if __name__ == "__main__":
    main()
