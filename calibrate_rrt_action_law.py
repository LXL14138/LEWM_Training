"""SIMULATION ONLY: calibrate the NOMINAL model of rrt_action_law on free-space executions in one scene.

No dataset episode is opened: every start pose is built from env.reset() (empty gripper) or by the data-collection
carry controller (exploration.ExplorationController, mode 'carry') grasping the green block from the reset scene.
The only file read is the WM action normalization of data/wm_full_001/index.json (needed for the WM-range clip).

From each start pose (restored bit-exactly with bench_restore_check.restore_state) the law executes
  * 18 fixed directions: edge (8 steps) + hold (12)      * N random planner sequences: 3 random edges + hold (12)
and every step's executed action, PI output, pinch, integral, lag and anti-windup flag is recorded. Runs with any
guard contact (robot or green block, > 1e-3 N) or a grasp change are NOT free space and are excluded from the fit.
Two models are fitted (rrt_action_law.fit_nominal_model): 'empty' on the empty-gripper poses, 'grasped' on the
grasped poses; the nominal-vs-executed action mismatch is reported in-sample and leave-one-start-out, and also for
the constants currently in rrt_action_law.NOMINAL_MODELS.

Outputs (NEW directory): run.json at start; runs.npz, summary.json (only on success), REPORT_ZH.md; FAILED.json on error.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

import rrt_action_law as law
from bench_restore_check import STATE_SPEC, pinch_xyz, restore_state
from exploration import ExplorationController, ExplorationScene
from wm_training.danger_strata import native, sha256, write_new_json
from wm_training.provenance import implementation_hashes

FORMAT = "rrt_action_law_calib_v1"
ROOT_FILES = ("calibrate_rrt_action_law.py", "rrt_action_law.py", "bench_restore_check.py", "exploration.py", "collect_grasp_trajs.py",
              "three_block_scene.py", "scene_config.py", "task_progress.py")
CONTACT_N = 1e-3
SETTLE_STEPS = 30
CARRY_SEEDS = {"grasped_goal": 3, "grasped_goal2": 11}
TOLERANCE_MM = {"median": 0.3, "max": 1.5}          # stated free-space tolerance (tests/test_rrt_action_law.py uses the same)


def fixed_directions():
    dirs = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            d = np.zeros(3)
            d[axis] = sign
            dirs.append(d)
    for sx in (1, -1):
        for sy in (1, -1):
            dirs.append(law.unit([sx, sy, 0.0]))
    for sz in (1, -1):
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dirs.append(law.unit([sx, sy, sz]))
    return dirs


class ScenePlant:
    """rrt_action_law plant on the real scene; remembers the largest guard contact seen."""

    def __init__(self, env):
        self.env, self.contact = env, 0.0

    def pinch(self):
        return pinch_xyz(self.env)

    def step(self, action):
        self.env.last_transition = None
        self.env.step(np.asarray(action, float))
        if self.env.last_transition is None or not np.array_equal(self.env.last_transition["action"], action):
            raise RuntimeError("the scene did not execute the law's action")
        self.contact = max(self.contact, float(np.max(self.env._contact_peak)), float(np.max(self.env._green_contact_peak)))


def get_state(env):
    state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
    return state


def settle(env, grip, config, segments=()):
    plant = ScenePlant(env)
    if segments:
        law.run_sequence(plant, list(segments), grip, config)
    r = law.run_sequence(plant, [law.hold(SETTLE_STEPS)], grip, config)
    return r["pinch_xyz"][-1] - r["pinch_xyz"][-2], plant.contact


def start_record(env, name, grip, velocity, contact):
    return {"name": name, "state": get_state(env), "goal": env.hover_goal.copy(), "grip": float(grip), "v0": velocity,
            "grasped": bool(env.task_report()["grasped"]), "pinch": pinch_xyz(env), "green_com": env.data.body("object_0").xipos.copy(),
            "settle_contact_n": contact}


def build_starts(env, config):
    """Empty-gripper poses from the reset scene and grasped poses from the carry controller; each settled by a hold."""
    starts = []
    env.reset(seed=0)
    v, c = settle(env, -1.0, config)
    starts.append(start_record(env, "empty_reset", -1.0, v, c))
    base = starts[0]
    for name, segs in (("empty_low", [law.edge([-1, 0, -1])] * 4), ("empty_side", [law.edge([0.3, 1, 0.8])] * 5)):
        restore_state(env, base["state"], base["goal"])
        v, c = settle(env, -1.0, config, segs)
        starts.append(start_record(env, name, -1.0, v, c))
    for name, seed in CARRY_SEEDS.items():
        env.reset(seed=0)
        controller = ExplorationController(env, np.random.default_rng(seed), "carry")
        steps = 0
        while not controller.finished and steps < 600:
            controller.step()
            steps += 1
            if controller.phase == "HOVER" and env.task_report()["grasped"]:
                break
        if controller.phase != "HOVER" or not env.task_report()["grasped"]:
            raise RuntimeError(f"carry controller (seed {seed}) did not reach a grasped hover: {controller.phase}")
        v, c = settle(env, 1.0, config)
        starts.append(start_record(env, name, 1.0, v, c))
        if name == "grasped_goal":
            goal_start = starts[-1]
            for derived, segs in (("grasped_back", [law.edge([-1, 0, 0])] * 4 + [law.edge([0, 0, -1])] * 3),
                                  ("grasped_high", [law.edge([0, 0, 1])] * 4)):
                restore_state(env, goal_start["state"], goal_start["goal"])
                v, c = settle(env, 1.0, config, segs)
                starts.append(start_record(env, derived, 1.0, v, c))
    for s in starts:
        if s["grasped"] != (s["grip"] > 0):
            raise RuntimeError(f"start {s['name']}: grasp state {s['grasped']} does not match its grip command {s['grip']}")
    return starts


def run_plans(env, starts, config, n_random, seed):
    rng = np.random.default_rng(seed)
    runs = []
    for s in starts:
        plans = [[law.edge(d), law.hold(law.CHECK_HOLD_STEPS)] for d in fixed_directions()]
        plans += [[law.edge(law.random_direction(rng)) for _ in range(3)] + [law.hold(law.CHECK_HOLD_STEPS)] for _ in range(n_random)]
        for segments in plans:
            restore_state(env, s["state"], s["goal"])
            if not np.array_equal(pinch_xyz(env), s["pinch"]):
                raise RuntimeError("start pose was not restored exactly")
            plant = ScenePlant(env)
            grasped_before = bool(env.task_report()["grasped"])
            r = law.run_sequence(plant, segments, s["grip"], config)
            runs.append({"start": s["name"], "group": "grasped" if s["grasped"] else "empty", "grip": s["grip"], "v0": s["v0"],
                         "segments": segments, "actions": r["actions"], "pinch_xyz": r["pinch_xyz"], "u_m": r["u_m"],
                         "integral_m": r["integral_m"], "lag_m": r["lag_m"], "reference_xyz": r["reference_xyz"],
                         "reference_frozen": r["reference_frozen"], "integral_clamped": r["integral_clamped"],
                         "reference_dragged": r["reference_dragged"], "box_clipped": r["box_clipped"], "wm_clipped": r["wm_clipped"],
                         "contact_n": plant.contact, "grasp_changed": bool(env.task_report()["grasped"]) != grasped_before})
    return runs


def mismatch_stats(runs, model, config):
    if not runs:
        return None
    d = []
    for r in runs:
        nominal = law.nominal_actions(r["segments"], r["pinch_xyz"][0], r["grip"], config, model, prev_velocity=r["v0"])
        d.append(law.action_mismatch(nominal, r["actions"], config.normalization))
    out = {}
    for unit_name in ("mm", "wm", "env"):
        v = np.concatenate([x[unit_name] for x in d])
        out[unit_name] = {"median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)), "p99": float(np.quantile(v, 0.99)),
                          "max": float(v.max())}
    out["steps"] = int(sum(len(x["mm"]) for x in d))
    out["within_tolerance"] = bool(out["mm"]["median"] <= TOLERANCE_MM["median"] and out["mm"]["max"] <= TOLERANCE_MM["max"])
    return out


def free_space_needs(runs):
    lag = np.concatenate([r["lag_m"] for r in runs])
    integral = np.concatenate([r["integral_m"] for r in runs])
    u = np.concatenate([r["u_m"] for r in runs])
    return {"lag_mm": {"median": float(1000 * np.median(lag)), "p99": float(1000 * np.quantile(lag, 0.99)), "max": float(1000 * lag.max())},
            "integral_abs_max_mm": (1000 * np.abs(integral).max(0)).tolist(), "pi_output_min_mm": (1000 * u.min(0)).tolist(),
            "pi_output_max_mm": (1000 * u.max(0)).tolist(),
            **{f"{k}_steps": int(sum(np.asarray(r[k]).any(-1).sum() if np.asarray(r[k]).ndim > 1 else np.asarray(r[k]).sum() for r in runs))
               for k in ("reference_frozen", "integral_clamped", "reference_dragged", "box_clipped", "wm_clipped")}}


def python_literal(models):
    def mat(m):
        return "(" + ", ".join("(" + ", ".join(f"{x:.6g}" for x in row) + ")" for row in np.asarray(m)) + ")"
    lines = []
    for name, model in models.items():
        lines.append(f'    "{name}": NominalModel(gain={mat(model.gain)},\n        memory={mat(model.memory)},\n'
                     f'        bias=({", ".join(f"{x:.6g}" for x in model.bias)}), source="{model.source}"),')
    return "NOMINAL_MODELS = {\n" + "\n".join(lines) + "\n}"


def save_runs(path, runs):
    arrays = {}
    for i, r in enumerate(runs):
        for key in ("actions", "pinch_xyz", "u_m", "integral_m", "lag_m", "reference_xyz", "reference_frozen", "integral_clamped",
                    "reference_dragged", "box_clipped", "wm_clipped", "v0"):
            arrays[f"r{i:04d}_{key}"] = np.asarray(r[key])
        arrays[f"r{i:04d}_segments"] = np.array(json.dumps([[k, np.asarray(a).tolist() if k == "edge" else a] for k, a in r["segments"]]))
    arrays["start"] = np.array([r["start"] for r in runs])
    arrays["grip"] = np.array([r["grip"] for r in runs])
    arrays["contact_n"] = np.array([r["contact_n"] for r in runs])
    arrays["grasp_changed"] = np.array([r["grasp_changed"] for r in runs])
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def calibrate(env, config, n_random, seed):
    starts = build_starts(env, config)
    runs = run_plans(env, starts, config, n_random, seed)
    is_free = [r["contact_n"] <= CONTACT_N and not r["grasp_changed"] for r in runs]
    free = [r for r, ok in zip(runs, is_free) if ok]
    dropped = [{"run": i, "start": r["start"], "contact_n": r["contact_n"], "grasp_changed": r["grasp_changed"]}
               for i, (r, ok) in enumerate(zip(runs, is_free)) if not ok]
    models, fits = {}, {}
    for group in ("empty", "grasped"):
        members = [r for r in free if r["group"] == group]
        model, rms = law.fit_nominal_model(members, f"{FORMAT}:{group}")
        models[group] = model
        names = sorted({r["start"] for r in members})
        fits[group] = {"starts": names, "runs": len(members), "residual_rms_mm": (1000 * rms).tolist(), "model": model.to_json(),
                       "in_sample": {n: mismatch_stats([r for r in members if r["start"] == n], model, config) for n in names},
                       "leave_one_start_out": {}, "module_constants": {}}
        for held in names:
            m, _ = law.fit_nominal_model([r for r in members if r["start"] != held], "cv")
            fits[group]["leave_one_start_out"][held] = mismatch_stats([r for r in members if r["start"] == held], m, config)
        for n in names:
            fits[group]["module_constants"][n] = mismatch_stats([r for r in members if r["start"] == n], law.NOMINAL_MODELS[group], config)
    wrong_model = {g: mismatch_stats([r for r in free if r["group"] == g], models["grasped" if g == "empty" else "empty"], config)
                   for g in ("empty", "grasped")}
    return starts, runs, free, dropped, models, fits, wrong_model


def write_report(output, summary):
    lines = ["# RRT 动作律：名义动作模型的自由空间标定\n",
             f"- 场景：只用 `env.reset()` 与采数据的搬运控制器（`ExplorationController` carry 模式，种子 {list(CARRY_SEEDS.values())}）从复位场景夹起绿块，"
             "不读任何数据集轨迹；唯一读取的文件是 `data/wm_full_001/index.json` 的动作归一化。",
             f"- 起始姿态 {len(summary['starts'])} 个；每个姿态 18 个固定方向（边 8 步 + 停 12 步）+ {summary['args']['random_sequences']} 条随机规划序列（3 条随机边 + 停 12 步）；"
             f"共 {summary['runs']} 条，自由空间（无红柱接触、夹持不变）{summary['free_runs']} 条，剔除 {len(summary['dropped'])} 条。",
             f"- 动作律常数：{json.dumps({k: summary['law'][k] for k in ('speed_mps', 'lag_margin_m', 'integral_max_m', 'lead_max_m')}, ensure_ascii=False)}。",
             f"- 判定容差（自由空间，名义与实际执行动作逐步逐轴差，以物理目标偏移计）：中位数 ≤ {TOLERANCE_MM['median']} mm、最大 ≤ {TOLERANCE_MM['max']} mm。\n",
             "## 1. 自由空间需要多少（决定防饱和常数）\n",
             f"- 跟踪滞后 |参考点 − 夹爪|：中位 {summary['free_space']['lag_mm']['median']:.2f} mm、p99 {summary['free_space']['lag_mm']['p99']:.2f}、最大 {summary['free_space']['lag_mm']['max']:.2f} mm（冻结阈值 {1000 * summary['law']['lag_margin_m']:g} mm）。",
             f"- 积分项绝对值最大（x, y, z）：{[round(v, 2) for v in summary['free_space']['integral_abs_max_mm']]} mm（积分上限 {1000 * summary['law']['integral_max_m']:g} mm）。",
             f"- PI 输出范围：最小 {[round(v, 2) for v in summary['free_space']['pi_output_min_mm']]}、最大 {[round(v, 2) for v in summary['free_space']['pi_output_max_mm']]} mm。",
             f"- 自由空间里防饱和的触发步数：参考冻结 {summary['free_space']['reference_frozen_steps']}、积分被截 {summary['free_space']['integral_clamped_steps']}、"
             f"参考被拖 {summary['free_space']['reference_dragged_steps']}、WM 范围裁剪 {summary['free_space']['wm_clipped_steps']}、工作框裁剪 {summary['free_space']['box_clipped_steps']}。\n",
             "## 2. 名义模型与失配\n",
             "模型：v_k = G u_k + M v_{k−1} + b（v = 夹爪每步位移，u = 执行动作的物理 xyz），最小二乘；empty / grasped 分开拟合。失配 = |名义 − 实际| 的 dx/dy/dz（物理 mm；WM 单位 = 除以归一化 scale）。\n",
             "| 组 | 起始姿态 | 步数 | 样本内 中位/p95/最大 (mm) | 留一姿态 中位/p95/最大 (mm) | 留一 最大 (WM) | 模块常数 中位/最大 (mm) |", "|---|---|---|---|---|---|---|"]
    for group, fit in summary["fits"].items():
        for n in fit["starts"]:
            a, b, c = fit["in_sample"][n], fit["leave_one_start_out"][n], fit["module_constants"][n]
            lines.append(f"| {group} | {n} | {a['steps']} | {a['mm']['median']:.3f} / {a['mm']['p95']:.3f} / {a['mm']['max']:.3f} | "
                         f"{b['mm']['median']:.3f} / {b['mm']['p95']:.3f} / {b['mm']['max']:.3f} | {b['wm']['max']:.3f} | {c['mm']['median']:.3f} / {c['mm']['max']:.3f} |")
    lines += ["", f"- 用错模型（empty 的序列用 grasped 模型，反之）：{json.dumps({g: None if v is None else round(v['mm']['max'], 3) for g, v in summary['wrong_model'].items()}, ensure_ascii=False)} mm 最大失配。",
              "- 把 summary.json 的 `python_literal` 贴进 `rrt_action_law.NOMINAL_MODELS` 后，“模块常数”一列应与样本内一列相同（再跑一次本工具或单元测试核对）。\n",
              "## 3. 说明（测了什么、没测什么）\n",
              "- 测了：自由空间（无接触）下名义动作与实际执行动作的差；起始姿态覆盖空手（复位、低、侧）与夹持（悬停处、后方低处、高处、另一个目标）。",
              "- 没测：有接触时（推柱、被倒下的柱子带走）名义动作不代表执行动作——那正是规划器要靠 WM 判断的情形；工作空间边缘、目标盒子以外的姿态只部分覆盖；夹持姿态只来自两个搬运种子。",
              f"- 用时 {summary['elapsed_seconds']:.0f} s。"]
    (Path(output) / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="NEW run directory, e.g. runs/rrt_action_law_calib_001")
    p.add_argument("--index", type=Path, default=Path("data/wm_full_001/index.json"), help="cache index holding the WM action normalization")
    p.add_argument("--random-sequences", type=int, default=10, help="random planner sequences per start pose")
    p.add_argument("--seed", type=int, default=20260928)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    if a.random_sequences < 1 or a.seed < 0:
        raise ValueError("--random-sequences >= 1 and --seed >= 0")
    normalization = law.load_wm_normalization(a.index)
    config = law.LawConfig(normalization=normalization)
    root = Path(__file__).resolve().parent
    output.mkdir(parents=True)
    run = {"format": FORMAT, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "input_sha256": {str(a.index.resolve()): normalization["sha256"]},
           "implementation_sha256": {**implementation_hashes(ROOT_FILES, package_dir=root), **implementation_hashes(("danger_strata.py", "provenance.py"))},
           "law": config.to_json(), "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                                                  "ogbench": version("ogbench")},
           "datasets_opened": [], "world_model_loaded": False, "tolerance_mm": TOLERANCE_MM}
    write_new_json(output / "run.json", run)
    env = None
    started = time.perf_counter()
    try:
        env = ExplorationScene()
        starts, runs, free, dropped, models, fits, wrong = calibrate(env, config, a.random_sequences, a.seed)
        save_runs(output / "runs.npz", runs)
        summary = {"format": FORMAT, "args": run["args"], "law": run["law"], "tolerance_mm": TOLERANCE_MM,
                   "starts": [{k: s[k] for k in ("name", "grip", "grasped", "pinch", "green_com", "v0", "settle_contact_n")} for s in starts],
                   "runs": len(runs), "free_runs": len(free), "dropped": dropped, "free_space": free_space_needs(free),
                   "models": {g: m.to_json() for g, m in models.items()}, "fits": fits, "wrong_model": wrong,
                   "python_literal": python_literal(models), "runs_npz_sha256": sha256(output / "runs.npz"),
                   "elapsed_seconds": time.perf_counter() - started}
        write_report(output, summary)
        write_new_json(output / "summary.json", summary)
        print(json.dumps(native({"free_space": summary["free_space"],
                                 "leave_one_start_out_max_mm": {g: {n: v["mm"]["max"] for n, v in f["leave_one_start_out"].items()} for g, f in fits.items()}}),
                         ensure_ascii=False), flush=True)
        print(summary["python_literal"], flush=True)
        return summary
    except BaseException as error:
        write_new_json(output / "FAILED.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
