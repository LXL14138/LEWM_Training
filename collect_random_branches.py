"""SIMULATION ONLY: stage 2 of the random-action branch study: roll out planner-like action sequences from real states.

Reads a stage-1 plan (random_branches_plan.py: anchors.json + cache_pixels.npz) and, for every anchor a and each of
its 8 rollouts (recorded, hold, 4 x random, 2 x guard):

  restore the recorded physics_state of frame a-10 with collect_counterfactual_branches.restore_rebased
  (reset -> hover goal at time 0 -> mj_setState(16383) -> clock 0 -> mj_forward -> post_step), replay the 10
  recorded actions bit-exactly (qpos/qvel at a must equal the record; anything else raises), then run 24 control
  steps of the rollout's action law and record every state and the front_pixels image, letterboxed EXACTLY like
  the cache (wm_training.prepare_data.letterbox).

Action laws: recorded = the recorded actions a..a+23; hold = bench_restore_check.AnchoredHold at the anchor pinch;
random / guard = the same PI tracker (P = I = 1, clipped to the +-5 cm action range, yaw 0, gripper command of the
recorded action at a) following a reference that starts at the anchor pinch and moves speed*0.05*d per step through
3 pre-drawn 8-step segments (random_branches_plan.reference_path, clipped to the reference box).

Output: one npz per anchor in a NEW dataset directory (manifest.json written once at the end), and run.json /
summary.json / REPORT_ZH.md (FAILED.json on error) in a NEW run directory. Datasets and caches are read-only.

Format v2 (random_branches_v2) adds, without changing any simulated rollout: wm_action computed with the float32
recipe of evaluate_bench_counterfactual.normalize_actions; per-step tracking_error_m / action_saturated arrays; the
raw 640x480 renders of all 11 prefix frames compared with the dataset PNGs; per-process peak RSS; and in summary /
report the S1 / S2 truth pools split by anchor tilt (< 10 deg vs >= 10 deg) and saturation.
"""
from __future__ import annotations

import argparse
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import threading
import time
import types

import mujoco
import numpy as np
from PIL import Image

from bench_restore_check import RENDER_KEYS, STATE_SPEC, AnchoredHold, load_episode, pinch_xyz
from collect_counterfactual_branches import restore_rebased
from collection_workers import CHILD_THREAD_LIMITS, run_workers
from exploration import ExplorationScene
import random_branches_plan as plan
from wm_training.danger_strata import native, read_json, sha256, write_new_json
from wm_training.provenance import implementation_hashes

FORMAT = "random_branches_v2"            # v2: float32 wm_action recipe, tracking_error_m / action_saturated, criteria pools
IMAGE_SIZE = 224
CAMERA = "front_pixels"
STEPS = plan.HORIZON + 1
STEP_FIELDS = {"guard_tilt_deg": ((2,), np.float64), "unsafe": ((), bool), "class_label": ((), np.int8),
               "hover_error_m": ((), np.float64), "target_speed_mps": ((), np.float64), "hover_conditions_met": ((), bool),
               "grasped": ((), bool), "object_com_xyz": ((3, 3), np.float64), "object_quat_wxyz": ((3, 4), np.float64),
               "pinch_xyz": ((3,), np.float64), "guard_contact_force_n": ((2,), np.float64),
               "guard_contact_peak_last_step_n": ((2,), np.float64), "green_guard_contact_peak_last_step_n": ((2,), np.float64),
               "physics_state": ((343,), np.float64)}
BYTES_PER_ANCHOR_ESTIMATE = 4_000_000        # replaced by the pilot's measurement in the docs
SECONDS_PER_ANCHOR_ESTIMATE = 10.0           # per worker
DISK_RESERVE_BYTES = 5 * 1024**3
ROOT_FILES = ("collect_random_branches.py", "random_branches_plan.py", "bench_restore_check.py", "collect_counterfactual_branches.py",
              "exploration.py", "collect_grasp_trajs.py", "collection_workers.py", "three_block_scene.py", "scene_config.py", "task_progress.py")
WM_FILES = ("prepare_data.py", "danger_strata.py", "provenance.py")


# ----- preprocessing ------------------------------------------------------------------------------------
def import_letterbox():
    """wm_training.prepare_data.letterbox, the function that built the caches.

    prepare_data imports h5py at module level, which the simulation venv lacks; letterbox itself only uses PIL and
    numpy. If h5py is missing, a placeholder module stands in for the duration of the import only (both the
    placeholder and the imported prepare_data module are removed from sys.modules afterwards, so nothing else can
    use them). Returns (function, how)."""
    try:
        from wm_training.prepare_data import letterbox
        return letterbox, "direct_import"
    except ModuleNotFoundError as error:
        if error.name != "h5py":
            raise
    placeholder = types.ModuleType("h5py")
    placeholder.__doc__ = "import placeholder: letterbox does not use h5py"
    sys.modules["h5py"] = placeholder
    try:
        module = importlib.import_module("wm_training.prepare_data")
        letterbox = module.letterbox
    finally:
        sys.modules.pop("h5py", None)
        sys.modules.pop("wm_training.prepare_data", None)
    if letterbox.__module__ != "wm_training.prepare_data":
        raise RuntimeError("letterbox was not taken from wm_training.prepare_data")
    return letterbox, "h5py_placeholder_during_import"


def preprocess(letterbox, frame):
    if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise RuntimeError(f"Unexpected render {frame.shape} {frame.dtype}")
    out = letterbox(Image.fromarray(frame), IMAGE_SIZE)
    if out.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or out.dtype != np.uint8:
        raise RuntimeError("letterbox returned an unexpected array")
    return out


def pixel_diff(a, b):
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    return {"identical": bool(d.max() == 0), "max_abs_diff": int(d.max()), "fraction_gt2": float((d.max(-1) > 2).mean())}


def peak_rss_kb():
    """This process's peak resident set size (kB, Linux ru_maxrss); None where the resource module is missing."""
    try:
        import resource
    except ImportError:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _proc_descendants(root):
    """root and every live descendant process (Linux /proc)."""
    parents = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat", encoding="ascii", errors="replace") as handle:
                    parents[int(name)] = int(handle.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
    found, frontier = {root}, [root]
    while frontier:
        parent = frontier.pop()
        for pid, ppid in parents.items():
            if ppid == parent and pid not in found:
                found.add(pid)
                frontier.append(pid)
    return found


def _proc_rollup_kb(pid):
    """{'Rss': kB, 'Pss': kB} of one process from /proc/<pid>/smaps_rollup; None if it is gone."""
    values = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup", encoding="ascii", errors="replace") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("Rss", "Pss"):
                    values[key] = int(rest.split()[0])
    except (OSError, ValueError):
        return None
    return values


class MemorySampler:
    """Background sampler of the job's real memory footprint (Linux /proc; inactive elsewhere).

    Every ``interval`` seconds it sums, over this process and all its descendants (the workers), the PSS
    (proportional set size: a shared page is split between the processes that map it, so the sum is the job's real
    footprint) and the RSS (shared pages counted once per process: an over-estimate), and reads MemAvailable."""

    def __init__(self, interval=2.0):
        self.interval, self.root = float(interval), os.getpid()
        self.active = os.path.exists("/proc/self/smaps_rollup")
        self.samples, self.max_processes = 0, 0
        self.peak_pss_kb, self.peak_rss_sum_kb, self.min_available_kb = 0, 0, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-sampler", daemon=True)

    def sample_once(self):
        stats = [v for v in (_proc_rollup_kb(pid) for pid in _proc_descendants(self.root)) if v]
        self.samples += 1
        self.max_processes = max(self.max_processes, len(stats))
        self.peak_pss_kb = max(self.peak_pss_kb, sum(v.get("Pss", 0) for v in stats))
        self.peak_rss_sum_kb = max(self.peak_rss_sum_kb, sum(v.get("Rss", 0) for v in stats))
        try:
            with open("/proc/meminfo", encoding="ascii") as handle:
                available = next(int(line.split()[1]) for line in handle if line.startswith("MemAvailable:"))
            self.min_available_kb = available if self.min_available_kb is None else min(self.min_available_kb, available)
        except (OSError, StopIteration, ValueError):
            pass

    def _run(self):
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval)

    def start(self):
        if self.active:
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        return self.result()

    def result(self):
        return {"active": self.active, "interval_s": self.interval, "samples": self.samples, "max_processes": self.max_processes,
                "peak_pss_sum_kb": self.peak_pss_kb if self.samples else None,
                "peak_rss_sum_kb": self.peak_rss_sum_kb if self.samples else None, "min_mem_available_kb": self.min_available_kb}


def recorded_png(loaded, frame):
    return np.asarray(Image.open(Path(loaded["dir"]) / "images" / f"{frame:06d}.png").convert("RGB"))


# ----- controllers --------------------------------------------------------------------------------------
class ReferenceTracker(AnchoredHold):
    """AnchoredHold's PI law (same gains, same +-5 cm clipping of error and integral) on a pre-computed reference path."""

    def __init__(self, path, grip, dt):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or not np.isfinite(path).all():
            raise ValueError("path must be (steps, 3) finite")
        super().__init__(path[0], grip, dt=dt)
        self.path = path

    def reference(self, step_index):
        return self.path[min(step_index, len(self.path) - 1)].copy()


def build_controller(env, spec, anchor_pinch, grip, dt):
    """-> (controller or None for 'recorded', reference path (HORIZON, 3), box or None)."""
    family = spec["family"]
    if family == "recorded":
        return None, np.full((plan.HORIZON, 3), np.nan), None
    if family == "hold":
        return AnchoredHold(anchor_pinch, grip, dt=dt), np.tile(anchor_pinch, (plan.HORIZON, 1)), None
    low, high = plan.reference_box(anchor_pinch, *[np.asarray(b, float) for b in env._workspace_bounds])
    path = plan.reference_path(anchor_pinch, spec["directions"], spec["speeds_mps"], low, high, dt=dt)
    return ReferenceTracker(path, grip, dt), path, (low, high)


# ----- one anchor ---------------------------------------------------------------------------------------
def load_source(anchor):
    src = anchor["source"]
    folder = Path(src["episode_dir"])
    if sha256(folder / "trajectory.npz") != src["npz_sha256"] or sha256(folder / "metadata.json") != src["metadata_sha256"]:
        raise ValueError(f"{folder}: source files differ from the plan")
    loaded = load_episode(folder)
    with np.load(folder / "trajectory.npz", allow_pickle=False) as npz:
        loaded["class_label"] = np.asarray(npz["class_label"])
    frame = int(anchor["frame"])
    if loaded["frames"] != int(src["frames"]) or not (plan.MIN_ANCHOR <= frame and frame + plan.HORIZON <= loaded["frames"] - 1):
        raise ValueError(f"{folder}: anchor {frame} not admissible for {loaded['frames']} frames")
    if loaded["camera"] != CAMERA or loaded["class_label"].shape != (loaded["frames"],):
        raise ValueError(f"{folder}: unexpected camera or labels")
    return loaded


def capture(env):
    state = env.exploration_state()
    return {k: np.asarray(state[k]) for k in STEP_FIELDS}


def run_anchor(env, job, letterbox, cache_block):
    """All rollouts of one anchor -> (npz arrays, manifest row)."""
    started = time.perf_counter()
    anchor, normalization = job["anchor"], job["normalization"]
    loaded = load_source(anchor)
    a = int(anchor["frame"])
    start = a - plan.PREFIX
    dt = loaded["control_dt"]
    grip = float(loaded["actions"][a][4])
    specs = anchor["sequences"]
    if [s["family"] for s in specs] != list(plan.FAMILIES):
        raise ValueError("rollout families differ from the plan definition")
    if cache_block.shape != (plan.PREFIX + 1, IMAGE_SIZE, IMAGE_SIZE, 3) or cache_block.dtype != np.uint8:
        raise ValueError("cache pixel block has the wrong shape")
    ws_low, ws_high = (np.asarray(b, float) for b in env._workspace_bounds)
    if not (np.array_equal(ws_low, plan.WORKSPACE_LOW) and np.array_equal(ws_high, plan.WORKSPACE_HIGH)):
        raise RuntimeError("scene workspace bounds differ from the plan's constants")
    R = len(specs)
    arrays = {k: np.zeros((R, STEPS, *shape), dtype=dtype) for k, (shape, dtype) in STEP_FIELDS.items()}
    arrays.update(pixels=np.zeros((R, STEPS, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8), actions=np.zeros((R, plan.HORIZON, 5)),
                  reference_xyz=np.zeros((R, plan.HORIZON, 3)), segment_direction=np.full((R, plan.SEGMENTS, 3), np.nan),
                  segment_speed_mps=np.full((R, plan.SEGMENTS), np.nan))
    prefix_pixels = np.zeros((plan.PREFIX + 1, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    replay_diff, anchor_images_equal, rec = 0.0, True, {"prefix_raw_png": []}
    for r, spec in enumerate(specs):
        restore_rebased(env, loaded, start)
        if not np.array_equal(env.hover_goal, loaded["hover_goal"]):
            raise RuntimeError("hover goal was not restored")
        if r == 0:
            raw = env.render(camera=CAMERA)
            prefix_pixels[0] = preprocess(letterbox, raw)
            rec["prefix_raw_png"].append(pixel_diff(raw, recorded_png(loaded, start)))
        for j in range(plan.PREFIX):
            frame = start + j
            env.last_transition = None
            env.step(np.asarray(loaded["actions"][frame], dtype=float))
            if env.last_transition is None or not np.array_equal(env.last_transition["action"], loaded["actions"][frame]):
                raise RuntimeError("prefix action was not executed as recorded")
            replay_diff = max(replay_diff, float(np.abs(env.data.qpos - loaded["qpos"][frame + 1]).max()),
                              float(np.abs(env.data.qvel - loaded["qvel"][frame + 1]).max()))
            if r == 0:
                raw = env.render(camera=CAMERA)
                prefix_pixels[j + 1] = preprocess(letterbox, raw)
                rec["prefix_raw_png"].append(pixel_diff(raw, recorded_png(loaded, frame + 1)))
        if not np.array_equal(env.data.qpos, loaded["qpos"][a]) or not np.array_equal(env.data.qvel, loaded["qvel"][a]):
            raise RuntimeError(f"{loaded['dir']} anchor {a}: state after the replayed prefix differs from the record")
        anchor_pinch = pinch_xyz(env)
        controller, path, _ = build_controller(env, spec, anchor_pinch, grip, dt)
        arrays["reference_xyz"][r] = path
        if spec["directions"] is not None:
            arrays["segment_direction"][r] = spec["directions"]
            arrays["segment_speed_mps"][r] = spec["speeds_mps"]
        for key, value in capture(env).items():
            arrays[key][r, 0] = value
        if r == 0:
            arrays["pixels"][r, 0] = prefix_pixels[plan.PREFIX]
            rec["raw_png"] = rec["prefix_raw_png"][plan.PREFIX]                 # the anchor frame a
        else:
            arrays["pixels"][r, 0] = preprocess(letterbox, env.render(camera=CAMERA))
            anchor_images_equal &= bool(np.array_equal(arrays["pixels"][r, 0], arrays["pixels"][0, 0]))
        for k in range(plan.HORIZON):
            action = np.asarray(loaded["actions"][a + k], dtype=float) if controller is None else controller.action(env)
            env.last_transition = None
            env.step(action)
            if env.last_transition is None:
                raise RuntimeError("Missing executed transition")
            arrays["actions"][r, k] = env.last_transition["action"]
            for key, value in capture(env).items():
                arrays[key][r, k + 1] = value
            arrays["pixels"][r, k + 1] = preprocess(letterbox, env.render(camera=CAMERA))
            if controller is None:
                rec.setdefault("qpos_diff", []).append(float(np.abs(env.data.qpos - loaded["qpos"][a + k + 1]).max()))
    frames = slice(a, a + STEPS)
    source_tilt, source_class, source_unsafe = loaded["guard_tilt_deg"][frames], loaded["class_label"][frames].astype(np.int8), loaded["unsafe"][frames]
    cache_frames = [pixel_diff(prefix_pixels[i], cache_block[i]) for i in range(plan.PREFIX + 1)]
    sanity = {"prefix_replay_max_diff": replay_diff, "prefix_replay_exact": bool(replay_diff == 0.0),
              "recorded_max_tilt_diff_deg": float(np.abs(arrays["guard_tilt_deg"][0] - source_tilt).max()),
              "recorded_max_qpos_diff": float(max(rec["qpos_diff"])),
              "recorded_max_com_diff_m": float(np.abs(arrays["object_com_xyz"][0] - loaded["object_com_xyz"][frames]).max()),
              "recorded_class_mismatches": int((arrays["class_label"][0] != source_class).sum()),
              "recorded_unsafe_mismatches": int((arrays["unsafe"][0] != source_unsafe).sum()),
              "recorded_actions_equal": bool(np.array_equal(arrays["actions"][0], loaded["actions"][a:a + plan.HORIZON])),
              "anchor_cache": cache_frames[plan.PREFIX], "prefix_cache_identical_frames": int(sum(f["identical"] for f in cache_frames)),
              "prefix_cache_max_abs_diff": int(max(f["max_abs_diff"] for f in cache_frames)), "raw_png": rec["raw_png"],
              "prefix_raw_png_identical_frames": int(sum(f["identical"] for f in rec["prefix_raw_png"])),
              "prefix_raw_png_max_abs_diff": int(max(f["max_abs_diff"] for f in rec["prefix_raw_png"])),
              "anchor_images_equal_across_rollouts": anchor_images_equal,
              "anchor_state_matches_plan": bool(np.array_equal(arrays["pinch_xyz"][0, 0], np.asarray(anchor["state"]["pinch_xyz"])))}
    sanity["recorded_exact"] = bool(sanity["recorded_max_tilt_diff_deg"] == 0 and sanity["recorded_max_qpos_diff"] == 0
                                    and sanity["recorded_class_mismatches"] == 0 and sanity["recorded_unsafe_mismatches"] == 0)
    arrays.update(
        wm_action=plan.normalize_wm_actions(arrays["actions"], normalization), prefix_pixels=prefix_pixels,
        tracking_error_m=np.stack([plan.tracking_error(arrays["pinch_xyz"][r], arrays["reference_xyz"][r]) for r in range(R)]),
        action_saturated=plan.saturated_steps(arrays["actions"]),
        prefix_actions=np.asarray(loaded["actions"][start:a], dtype=np.float64),
        prefix_wm_action=plan.normalize_wm_actions(loaded["actions"][start:a], normalization),
        source_guard_tilt_deg=source_tilt, source_class_label=source_class, source_unsafe=source_unsafe.astype(bool),
        hover_goal_xyz=np.asarray(loaded["hover_goal"], np.float64), families=np.array(plan.FAMILIES),
        anchor_id=np.int64(anchor["id"]), source_frame=np.int64(a), cache_episode_id=np.int64(anchor["source"]["cache_episode_id"]),
        set=np.array(anchor["set"]), group=np.array(anchor["group"]), stratum=np.array(anchor["stratum"]),
        grip_command=np.float64(grip))
    outcomes = []
    for r, spec in enumerate(specs):
        o = plan.rollout_outcome(arrays["guard_tilt_deg"][r], arrays["unsafe"][r], arrays["class_label"][r], arrays["pinch_xyz"][r],
                                 arrays["guard_contact_peak_last_step_n"][r], arrays["green_guard_contact_peak_last_step_n"][r],
                                 reference=arrays["reference_xyz"][r], actions=arrays["actions"][r], wm_action=arrays["wm_action"][r])
        outcomes.append({"rollout": r, "family": spec["family"], **o})
    row = {"id": int(anchor["id"]), "set": anchor["set"], "group": anchor["group"], "stratum": anchor["stratum"],
           "cache_episode_id": int(anchor["source"]["cache_episode_id"]), "frame": a, "lead_k": anchor.get("lead_k"),
           "grip_command": grip, "sanity": sanity, "rollouts": outcomes}
    return arrays, row, time.perf_counter() - started


def write_anchor(output_dir, anchor_id, arrays):
    path = Path(output_dir) / f"anchor_{anchor_id:04d}.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return path


def process_job(env, job, letterbox, cache_block):
    arrays, row, seconds = run_anchor(env, job, letterbox, cache_block)
    path = write_anchor(job["output_dir"], row["id"], arrays)
    row.update(file=path.name, sha256=sha256(path), bytes=path.stat().st_size, wall_seconds=seconds, worker_pid=os.getpid(),
               worker_peak_rss_kb=peak_rss_kb())
    return row


def _worker(pipe, stop):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env, cache, letterbox = None, None, None
    try:
        while True:
            job = pipe.recv()
            if job is None or stop.is_set():
                return
            try:
                if env is None:
                    env = ExplorationScene()
                    letterbox, _ = import_letterbox()
                    cache = np.load(job["cache_pixels"], allow_pickle=False)
                row = process_job(env, job, letterbox, np.asarray(cache[job["cache_key"]]))
            except (Exception, KeyboardInterrupt) as error:
                pipe.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
                return
            pipe.send({"status": "complete", "row": row})
    finally:
        if env is not None:
            env.close()
        if cache is not None:
            cache.close()
        pipe.close()


# ----- summary / report ---------------------------------------------------------------------------------
def summarize(rows, wall_seconds, workers, sampled=None):
    flat = [{**o, "set": r["set"], "group": r["group"]} for r in rows for o in r["rollouts"]]
    s = [r["sanity"] for r in rows]
    n = len(rows)
    sanity = {"anchors": n,
              "prefix_replay_exact_anchors": sum(x["prefix_replay_exact"] for x in s),
              "recorded_exact_anchors": sum(x["recorded_exact"] for x in s),
              "recorded_max_tilt_diff_deg": max(x["recorded_max_tilt_diff_deg"] for x in s),
              "recorded_max_qpos_diff": max(x["recorded_max_qpos_diff"] for x in s),
              "recorded_max_com_diff_m": max(x["recorded_max_com_diff_m"] for x in s),
              "recorded_class_mismatches": sum(x["recorded_class_mismatches"] for x in s),
              "recorded_unsafe_mismatches": sum(x["recorded_unsafe_mismatches"] for x in s),
              "recorded_actions_equal_anchors": sum(x["recorded_actions_equal"] for x in s),
              "anchor_frame_equals_cache_fraction": sum(x["anchor_cache"]["identical"] for x in s) / n,
              "anchor_frame_cache_max_abs_diff": max(x["anchor_cache"]["max_abs_diff"] for x in s),
              "prefix_frames_equal_cache": f"{sum(x['prefix_cache_identical_frames'] for x in s)}/{n * (plan.PREFIX + 1)}",
              "prefix_frames_cache_max_abs_diff": max(x["prefix_cache_max_abs_diff"] for x in s),
              "anchor_raw_png_identical_fraction": sum(x["raw_png"]["identical"] for x in s) / n,
              "anchor_raw_png_max_abs_diff": max(x["raw_png"]["max_abs_diff"] for x in s),
              "prefix_frames_raw_png_identical": f"{sum(x['prefix_raw_png_identical_frames'] for x in s)}/{n * (plan.PREFIX + 1)}",
              "prefix_frames_raw_png_max_abs_diff": max(x["prefix_raw_png_max_abs_diff"] for x in s),
              "anchor_images_equal_across_rollouts_anchors": sum(x["anchor_images_equal_across_rollouts"] for x in s),
              "anchor_state_matches_plan_anchors": sum(x["anchor_state_matches_plan"] for x in s)}
    counts = {"anchors_by_set": {}, "anchors_by_group": {}}
    for r in rows:
        counts["anchors_by_set"][r["set"]] = counts["anchors_by_set"].get(r["set"], 0) + 1
        counts["anchors_by_group"][r["group"]] = counts["anchors_by_group"].get(r["group"], 0) + 1
    ks = [r["lead_k"] for r in rows if r["lead_k"] is not None]
    return {"sanity": sanity, "counts": counts, "outcomes": plan.aggregate_outcomes(flat), "criteria_pools": plan.criteria_pools(flat),
            "worst_tracking": worst_tracking(rows),
            "fall_lead_k": {"n": len(ks), "min": min(ks) if ks else None, "median": float(np.median(ks)) if ks else None, "max": max(ks) if ks else None},
            "bytes": int(sum(r["bytes"] for r in rows)), "bytes_per_anchor": float(np.mean([r["bytes"] for r in rows])),
            "wall_seconds": wall_seconds, "workers": workers, "memory": memory_summary(rows, sampled),
            "anchor_seconds_median": float(np.median([r["wall_seconds"] for r in rows])),
            "anchors_per_minute": 60.0 * n / wall_seconds if wall_seconds else None}


def worst_tracking(rows, count=10):
    """The rollouts whose pinch ended farthest from its reference (anchor id, group, family, errors, contact, saturation)."""
    items = [(o["tracking_error_final_m"], r, o) for r in rows for o in r["rollouts"] if o.get("tracking_error_final_m") is not None]
    items.sort(key=lambda t: -t[0])
    return [{"anchor": r["id"], "group": r["group"], "rollout": o["rollout"], "family": o["family"],
             "final_error_cm": 100 * e, "max_error_cm": 100 * o["tracking_error_max_m"], "saturated_steps": o["saturated_steps"],
             "contact_steps": o["contact_steps"], "anchor_tilt_deg": o["anchor_tilt_deg"], "max_tilt_h24": o["max_tilt_h24"]}
            for e, r, o in items[:count]]


def memory_summary(rows, sampled=None):
    """Peak RSS per worker process (measured inside each worker after each anchor) and of this main process, plus the
    MemorySampler result (``sampled``: peak PSS summed over the job's processes = its real footprint).

    The sum of the per-process RSS peaks is an upper bound of the job's concurrent RSS (the peaks need not coincide and
    shared pages are counted once per process)."""
    main_pid, main_kb = os.getpid(), peak_rss_kb()
    per_worker = {}
    for r in rows:
        if r.get("worker_peak_rss_kb") is not None:
            per_worker[r["worker_pid"]] = max(per_worker.get(r["worker_pid"], 0), r["worker_peak_rss_kb"])
    workers_kb = {pid: kb for pid, kb in per_worker.items() if pid != main_pid}
    total = None if main_kb is None else main_kb + sum(workers_kb.values())
    return {"main_pid": main_pid, "main_peak_rss_kb": main_kb, "worker_peak_rss_kb": {str(k): v for k, v in sorted(workers_kb.items())},
            "worker_peak_rss_kb_max": max(workers_kb.values()) if workers_kb else None,
            "sum_of_process_peaks_kb": total, "sampled": sampled,
            "note": "sum_of_process_peaks_kb = main + every worker ru_maxrss: an upper bound of concurrent RSS; "
                    "sampled.peak_pss_sum_kb = peak of the PSS summed over main + workers (shared pages split): the real footprint"}


def _cell(text):
    return str(text).replace("|", "\\|")


def _pool_cell(p):
    return f"{p['all']} / {p['anchor_tilt_lt10']} / {p['anchor_tilt_lt10_no_saturated_action']}"


def _fmt(value, spec=".2f"):
    return "–" if value is None else format(value, spec)


def _wm_cell(w):
    return "–" if w is None else f"{w['median']:.1f} / {w['p90']:.1f} / {w['max']:.1f}"


def write_report(report_dir, summary, run):
    s, o = summary["sanity"], summary["outcomes"]
    mem = summary["memory"]
    sampled = mem.get("sampled") or {}
    gib = (lambda kb: "–" if kb is None else f"{kb / 1024**2:.2f} GiB")
    lines = ["# 随机动作分支（第 2 阶段）：从真实状态出发的仿真 rollout\n",
             f"- 数据集：`{run['output']}`；计划：`{run['plan']}`；锚点 {s['anchors']} 个，按集合 {json.dumps(summary['counts']['anchors_by_set'], ensure_ascii=False)}，按组 {json.dumps(summary['counts']['anchors_by_group'], ensure_ascii=False)}。",
             f"- 每个锚点 8 条 rollout（{', '.join(plan.FAMILIES)}），每条 = 恢复 a−{plan.PREFIX} 帧 + 逐位重放 {plan.PREFIX} 步记录动作 + {plan.HORIZON} 步控制。",
             f"- 渲染环境：{json.dumps(run['render_environment'], ensure_ascii=False)}；与来源数据集 manifest 一致：{json.dumps(run['render_environment_matches_source'], ensure_ascii=False)}。",
             f"- 用时 {summary['wall_seconds']:.0f} s（{summary['workers']} 个 worker，每锚点中位 {summary['anchor_seconds_median']:.1f} s）；磁盘 {summary['bytes'] / 1024**3:.2f} GiB（每锚点 {summary['bytes_per_anchor'] / 1024**2:.1f} MiB）。",
             f"- 内存（实测）：整个作业（主进程 + 全部 worker）的 PSS 之和峰值 **{gib(sampled.get('peak_pss_sum_kb'))}**（共享页按进程分摊 = 实际占用；"
             f"每 {sampled.get('interval_s', '–')} s 采样，共 {sampled.get('samples', 0)} 次，最多 {sampled.get('max_processes', 0)} 个进程）；"
             f"RSS 直接相加的峰值 {gib(sampled.get('peak_rss_sum_kb'))}（共享页重复计算，偏高）；采样期间系统 MemAvailable 最低 {gib(sampled.get('min_mem_available_kb'))}。"
             f"各进程自测 ru_maxrss：主进程 {gib(mem['main_peak_rss_kb'])}，worker 最大 {gib(mem['worker_peak_rss_kb_max'])}，全部相加 {gib(mem['sum_of_process_peaks_kb'])}（上界）。\n",
             "## 1. 复现与像素检查\n",
             f"- 前缀重放逐位一致的锚点：{s['prefix_replay_exact_anchors']}/{s['anchors']}（不一致会直接报错）。",
             f"- recorded 续推复现记录：完全一致 {s['recorded_exact_anchors']}/{s['anchors']}；倾角最大差 {s['recorded_max_tilt_diff_deg']:.3g}°，qpos 最大差 {s['recorded_max_qpos_diff']:.3g}，COM 最大差 {s['recorded_max_com_diff_m']:.3g} m，class_label 不一致 {s['recorded_class_mismatches']} 帧，unsafe 不一致 {s['recorded_unsafe_mismatches']} 帧；执行的动作与记录逐位相同 {s['recorded_actions_equal_anchors']}/{s['anchors']}。",
             f"- 锚点帧：重渲染 + letterbox 与缓存像素逐位相同的比例 **{s['anchor_frame_equals_cache_fraction']:.3f}**，最大绝对差 {s['anchor_frame_cache_max_abs_diff']}；前缀 {plan.PREFIX + 1} 帧与缓存逐位相同 {s['prefix_frames_equal_cache']}（最大差 {s['prefix_frames_cache_max_abs_diff']}）。",
             f"- 640×480 原始渲染与数据集 PNG：锚点帧逐位相同比例 {s['anchor_raw_png_identical_fraction']:.3f}（最大差 {s['anchor_raw_png_max_abs_diff']}），前缀 {plan.PREFIX + 1} 帧 {s['prefix_frames_raw_png_identical']}（最大差 {s['prefix_frames_raw_png_max_abs_diff']}）；8 条 rollout 的锚点图像彼此相同的锚点 {s['anchor_images_equal_across_rollouts_anchors']}/{s['anchors']}。以上都是本工具自己的检查。\n",
             "## 2. 结果计数（倾角 = 两根红柱取大，步 1..h；>30 为 unsafe）\n",
             "“锚点≥10°”= 锚点帧已有红柱 ≥10° 的 rollout 数（这些 rollout 的 ≥10° / >30° 多半是已倾斜的柱子保持倾斜，不是新扰动）。\n"]
    header = ("| 组 | 族 | rollout | 锚点≥10° | >30° h8 | >30° h16 | >30° h24 | ≥10° h8 | ≥10° h16 | ≥10° h24 | <5° h8 | <5° h16 | <5° h24 | "
              "含 class0 帧 | 有接触 | 夹爪最大位移 中位/p90 (cm) |")
    sep = "|" + "---|" * 16
    for title, blocks in (("全部", {"全部": o["overall"]}), ("按集合", o["by_set"]), ("按组", o["by_group"])):
        lines += [f"### {title}\n", header, sep]
        for name, fams in blocks.items():
            for fam, b in fams.items():
                if b is None:
                    continue
                lines.append(f"| {_cell(name)} | {fam} | {b['rollouts']} | {b['anchor_tilt_ge10']} | {b['over30_h8']} | {b['over30_h16']} | {b['over30_h24']} | "
                             f"{b['over10_h8']} | {b['over10_h16']} | {b['over10_h24']} | {b['below5_h8']} | {b['below5_h16']} | {b['below5_h24']} | "
                             f"{b['with_class0']} | {b['with_contact']} | {b['pinch_disp_max_cm']['median']:.2f} / {b['pinch_disp_max_cm']['p90']:.2f} |")
        lines.append("")
    lines += ["## 3. S1 / S2 的真值池（判据本身不在这里评估）\n",
              "每格 = 全部 / 其中锚点最差倾角 < 10° / 其中锚点 < 10° 且不含饱和动作。S1（h16）：正例 = 16 步内 >30°，负例 = 16 步内一直 <5°。"
              "S2（h8、h16）：正例 = h 步内 ≥10°，负例 = 其余。锚点已 ≥10° 的正例可以靠“已倾斜的柱子保持倾斜”得分，所以另列锚点 < 10° 的子集。\n",
              "| 族 | S1 正例 | S1 负例 | S2 h8 正例 | S2 h8 负例 | S2 h16 正例 | S2 h16 负例 | S1 正例中来自 B/fall |", "|---|---|---|---|---|---|---|---|"]
    for fam, p in summary["criteria_pools"].items():
        lines.append(f"| {fam} | {_pool_cell(p['S1_h16']['positives'])} | {_pool_cell(p['S1_h16']['negatives'])} | {_pool_cell(p['S2_h8']['positives'])} | "
                     f"{_pool_cell(p['S2_h8']['negatives'])} | {_pool_cell(p['S2_h16']['positives'])} | {_pool_cell(p['S2_h16']['negatives'])} | "
                     f"{p['S1_h16']['positives']['by_group'].get('B/fall', 0)} |")
    lines += ["", "## 4. 参考点跟踪、饱和动作、动作幅度、高度\n",
              f"- 跟踪误差 = |pinch[k+1] − reference[k]|（npz 里的 `tracking_error_m`）；“丢失” = 末步误差 > {100 * plan.TRACKING_LOST_M:g} cm。"
              "饱和 = 某步 dx/dy/dz 在 ±1（±5 cm）边界上（`action_saturated`）。|wm_action| 以缓存归一化的 scale（≈训练动作的标准差）为单位；recorded 一列给出记录动作本身的幅度作对照。"
              f"“参考点低于 9 cm”= 参考点低于 min(9 cm, 锚点高度)（exploration `_free` 的下限；本工具的参考框下限是 2.5 cm）。\n",
              "| 集合 | 族 | rollout | 末误差 >2 cm | >5 cm | 最大 (cm) | 丢失且含饱和 | 丢失且有接触 | 含饱和动作 | wm_action 绝对值最大 中位/p90/最大 | >5 | >10 | 参考点低于 9 cm | 夹爪到过 9 cm 以下 |",
              "|" + "---|" * 15]
    for name, fams in (("全部", o["overall"]), *o["by_set"].items()):
        for fam, b in fams.items():
            if b is None:
                continue
            t, w = b["tracking"], b["wm_action_absmax"]
            lines.append(f"| {_cell(name)} | {fam} | {b['rollouts']} | {'–' if t is None else t['final_gt2cm']} | {'–' if t is None else t['final_gt5cm']} | "
                         f"{'–' if t is None else _fmt(t['final_max_cm'])} | {'–' if t is None else t['final_gt2cm_with_saturated_action']} | "
                         f"{'–' if t is None else t['final_gt2cm_with_contact']} | {b['with_saturated_action']} | {_wm_cell(w)} | "
                         f"{'–' if w is None else w['gt5']} | {'–' if w is None else w['gt10']} | {b['reference_below_free_floor']} | {b['pinch_below_free_floor']} |")
    lines += ["", "末误差最大的 rollout：", "", "| 锚点 | 组 | r | 族 | 末误差 (cm) | 最大误差 (cm) | 饱和步数 | 接触步数 | 锚点倾角 (°) | h24 最大倾角 (°) |", "|" + "---|" * 10]
    for w in summary["worst_tracking"][:8]:
        lines.append(f"| {w['anchor']} | {_cell(w['group'])} | {w['rollout']} | {w['family']} | {w['final_error_cm']:.2f} | {w['max_error_cm']:.2f} | "
                     f"{w['saturated_steps']} | {w['contact_steps']} | {w['anchor_tilt_deg']:.1f} | {w['max_tilt_h24']:.1f} |")
    k = summary["fall_lead_k"]
    lines += ["", "## 5. 说明（测了什么、没测什么）\n",
              f"- B 集倒柱事件的提前量 k：{k['n']} 个，最小 {k['min']}、中位 {k['median']}、最大 {k['max']}（k > 24 时 recorded 续推在 24 步内不会越过 30°）。",
              "- 这里只有仿真真值：没有加载 world model、没有打分；S1–S3 的判定由后续评估脚本在这些 rollout 上做。第 3 节只数真值池的大小和构成。",
              "- recorded 续推必须复现记录（测了：逐步倾角、qpos、COM、class_label、unsafe、执行的动作）；hold / random / guard 没有记录可比，只保证动作律按定义执行。",
              "- 跟踪：合成场景的单元测试（无接触）要求末误差 < 2 cm；真实数据上**不做断言**，只在第 4 节报告。丢失跟踪的 rollout 里 PI 积分饱和，夹爪被正在倒的红柱带着走（有接触）；"
              "执行的动作照实记录，所以它们仍是合法的仿真真值，但这些饱和动作（|wm_action| 可达约 20）远在记录动作的范围之外，评估时宜把“含饱和动作”的 rollout 单独报告。",
              "- 像素：锚点前 11 帧（a−10..a）与缓存像素、与数据集 PNG 都逐位比对了；rollout 内 24 步的新图像没有缓存可比（新状态），其一致性依赖同一渲染后端与同一 letterbox 函数。",
              "- wm_action 用 `evaluate_bench_counterfactual.normalize_actions` 的 float32 算法（单元测试在训练 venv 里与该函数逐位比对）。",
              "- 悬停计时 / episode_failed 锁存不在 physics_state 里，从 a−10 帧重新开始；class_label 只依赖当前状态（unsafe、grasped、离目标距离、速度），不受影响（recorded 续推的 class_label 逐帧核对过）。",
              "- 随机方向在锚点处固定抽样（种子记录在计划里），不依赖 rollout 中的状态；guard 族首段指向锚点时较近的红柱，是否真的接触由仿真决定（见“有接触”列）。",
              "- 内存：PSS 峰值由主进程里的采样线程读 /proc/<pid>/smaps_rollup 得到（采样间隔内的尖峰可能漏掉）；ru_maxrss 是每个进程自己在每个锚点结束时读的，各进程峰值之和是上界，不是同一时刻的总量。"]
    (Path(report_dir) / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----- CLI ------------------------------------------------------------------------------------------------
def select_anchors(anchors, ids=None, keep_fraction=1.0):
    """All anchors, an explicit id list, or the first ceil(f * n) anchors of every group (plan order)."""
    if ids is not None:
        by_id = {a["id"]: a for a in anchors}
        missing = [i for i in ids if i not in by_id]
        if missing or len(set(ids)) != len(ids):
            raise ValueError(f"unknown or repeated anchor ids: {missing}")
        return [by_id[i] for i in sorted(ids)]
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    groups = {}
    for anchor in anchors:
        groups.setdefault(anchor["group"], []).append(anchor)
    keep = {id(a) for g in groups.values() for a in g[:int(np.ceil(keep_fraction * len(g)))]}
    return [a for a in anchors if id(a) in keep]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, required=True, help="stage-1 run directory (random_branches_plan.py)")
    p.add_argument("--output", type=Path, required=True, help="NEW dataset directory, e.g. data/random_branches_001")
    p.add_argument("--report", type=Path, required=True, help="NEW run directory for run.json / summary.json / REPORT_ZH.md")
    p.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    p.add_argument("--anchor-ids", default=None, help="comma-separated plan anchor ids (pilot); default all")
    p.add_argument("--keep-fraction", type=float, default=1.0, help="keep the first ceil(f*n) anchors of every group")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    output, report, plan_dir = a.output.resolve(), a.report.resolve(), a.plan.resolve()
    for path in (output, report):
        if path.exists():
            raise FileExistsError(f"Use a NEW directory: {path}")
    if output == report or output.is_relative_to(report) or report.is_relative_to(output):
        raise ValueError("dataset and report directories must be separate")
    plan_summary = read_json(plan_dir / "summary.json")
    if plan_summary.get("format") != plan.PLAN_FORMAT or (plan_dir / "FAILED.json").exists():
        raise ValueError(f"{plan_dir}: not a successful {plan.PLAN_FORMAT} run")
    files_sha = {name: sha256(plan_dir / name) for name in ("anchors.json", "cache_pixels.npz")}
    if files_sha != plan_summary["files_sha256"]:
        raise ValueError(f"{plan_dir}: anchors.json / cache_pixels.npz changed since the plan was written")
    doc = read_json(plan_dir / "anchors.json")
    if (doc["prefix"], doc["horizon"], tuple(doc["families"])) != (plan.PREFIX, plan.HORIZON, plan.FAMILIES):
        raise ValueError("plan constants differ from this code")
    ids = None if a.anchor_ids is None else [int(t) for t in a.anchor_ids.split(",") if t.strip()]
    anchors = select_anchors(doc["anchors"], ids, a.keep_fraction)
    for anchor in anchors:
        source = Path(anchor["source"]["dataset_dir"]).resolve()
        if output.is_relative_to(source) or report.is_relative_to(source):
            raise ValueError("outputs must be outside the source datasets")
    estimate_bytes = len(anchors) * BYTES_PER_ANCHOR_ESTIMATE
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < estimate_bytes + DISK_RESERVE_BYTES:
        raise OSError("not enough free disk space")
    workers = min(a.workers, len(anchors))
    render_env = {k: os.environ.get(k) for k in RENDER_KEYS}
    source_manifests = sorted({(anchor["source"]["dataset_dir"], anchor["source"]["dataset_manifest_sha256"]) for anchor in anchors})
    matches = {}
    for folder, digest in source_manifests:
        if sha256(Path(folder) / "manifest.json") != digest:
            raise ValueError(f"{folder}: manifest changed since the plan")
        recorded = read_json(Path(folder) / "manifest.json").get("execution", {}).get("render_environment")
        matches[folder] = (recorded == render_env) if recorded else None
    root = Path(__file__).resolve().parent
    run = {"format": FORMAT, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "plan": str(plan_dir), "output": str(output), "report": str(report),
           "input_sha256": {**{str(plan_dir / k): v for k, v in files_sha.items()}, str(plan_dir / "summary.json"): sha256(plan_dir / "summary.json")},
           "source_note": "per-episode sha256 of trajectory.npz / metadata.json are recorded in anchors.json and re-checked before each anchor",
           "source_manifest_sha256": {f: d for f, d in source_manifests},
           "implementation_sha256": {**implementation_hashes(ROOT_FILES, package_dir=root), **implementation_hashes(WM_FILES)},
           "anchors": [x["id"] for x in anchors], "workers": workers, "render_environment": render_env,
           "render_environment_matches_source": matches, "state_spec": STATE_SPEC,
           "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                        "pillow": version("pillow"), "ogbench": version("ogbench")},
           "estimate": {"bytes": estimate_bytes, "seconds": len(anchors) * SECONDS_PER_ANCHOR_ESTIMATE / workers},
           "reporting_constants": {"free_floor_z_m": plan.FREE_FLOOR_Z_M, "disturbed_deg": plan.DISTURBED_DEG,
                                   "tracking_lost_m": plan.TRACKING_LOST_M, "saturation_tol": plan.SATURATION_TOL},
           "world_model_loaded": False, "test_episodes_opened": False}
    output.mkdir(parents=True)
    report.mkdir(parents=True)
    write_new_json(report / "run.json", run)
    jobs = [{"id": x["id"], "anchor": x, "normalization": doc["normalization"], "output_dir": str(output),
             "cache_pixels": str(plan_dir / "cache_pixels.npz"), "cache_key": f"a{x['id']:04d}"} for x in anchors]
    rows, started = [], time.perf_counter()

    def record(row):
        rows.append(row)
        elapsed = time.perf_counter() - started
        s = row["sanity"]
        eta = elapsed / len(rows) * (len(jobs) - len(rows))
        print(f"[{len(rows)}/{len(jobs)}] anchor {row['id']:04d} {row['group']} ep{row['cache_episode_id']} f{row['frame']} | "
              f"recorded_exact={s['recorded_exact']} cache_px={s['anchor_cache']['identical']}/{s['anchor_cache']['max_abs_diff']} | "
              f"max30@24={sum(o['max_tilt_h24'] > 30 for o in row['rollouts'])} | {row['wall_seconds']:.1f}s | "
              f"{row['bytes'] / 1e6:.1f} MB | elapsed {elapsed:.0f}s eta {eta:.0f}s", flush=True)

    sampler = MemorySampler().start()
    try:
        print(f"{len(jobs)} anchors x {len(plan.FAMILIES)} rollouts with {workers} worker(s)", flush=True)
        if workers == 1:
            env = ExplorationScene()
            letterbox, _ = import_letterbox()
            try:
                with np.load(plan_dir / "cache_pixels.npz", allow_pickle=False) as cache:
                    for job in jobs:
                        record(process_job(env, job, letterbox, np.asarray(cache[job["cache_key"]])))
            finally:
                env.close()
        else:
            run_workers(jobs, _worker, workers, record)
        rows.sort(key=lambda r: r["id"])
        sampled = sampler.stop()
        _, how = import_letterbox()
        summary = {"format": FORMAT, **summarize(rows, time.perf_counter() - started, workers, sampled), "letterbox_import": how,
                   "child_thread_limits": CHILD_THREAD_LIMITS if workers > 1 else {}}
        manifest = {"format": FORMAT, "status": "complete", "plan": str(plan_dir), "plan_files_sha256": files_sha,
                    "families": list(plan.FAMILIES), "prefix": plan.PREFIX, "horizon": plan.HORIZON, "image_size": IMAGE_SIZE,
                    "camera": CAMERA, "alignment": "pixels[r, i] -- actions[r, i] --> pixels[r, i+1]; index 0 = anchor frame a",
                    "prefix_alignment": "prefix_pixels[j] = frame a-10+j (j = 10 is the anchor); prefix_actions[j] = recorded action a-10+j",
                    "wm_action": {"columns": list(plan.WM_ACTION_COLUMNS), "normalization": doc["normalization"],
                                  "rule": "((float32(action)[..., [0,1,2,4]] - float32(mean)) / float32(scale)).astype(float32)",
                                  "identical_to": "wm_training.evaluate_bench_counterfactual.normalize_actions"},
                    "diagnostic_arrays": {"tracking_error_m": "(R, 24) |pinch[r, k+1] - reference_xyz[r, k]| in m; NaN for 'recorded'",
                                          "action_saturated": "(R, 24) bool: |actions[r, k, 0:3]| on the +-1 (= +-5 cm) bound"},
                    "reporting_constants": {"free_floor_z_m": plan.FREE_FLOOR_Z_M, "disturbed_deg": plan.DISTURBED_DEG,
                                            "tracking_lost_m": plan.TRACKING_LOST_M, "saturation_tol": plan.SATURATION_TOL},
                    "render_environment": render_env, "render_environment_matches_source": matches,
                    "versions": run["versions"], "anchors": [{"plan_anchor": next(x for x in anchors if x["id"] == r["id"]), **r} for r in rows],
                    "summary": summary}
        write_new_json(output / "manifest.json", manifest)
        write_report(report, summary, run)
        write_new_json(report / "summary.json", {**summary, "manifest_sha256": sha256(output / "manifest.json")})
        print(json.dumps(native(summary["sanity"]), ensure_ascii=False), flush=True)
        return summary
    except BaseException as error:
        sampler.stop()
        failure = {"error": f"{type(error).__name__}: {error}", "anchors_done": len(rows),
                   "note": "partial output; npz files of finished anchors are complete but the dataset has no manifest"}
        write_new_json(report / "FAILED.json", failure)
        write_new_json(output / "FAILED.json", failure)
        raise


if __name__ == "__main__":
    main()
