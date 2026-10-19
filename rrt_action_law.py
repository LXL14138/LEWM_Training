"""The RRT pilot's action law (user decision 2026-09-27), as one shared numpy-only module.

Planner, data collector (collect_rrt_branches.py) and tests all use this file, so the actions the world model is
trained on, the actions the planner imagines and the actions the robot executes come from ONE implementation.

Law (binding user decision; see RRT_ACTION_LAW.md for the reasoning and the measured numbers):

* gripper translation only, at a FIXED 2 cm/s: the reference advances 1 mm per 0.05 s control step; yaw 0; the
  gripper command is held (the pilot starts with the green block grasped; the planner never opens or closes it);
* edges of 8 steps with a constant unit direction; directions change only between edges; 'hold' (anchored at the
  pinch where it starts) is always available; the safety check of a candidate edge imagines edge (8) + hold (12);
* PI reference tracking as the data controllers use (bench_restore_check.AnchoredHold: P = 1, I = 1) WITH
  anti-windup (constants below): when the gripper lags its reference by more than LAG_MARGIN_M the reference stops
  advancing and the integral stops winding further (it may still unwind); the integral is always clamped to
  +-INTEGRAL_MAX_M per axis; the reference is never allowed to lead the pinch by more than LEAD_MAX_M (if contact
  drags the gripper away, the reference is dragged along instead of winding the action up);
* a hold always starts fresh at the CURRENT pinch with a zero integral (exactly the AnchoredHold start), so choosing
  'hold' stops any push at once; an edge continues the previous edge's reference and integral while the gripper keeps
  up (one continuous reference polyline, like random_branches' ReferenceTracker, so the pinch really moves at 1 mm per
  step), else it starts fresh at the pinch (LawRunner / continues);
* nominal actions (what the planner feeds the WM before executing) = this law driving the calibrated free-space
  plant model (NominalModel, NOMINAL_MODELS 'grasped' / 'empty', calibrate_rrt_action_law.py);
* reference clipped to the data-collection box (random_branches_plan.REF_BOX_LOW/HIGH: x/y and the upper z of
  exploration.py _free, the lower z = the lowest pinch height exploration.py _grasp commands); a reference that
  starts outside the box never jumps and never moves further out (per-axis box widened to the current reference);
* executed env action = PI output (metres) -> clipped to the scene's +-5 cm action range -> normalised like
  ogbench normalize_action -> dx/dy/dz clipped so that the WM-normalised action (float32 recipe of
  evaluate_bench_counterfactual.normalize_actions with the data/wm_full_001/index.json normalization) stays inside
  B's training range (normalised dx [-6.93, 3.15], dy [-6.97, 5.24], dz [-5.51, 2.67]).

Nothing here imports MuJoCo: a "plant" is anything that gives the pinch position before each action. The
simulation tests (tests/test_rrt_action_law.py) run the law in the real scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path

import numpy as np

LAW_FORMAT = "rrt_action_law_v1"
SPEED_MPS = 0.02                     # fixed gripper speed (user decision)
CONTROL_DT = 0.05                    # scene control step (s)
STEP_M = SPEED_MPS * CONTROL_DT      # 1 mm reference advance per step
EDGE_STEPS = 8                       # one edge = 8 steps with one direction
CHECK_HOLD_STEPS = 12                # safety check = edge (8) + hold (12)
CHECK_STEPS = EDGE_STEPS + CHECK_HOLD_STEPS
GAIN_P, GAIN_I = 1.0, 1.0            # = bench_restore_check.HOLD_GAIN_P / HOLD_GAIN_I (checked by the tests)
# Anti-windup (chosen and measured, see RRT_ACTION_LAW.md section 2):
LAG_MARGIN_M = 0.005                 # |reference - pinch| above this: reference stops advancing, integral stops winding further
                                     #   (free-space lag measured <= 2.5 mm, so the margin is 2x what free motion needs)
INTEGRAL_MAX_M = 0.012               # |integral| per axis never exceeds this (free-space need measured: x 5.9, y 4.1, z 7.8 mm)
LEAD_MAX_M = 0.010                   # the reference never leads the pinch by more than this (dragged along otherwise)
# ogbench manipspace action range [dx, dy, dz, yaw, dgripper] (ManipSpaceEnv: action_range); the scene's
# normalize_action is 2 * (a - low) / (high - low) - 1 with low = -range, high = +range (checked against the env in the tests).
ACTION_RANGE = np.array([0.05, 0.05, 0.05, 0.3, 1.0])
ACTION_XYZ_LIMIT_M = 0.05            # = bench_restore_check.ACTION_XYZ_LIMIT_M
# Reference box = random_branches_plan.REF_BOX_LOW / REF_BOX_HIGH (checked by the tests), intersected with the scene
# workspace (ogbench _workspace_bounds).
BOX_LOW = np.array([0.32, -0.10, 0.025])
BOX_HIGH = np.array([0.50, 0.10, 0.25])
WORKSPACE_LOW = np.array([0.25, -0.35, 0.02])
WORKSPACE_HIGH = np.array([0.60, 0.35, 0.35])
# B's training range of the WM-normalised dx, dy, dz (user decision; measured in runs/data_coverage_001/summary.json
# wm_range_B_training: min [-6.92961, -6.965268, -5.506015], max [3.149974, 5.242077, 2.672824]).
WM_RANGE_LOW = np.array([-6.93, -6.97, -5.51])
WM_RANGE_HIGH = np.array([3.15, 5.24, 2.67])
WM_BOUND_MARGIN = 1e-4               # the env-unit bounds sit this far (WM units) inside the range, then are verified
WM_ACTION_COLUMNS = (0, 1, 2, 4)
SEGMENT_KINDS = ("edge", "hold")


# ----- small helpers ------------------------------------------------------------------------------------------
def unit(vector):
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("direction must be three finite numbers")
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("zero direction")
    return v / n


def _vec3(value, name):
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must be three finite numbers")
    return v.copy()


def normalize_env(physical5):
    """ogbench ManipSpaceEnv.normalize_action on a physical 5-vector (same float operations, then clipped to [-1, 1])."""
    a = np.asarray(physical5, dtype=float)
    low, high = -ACTION_RANGE, ACTION_RANGE
    return np.clip(2 * (a - low) / (high - low) - 1, -1, 1)


def wm_normalize(actions5, normalization):
    """Env actions (..., 5) -> WM input (..., 4); float32 recipe of evaluate_bench_counterfactual.normalize_actions.

    Identical arithmetic to random_branches_plan.normalize_wm_actions (the tests compare both bit for bit)."""
    a = np.asarray(actions5, dtype=np.float32)
    if a.shape[-1] != 5:
        raise ValueError("expected 5 action columns")
    mean, scale = _normalization_arrays(normalization)
    return ((a[..., list(WM_ACTION_COLUMNS)] - mean.astype(np.float32)) / scale.astype(np.float32)).astype(np.float32)


def _normalization_arrays(normalization):
    mean, scale = np.asarray(normalization["mean"], float), np.asarray(normalization["scale"], float)
    if mean.shape != (4,) or scale.shape != (4,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("normalization needs 4 finite means and 4 positive scales")
    return mean, scale


def load_wm_normalization(index_path):
    """{'mean', 'scale', 'names', 'source', 'sha256'} from a cache index.json (data/wm_full_001/index.json)."""
    import hashlib
    path = Path(index_path)
    raw = path.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("status") != "complete" or list(doc.get("source_action_columns", [])) != [0, 1, 2, 4]:
        raise ValueError(f"{path}: not a complete cache index with action columns [0, 1, 2, 4]")
    norm = doc["normalization"]
    _normalization_arrays(norm)
    if list(norm.get("names", [])) != ["dx", "dy", "dz", "dgripper"]:
        raise ValueError(f"{path}: unexpected normalization names {norm.get('names')}")
    return {"names": list(norm["names"]), "mean": [float(v) for v in norm["mean"]], "scale": [float(v) for v in norm["scale"]],
            "source": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def wm_env_bounds(normalization, wm_low=WM_RANGE_LOW, wm_high=WM_RANGE_HIGH, margin=WM_BOUND_MARGIN):
    """Env-unit bounds (low, high) of dx, dy, dz whose WM-normalised value lies inside [wm_low, wm_high].

    The bounds sit ``margin`` WM units inside the range and are then checked with the float32 recipe (float32
    conversion, subtraction and division by a positive constant are monotone, so checking the two ends suffices).
    They are also intersected with the env's [-1, 1]."""
    mean, scale = _normalization_arrays(normalization)
    wm_low, wm_high = np.asarray(wm_low, float), np.asarray(wm_high, float)
    if wm_low.shape != (3,) or wm_high.shape != (3,) or np.any(wm_low >= wm_high) or margin < 0:
        raise ValueError("bad WM range")
    low = np.maximum(mean[:3] + (wm_low + margin) * scale[:3], -1.0)
    high = np.minimum(mean[:3] + (wm_high - margin) * scale[:3], 1.0)
    if np.any(low >= high):
        raise ValueError("empty WM action range")
    probe = np.zeros((2, 5))
    probe[0, :3], probe[1, :3] = low, high
    z = wm_normalize(probe, normalization)[:, :3]
    if np.any(z < wm_low) or np.any(z > wm_high):
        raise RuntimeError("float32 check of the WM action bounds failed")
    return low, high


def box_step(reference, delta, low=BOX_LOW, high=BOX_HIGH):
    """reference + delta clipped per axis to [min(low, reference), max(high, reference)].

    A reference inside the box stays inside; one outside never jumps and never moves further out on any axis."""
    ref = _vec3(reference, "reference")
    lo, hi = np.minimum(np.asarray(low, float), ref), np.maximum(np.asarray(high, float), ref)
    target = ref + np.asarray(delta, float)
    out = np.clip(target, lo, hi)
    return out, bool(np.any(out != target))


# ----- configuration --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LawConfig:
    """Every constant of the law; ``normalization`` = data/wm_full_001/index.json normalization (mean / scale)."""
    normalization: dict
    speed_mps: float = SPEED_MPS
    dt: float = CONTROL_DT
    edge_steps: int = EDGE_STEPS
    gain_p: float = GAIN_P
    gain_i: float = GAIN_I
    lag_margin_m: float = LAG_MARGIN_M
    integral_max_m: float = INTEGRAL_MAX_M
    lead_max_m: float = LEAD_MAX_M
    box_low: tuple = tuple(BOX_LOW)
    box_high: tuple = tuple(BOX_HIGH)
    workspace_low: tuple = tuple(WORKSPACE_LOW)
    workspace_high: tuple = tuple(WORKSPACE_HIGH)
    wm_low: tuple = tuple(WM_RANGE_LOW)
    wm_high: tuple = tuple(WM_RANGE_HIGH)
    _bounds: tuple = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for name in ("speed_mps", "dt", "gain_p", "gain_i", "lag_margin_m", "integral_max_m", "lead_max_m"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if type(self.edge_steps) is not int or self.edge_steps <= 0:
            raise ValueError("edge_steps must be a positive integer")
        if self.speed_mps * self.dt > ACTION_XYZ_LIMIT_M:
            raise ValueError("speed exceeds the per-step action range")
        if self.lead_max_m < self.lag_margin_m + self.step_m:
            raise ValueError("lead_max_m must exceed lag_margin_m + one reference step (else a free edge is dragged)")
        low, high = self.box()
        if np.any(low >= high):
            raise ValueError("empty reference box")
        object.__setattr__(self, "_bounds", wm_env_bounds(self.normalization, self.wm_low, self.wm_high))

    @property
    def step_m(self):
        return self.speed_mps * self.dt

    def box(self):
        """Reference box = [box_low, box_high] intersected with the scene workspace."""
        low = np.maximum(np.asarray(self.box_low, float), np.asarray(self.workspace_low, float))
        high = np.minimum(np.asarray(self.box_high, float), np.asarray(self.workspace_high, float))
        return low, high

    def env_bounds(self):
        return self._bounds[0].copy(), self._bounds[1].copy()

    def to_json(self):
        out = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        out["normalization"] = {k: self.normalization[k] for k in ("mean", "scale") if k in self.normalization}
        low, high = self.env_bounds()
        out.update(format=LAW_FORMAT, step_m=self.step_m, env_action_xyz_low=low.tolist(), env_action_xyz_high=high.tolist(),
                   action_range=ACTION_RANGE.tolist())
        return out


def env_action(u_xyz, grip, config):
    """PI output (metres) -> env action (5,) and flags.

    Order: clip to the +-5 cm action range, normalise (ogbench formula), clip dx/dy/dz to the env-unit bounds of B's
    WM training range, yaw 0, gripper = ``grip``."""
    u = _vec3(u_xyz, "u_xyz")
    if not np.isfinite(grip) or abs(float(grip)) > 1:
        raise ValueError("grip must be a finite value in [-1, 1]")
    physical = np.array([*np.clip(u, -ACTION_XYZ_LIMIT_M, ACTION_XYZ_LIMIT_M), 0.0, 0.0])
    action = normalize_env(physical)
    low, high = config.env_bounds()
    clipped = np.clip(action[:3], low, high)
    flags = {"range_clipped": np.abs(u) > ACTION_XYZ_LIMIT_M, "wm_clipped": clipped != action[:3]}
    action[:3] = clipped
    action[3] = 0.0
    action[4] = float(grip)
    if not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise RuntimeError("invalid env action")
    return action, flags


# ----- one segment ------------------------------------------------------------------------------------------------
STEP_LOG_KEYS = ("reference_xyz", "error_m", "integral_m", "u_m", "lag_m", "reference_frozen", "integral_clamped",
                 "reference_dragged", "box_clipped", "range_clipped", "wm_clipped")


class SegmentTracker:
    """One edge (reference advances STEP_M * direction per step for ``steps`` steps) or one hold (fixed reference).

    A fresh segment starts at ``start_pinch`` with the reference on it and a zero integral (the AnchoredHold start).
    An edge that CONTINUES the previous edge (LawRunner decides) instead starts from ``start_reference`` /
    ``start_integral`` = the previous edge's reference and integral, so consecutive edges form one continuous reference
    polyline tracked by one PI (like random_branches' ReferenceTracker) and the pinch moves at the reference speed.
    ``action(pinch)`` takes the CURRENT pinch position and returns the env action; per step it

      1. lag = |reference_{k-1} - pinch_k| (the previous step's tracking error); frozen = lag > lag_margin;
      2. edge: the reference advances by STEP_M * direction (box-clipped, box_step) unless frozen;
      3. drag: if |reference - pinch| > lead_max, the reference is pulled to pinch + lead_max * unit(reference - pinch);
      4. error e = reference - pinch; integral += e, except while frozen, where an axis only integrates if that
         shrinks |integral| (unwinding is allowed, winding further is not); then clipped to +-integral_max per axis;
      5. u = P * e + I * integral -> env_action (range clip, normalisation, WM-range clip).
    """

    def __init__(self, kind, start_pinch, direction=None, steps=None, grip=1.0, config=None, start_reference=None, start_integral=None):
        if kind not in SEGMENT_KINDS:
            raise ValueError(f"kind must be one of {SEGMENT_KINDS}")
        if config is None:
            raise ValueError("a LawConfig is required (it carries the WM normalization)")
        self.kind, self.config = kind, config
        self.start = _vec3(start_pinch, "start_pinch")
        if kind == "edge":
            if direction is None:
                raise ValueError("an edge needs a direction")
            self.direction = unit(direction)
            self.steps = config.edge_steps if steps is None else steps
        else:
            if direction is not None:
                raise ValueError("a hold has no direction")
            self.direction = None
            self.steps = steps
        if self.steps is not None and (type(self.steps) is not int or self.steps <= 0):
            raise ValueError("steps must be a positive integer (or None for an open-ended hold)")
        if not np.isfinite(grip) or abs(float(grip)) > 1:
            raise ValueError("grip must be a finite value in [-1, 1]")
        self.grip = float(grip)
        if (start_reference is None) != (start_integral is None):
            raise ValueError("start_reference and start_integral go together")
        if start_reference is not None and kind != "edge":
            raise ValueError("only an edge can continue a previous edge")
        self.continued = start_reference is not None
        self.reference = self.start.copy() if start_reference is None else _vec3(start_reference, "start_reference")
        self.integral = np.zeros(3) if start_integral is None else _vec3(start_integral, "start_integral")
        if np.any(np.abs(self.integral) > config.integral_max_m + 1e-15):
            raise ValueError("start_integral exceeds the integral clamp")
        self.k = 0
        self.log = {key: [] for key in STEP_LOG_KEYS}

    @property
    def finished(self):
        return self.steps is not None and self.k >= self.steps

    def action(self, pinch):
        if self.finished:
            raise RuntimeError("segment already finished")
        c = self.config
        pinch = _vec3(pinch, "pinch")
        lag = float(np.linalg.norm(self.reference - pinch))
        frozen = lag > c.lag_margin_m
        reference, box_clipped = self.reference.copy(), False
        if self.kind == "edge" and not frozen:
            low, high = c.box()
            reference, box_clipped = box_step(reference, c.step_m * self.direction, low, high)
        lead = reference - pinch
        lead_norm = float(np.linalg.norm(lead))
        dragged = lead_norm > c.lead_max_m
        if dragged:
            reference = pinch + lead * (c.lead_max_m / lead_norm)
        error = reference - pinch
        if frozen:
            raw = np.where(error * self.integral < 0, self.integral + error, self.integral)
        else:
            raw = self.integral + error
        integral = np.clip(raw, -c.integral_max_m, c.integral_max_m)
        u = c.gain_p * error + c.gain_i * integral
        action, flags = env_action(u, self.grip, c)
        self.reference, self.integral = reference, integral
        self.k += 1
        for key, value in (("reference_xyz", reference.copy()), ("error_m", error), ("integral_m", integral.copy()), ("u_m", u),
                           ("lag_m", lag), ("reference_frozen", frozen), ("integral_clamped", bool(np.any(raw != integral))),
                           ("reference_dragged", dragged), ("box_clipped", box_clipped),
                           ("range_clipped", bool(flags["range_clipped"].any())), ("wm_clipped", flags["wm_clipped"].copy())):
            self.log[key].append(value)
        return action


# ----- sequences of segments ---------------------------------------------------------------------------------------
def edge(direction):
    return ("edge", unit(direction))


def hold(steps):
    if type(steps) is not int or steps <= 0:
        raise ValueError("hold steps must be a positive integer")
    return ("hold", int(steps))


def validate_segments(segments):
    out = []
    for seg in segments:
        if not isinstance(seg, (tuple, list)) or len(seg) != 2 or seg[0] not in SEGMENT_KINDS:
            raise ValueError(f"bad segment {seg!r}")
        out.append(edge(seg[1]) if seg[0] == "edge" else hold(seg[1]))
    if not out:
        raise ValueError("empty segment list")
    return out


def sequence_steps(segments, config):
    return int(sum(config.edge_steps if kind == "edge" else arg for kind, arg in validate_segments(segments)))


def edge_then_hold(direction, hold_steps=CHECK_HOLD_STEPS):
    """The safety check's candidate: one edge (8 steps) then hold (12 steps)."""
    return [edge(direction), hold(hold_steps)]


def continues(previous, pinch, config):
    """True if a new edge continues ``previous`` (a carry state: {'kind', 'reference', 'integral', 'dragged'}).

    Only an edge continues an edge, and only while the gripper keeps up (|reference - pinch| <= lag_margin and the
    last step was not dragged); otherwise (after a hold, at the start, or when blocked / pushed) the new edge starts
    fresh at the pinch with a zero integral, so a wound-up push never carries into a new direction."""
    if previous is None or previous["kind"] != "edge" or previous["dragged"]:
        return False
    return float(np.linalg.norm(np.asarray(previous["reference"], float) - _vec3(pinch, "pinch"))) <= config.lag_margin_m


class LawRunner:
    """Runs a list of segments.

    A hold always starts fresh at the pinch where it begins (AnchoredHold). An edge continues the previous edge's
    reference and integral when ``continues`` allows it, else it starts fresh at the pinch. ``previous`` (a
    ``carry_state()`` of an earlier runner) lets a planner that builds one runner per decision keep the chain.
    ``action(pinch)`` -> env action (5,); ``records()`` -> per-step arrays (segment index, continued flag, the
    SegmentTracker log fields)."""

    def __init__(self, segments, grip=1.0, config=None, previous=None):
        if config is None:
            raise ValueError("a LawConfig is required")
        self.segments, self.grip, self.config = validate_segments(segments), float(grip), config
        self.total_steps = sequence_steps(self.segments, config)
        self.index, self.tracker, self.k = -1, None, 0
        self.trackers, self.segment_of_step = [], []
        self.previous = None if previous is None else dict(previous)

    @property
    def finished(self):
        return self.k >= self.total_steps

    def carry_state(self):
        """What a following edge may continue from (None before the first step)."""
        t = self.tracker
        if t is None:
            return self.previous
        return {"kind": t.kind, "reference": t.reference.copy(), "integral": t.integral.copy(),
                "dragged": bool(t.log["reference_dragged"][-1]) if t.log["reference_dragged"] else False}

    def action(self, pinch):
        if self.finished:
            raise RuntimeError("sequence already finished")
        if self.tracker is None or self.tracker.finished:
            previous = self.carry_state()
            self.index += 1
            kind, arg = self.segments[self.index]
            if kind == "hold":
                self.tracker = SegmentTracker("hold", pinch, steps=arg, grip=self.grip, config=self.config)
            elif continues(previous, pinch, self.config):
                self.tracker = SegmentTracker("edge", pinch, direction=arg, grip=self.grip, config=self.config,
                                              start_reference=previous["reference"], start_integral=previous["integral"])
            else:
                self.tracker = SegmentTracker("edge", pinch, direction=arg, grip=self.grip, config=self.config)
            self.trackers.append(self.tracker)
        self.segment_of_step.append(self.index)
        self.k += 1
        return self.tracker.action(pinch)

    def records(self):
        out = {"segment_index": np.asarray(self.segment_of_step, dtype=np.int64),
               "segment_continued": np.asarray([t.continued for t in self.trackers], dtype=bool)}
        for key in STEP_LOG_KEYS:
            values = [v for t in self.trackers for v in t.log[key]]
            out[key] = np.asarray(values)
        return out


def run_sequence(plant, segments, grip=1.0, config=None, previous=None):
    """Drive ``plant`` (object with .pinch() -> (3,) and .step(action5)) through ``segments``.

    Returns {'actions': (N, 5), 'pinch_xyz': (N + 1, 3), 'carry_state': ..., **LawRunner.records()}."""
    runner = LawRunner(segments, grip, config, previous)
    actions, pinches = [], [np.asarray(plant.pinch(), float).copy()]
    while not runner.finished:
        action = runner.action(pinches[-1])
        plant.step(action)
        actions.append(action)
        pinches.append(np.asarray(plant.pinch(), float).copy())
    return {"actions": np.asarray(actions), "pinch_xyz": np.asarray(pinches), "carry_state": runner.carry_state(), **runner.records()}


# ----- nominal actions (what the planner feeds to the world model) -------------------------------------------------
@dataclass(frozen=True)
class NominalModel:
    """Calibrated free-space plant for the NOMINAL actions (RRT_ACTION_LAW.md section 3).

    With v_k = pinch_{k+1} - pinch_k (metres per step) and u_k the physical xyz of the executed action k (metres):
        v_k = G u_k + M v_{k-1} + b          (G, M: 3 x 3, b: 3)
    fitted by least squares (fit_nominal_model) on free-space executions of this law in the real scene. The arm is a
    sluggish position servo (the pinch covers about 40 % of the commanded offset per step, gravity sags it about
    1 mm per step), so an 'ideal follower' model would be badly wrong. ``prev_velocity`` = the pinch velocity per
    step when the sequence starts (the motion left over from the previous segment)."""
    gain: tuple
    memory: tuple
    bias: tuple
    source: str = ""

    def __post_init__(self):
        for name, shape in (("gain", (3, 3)), ("memory", (3, 3)), ("bias", (3,))):
            v = np.asarray(getattr(self, name), float)
            if v.shape != shape or not np.isfinite(v).all():
                raise ValueError(f"NominalModel.{name} must be finite with shape {shape}")

    def step(self, velocity, u):
        return np.asarray(self.gain) @ np.asarray(u, float) + np.asarray(self.memory) @ np.asarray(velocity, float) + np.asarray(self.bias)

    def steady_velocity(self, direction, config, grip=1.0):
        """Model pinch velocity at the end of one edge in ``direction`` started at rest (= what a following segment inherits)."""
        plant = ModelPlant(self, np.array([0.41, 0.0, 0.17]), np.zeros(3))
        run_sequence(plant, [edge(direction)], grip, config)
        return plant.velocity.copy()

    def to_json(self):
        return {"gain": np.asarray(self.gain).tolist(), "memory": np.asarray(self.memory).tolist(), "bias": np.asarray(self.bias).tolist(),
                "source": self.source}


def calibration_rows(runs):
    """(X, Y) regression rows from free-space runs: dicts with 'actions' (N, 5), 'pinch_xyz' (N + 1, 3), 'v0' (3,)."""
    X, Y = [], []
    for r in runs:
        actions, pinch = np.asarray(r["actions"], float), np.asarray(r["pinch_xyz"], float)
        if actions.ndim != 2 or actions.shape[1] != 5 or pinch.shape != (len(actions) + 1, 3):
            raise ValueError("calibration run needs actions (N, 5) and pinch_xyz (N + 1, 3)")
        u = unnormalize_env(actions)[:, :3]
        v = np.diff(pinch, axis=0)
        previous = np.vstack([_vec3(r["v0"], "v0"), v[:-1]])
        X.append(np.concatenate([u, previous, np.ones((len(u), 1))], 1))
        Y.append(v)
    return np.concatenate(X), np.concatenate(Y)


def fit_nominal_model(runs, source=""):
    """Least-squares NominalModel from free-space runs (see calibration_rows); returns (model, residual rms per axis)."""
    X, Y = calibration_rows(runs)
    if len(X) < 20:
        raise ValueError("too few calibration steps")
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)                       # (7, 3): v = X @ coef
    residual = Y - X @ coef
    model = NominalModel(gain=tuple(map(tuple, coef[0:3].T)), memory=tuple(map(tuple, coef[3:6].T)), bias=tuple(coef[6]), source=source)
    return model, np.sqrt((residual ** 2).mean(0))


def unnormalize_env(actions5):
    """ogbench ManipSpaceEnv.unnormalize_action: normalized (..., 5) -> physical (..., 5)."""
    a = np.asarray(actions5, float)
    low, high = -ACTION_RANGE, ACTION_RANGE
    return 0.5 * (a + 1) * (high - low) + low


# Calibrated constants, copied at full precision from runs/rrt_action_law_calib_002/summary.json 'models'
# (calibrate_rrt_action_law.py on this final law; runs/rrt_action_law_calib_001 used an earlier per-segment-reset
# variant and is superseded; RRT_ACTION_LAW.md section 3). 'grasped' = green block held (the pilot), 'empty' = empty gripper.
NOMINAL_MODELS = {
    "empty": NominalModel(
        gain=((0.4526032605210534, 0.0253966488710223, -0.14104365999671414), (0.006071931401552425, 0.3997783582803455, 0.030365342723093876),
              (0.04366485463713483, -0.024357553882271635, 0.3237926532235855)),
        memory=((0.021745147978835797, -0.05488464911932027, 0.11139438332998602), (-0.041290785978810716, 0.07292274016411811, -0.008843230556944645),
                (-0.07890599126235094, 0.05393676952263291, 0.13404815552097749)),
        bias=(-0.0004822089325731162, 0.00018926122108040507, -0.0009872515239102942),
        source="runs/rrt_action_law_calib_002:empty"),
    "grasped": NominalModel(
        gain=((0.43366858611145676, 0.01653108292354148, -0.12618701942968133), (0.003738263493937684, 0.4124284994840504, 0.041904244721423865),
              (0.039280890138053706, -0.02101383620517514, 0.3295605157634377)),
        memory=((0.05406251787154801, -0.03617028297763336, 0.10470553434543309), (-0.033921686910893825, 0.0456554622882619, -0.03189330194652605),
                (-0.08320057216845304, 0.04032714525185542, 0.1281594792988629)),
        bias=(-0.000412191138722453, 0.00017178856438144368, -0.0011416238907010633),
        source="runs/rrt_action_law_calib_002:grasped"),
}
DEFAULT_NOMINAL = "grasped"          # the pilot starts with the block grasped


def resolve_model(model=None):
    """None -> the pilot's 'grasped' model; a name -> NOMINAL_MODELS[name]; a NominalModel is returned as is."""
    if model is None:
        return NOMINAL_MODELS[DEFAULT_NOMINAL]
    if isinstance(model, str):
        if model not in NOMINAL_MODELS:
            raise ValueError(f"unknown nominal model {model!r}; one of {sorted(NOMINAL_MODELS)}")
        return NOMINAL_MODELS[model]
    if not isinstance(model, NominalModel):
        raise TypeError("model must be None, a name or a NominalModel")
    return model


class ModelPlant:
    """Plant for run_sequence driven by a NominalModel (pure numpy)."""

    def __init__(self, model, start_pinch, velocity=None):
        self.model = model
        self.position = _vec3(start_pinch, "start_pinch")
        self.velocity = np.zeros(3) if velocity is None else _vec3(velocity, "velocity")
        self.u = []

    def pinch(self):
        return self.position.copy()

    def step(self, action):
        u = unnormalize_env(action)[:3]
        self.velocity = self.model.step(self.velocity, u)
        self.position = self.position + self.velocity


def nominal_actions(segments, start_pinch, grip=1.0, config=None, model=None, prev_velocity=None, previous=None):
    """NOMINAL env actions (N, 5) of ``segments`` started at ``start_pinch``: the actions this law outputs when the
    gripper moves like the calibrated free-space model (no contact). ``model``: None (= 'grasped', the pilot), 'empty',
    'grasped' or a NominalModel. ``prev_velocity``: pinch velocity per step at the start (the robot's own last
    displacement pinch_k - pinch_{k-1}; default at rest). ``previous``: the executing runner's carry_state(), so a
    candidate edge continues the current edge exactly as the executed one would. Box clipping is included."""
    plant = ModelPlant(resolve_model(model), start_pinch, prev_velocity)
    return run_sequence(plant, segments, grip, config, previous)["actions"]


def nominal_edge_then_hold(direction, start_pinch, grip=1.0, config=None, model=None, prev_velocity=None, previous=None,
                           hold_steps=CHECK_HOLD_STEPS):
    """The planner's safety-check input: nominal actions of edge (8) + hold (12) = (20, 5)."""
    return nominal_actions(edge_then_hold(direction, hold_steps), start_pinch, grip, config, model, prev_velocity, previous)


def nominal_prev_velocity(prev_direction, config, model=None, grip=1.0):
    """Model pinch velocity at the end of an edge in ``prev_direction`` (None = at rest)."""
    if prev_direction is None:
        return np.zeros(3)
    return resolve_model(model).steady_velocity(unit(prev_direction), config, grip)


def action_mismatch(nominal, executed, normalization):
    """|nominal - executed| of dx, dy, dz per step: {'env': (N, 3), 'mm': (N, 3) physical target offset, 'wm': (N, 3)}."""
    n, e = np.asarray(nominal, float), np.asarray(executed, float)
    if n.shape != e.shape or n.ndim != 2 or n.shape[1] != 5:
        raise ValueError("nominal and executed must both be (N, 5)")
    d = np.abs(n[:, :3] - e[:, :3])
    _, scale = _normalization_arrays(normalization)
    return {"env": d, "mm": 1000 * d * ACTION_XYZ_LIMIT_M, "wm": d / scale[:3]}


# ----- directions used by the planner and the collector ------------------------------------------------------------
DIRECTION_STD = (1.0, 1.0, 0.5)      # = random_branches_plan.DIRECTION_STD
GUARD_ANGLE_DEG = 20.0               # = random_branches_plan.GUARD_ANGLE_DEG


def random_direction(rng, std=DIRECTION_STD):
    while True:
        d = rng.normal(0.0, std)
        if np.linalg.norm(d) > 1e-9:
            return unit(d)
