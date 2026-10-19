"""Pure planning for the danger benchmark: cells, recipes, intents and schedules.

numpy only, no simulator imports, so it runs on Windows and in WSL. A schedule
fixes WHICH attempt each episode makes (recipe, guard, lean direction, stop
rule, post-event behaviour); the controller in benchmark_exploration.py samples
the continuous parameters and executes it. Which stratum an episode finally
lands in is measured afterwards with wm_training.danger_strata, never assumed.

Physics that shaped the recipes (see WP1_DESIGN.md): a free 4x4x10 cm guard
tips over at about 21.8 degrees, so sustained leans (S2/S3) need a support:
a finger pressing from above, the held green block pressing from above, or the
other guard (inward falls after the green block was carried away).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEDULE_FORMAT = "danger_bench_schedule_v1"
BENCH_INTERACTION_REVISION = 3
DIRECTIONS = ("toward_camera", "away_camera", "lateral_outward", "lateral_inward")
SEVERITIES = ("S1_flat", "S2_lean", "S3_just_over")
NEGATIVES = ("N1_touch", "N2_rock", "N3_near", "N4_held_near")
HARD_CELLS = tuple(f"{d}|{s}" for d in DIRECTIONS[:3] for s in SEVERITIES) + ("lateral_inward|S3_just_over",)
BEST_EFFORT_CELLS = ("lateral_inward|S1_flat", "lateral_inward|S2_lean")
NEGATIVE_CELLS = tuple(f"negative|{n}" for n in NEGATIVES)
ALL_CELLS = HARD_CELLS + BEST_EFFORT_CELLS + NEGATIVE_CELLS
RECIPES = ("push", "press", "slot_push", "carry_hit", "carry_press", "carry_sweep", "drop", "free",
           "touch", "rock", "near", "held_near")
RECIPE_MODE = {"push": "bench_push", "press": "bench_gap", "slot_push": "bench_gap",
               "carry_hit": "bench_carry_hit", "carry_press": "bench_carry_hit", "carry_sweep": "bench_carry_hit",
               "drop": "bench_drop", "free": "bench_free", "touch": "bench_neg_touch", "rock": "bench_neg_rock",
               "near": "bench_neg_near", "held_near": "bench_neg_hold"}
BENCH_MODES = tuple(sorted(set(RECIPE_MODE.values())))
LEANS = ("+x", "-x", "outward", "inward")
CELL_LEAN = {"toward_camera": "+x", "away_camera": "-x", "lateral_outward": "outward", "lateral_inward": "inward"}
AFTER_EVENTS = ("hold", "retreat", "continue")
NEEDS_GRASP = {"carry_hit", "carry_press", "carry_sweep", "drop", "held_near"}
# Controller phase names; the recorder stores action_phase as <U16, so 16 chars max.
BENCH_PHASES = ("BENCH_APPROACH", "BENCH_LOWER", "BENCH_SEEK", "BENCH_PUSH", "BENCH_PRESS_MOVE",
                "BENCH_PRESS_DOWN", "BENCH_PRESS", "BENCH_HIT_MOVE", "BENCH_HIT_LOWER", "BENCH_HIT",
                "BENCH_DROP_MOVE", "BENCH_DROP_LOWER", "BENCH_DROP", "BENCH_PARK", "BENCH_PARK_LOWER",
                "BENCH_PARK_OPEN", "BENCH_RETRACT", "BENCH_NEAR", "BENCH_FREE", "EVENT_RETREAT")
PHASE_NAME_LIMIT = 16
TIP_ANGLE_DEG = float(np.degrees(np.arctan2(0.02, 0.05)))   # static tipping angle of a free guard

# Per-cell attempt options: (weight, recipe, lean, stop, after_event, park_green, height range).
# stop = ("stroke", low, high) in metres of post-contact travel, or ("tilt", low, high) in degrees.
_S1_AFTER = "random"
_STOP_FLAT = ("stroke", 0.06, 0.12)
_STOP_S2, _STOP_S3 = ("tilt", 42.0, 58.0), ("tilt", 31.0, 38.0)


def _x_cell_options(lean):
    return {
        # Pre-checks 1-2: drop (0/3) bounced off the guard's top edge and is not scheduled; carry_sweep is the
        # task-like mechanism but only touches in about half the attempts, so it keeps a small share.
        "S1_flat": [(0.5, "push", lean, _STOP_FLAT, _S1_AFTER, False, (0.035, 0.095)),
                    (0.3, "carry_hit", lean, ("stroke", 0.05, 0.10), _S1_AFTER, False, (0.10, 0.13)),
                    (0.15, "carry_sweep", lean, None, _S1_AFTER, False, (0.12, 0.13)),
                    (0.05, "free", None, None, _S1_AFTER, False, (0.03, 0.16))],
        # Pre-check 1 (2026-09-23): the finger press reached S2/S3 for +x once it caught the trailing edge;
        # pressing with the held green block slipped in the pads (0/10), so it keeps only a small share.
        "S2_lean": [(0.8, "press", lean, _STOP_S2, "continue", False, (0.0, 0.0)),
                    (0.2, "carry_press", lean, _STOP_S2, "continue", False, (0.0, 0.0))],
        "S3_just_over": [(0.8, "press", lean, _STOP_S3, "continue", False, (0.0, 0.0)),
                         (0.2, "carry_press", lean, _STOP_S3, "continue", False, (0.0, 0.0))],
    }


CELL_OPTIONS = {
    **{f"toward_camera|{s}": v for s, v in _x_cell_options("+x").items()},
    **{f"away_camera|{s}": v for s, v in _x_cell_options("-x").items()},
    # Pre-check 2: a 6 cm slot push with "hold" slid the guard and it rocked back (transient); push higher, farther,
    # and never stop at the first hazard. "Press to 24-29 deg then release" recovered twice, so it moved to N2.
    "lateral_outward|S1_flat": [(0.45, "slot_push", "outward", ("stroke", 0.05, 0.08), "random_cr", True, (0.07, 0.09)),
                                (0.45, "carry_hit", "outward", ("stroke", 0.04, 0.07), "random_cr", False, (0.10, 0.13)),
                                (0.1, "free", None, None, _S1_AFTER, False, (0.03, 0.16))],
    # Full collection (danger_bench_001): outward presses held up to a 52 deg target (20/21) and fell flat above
    # 53 deg (17/19); carry_press fell flat 17/20. S3 targets above 36.3 deg overshot into S2. Bands narrowed
    # for the top-up; the main collection's intents are recorded in its schedule file.
    "lateral_outward|S2_lean": [(1.0, "press", "outward", ("tilt", 42.0, 52.0), "continue", False, (0.0, 0.0))],
    "lateral_outward|S3_just_over": [(1.0, "press", "outward", ("tilt", 31.0, 35.5), "continue", False, (0.0, 0.0))],
    # Inward leans need the green block parked: the guard then falls onto the other guard (~37 deg) or is held by
    # the finger. Pre-check 1 confirmed push+park+retreat (2/2 at 37 deg); carry_hit slid the guard instead.
    "lateral_inward|S3_just_over": [(0.5, "push", "inward", ("stroke", 0.03, 0.05), "retreat", True, (0.035, 0.095)),
                                    (0.3, "press", "inward", _STOP_S3, "continue", True, (0.0, 0.0)),
                                    (0.2, "carry_hit", "inward", ("stroke", 0.03, 0.05), "retreat", False, (0.10, 0.13))],
    "lateral_inward|S2_lean": [(0.7, "press", "inward", _STOP_S2, "continue", True, (0.0, 0.0)),
                               (0.3, "carry_press", "inward", _STOP_S2, "continue", False, (0.0, 0.0))],
    "lateral_inward|S1_flat": [(0.5, "push", "inward", ("stroke", 0.08, 0.12), "continue", True, (0.035, 0.095)),
                               (0.5, "carry_hit", "inward", ("stroke", 0.08, 0.12), "continue", False, (0.10, 0.13))],
    "negative|N1_touch": [(0.3, "touch", "+x", ("stroke", 0.004, 0.012), "retreat", False, (0.07, 0.095)),
                          (0.3, "touch", "-x", ("stroke", 0.004, 0.012), "retreat", False, (0.07, 0.095)),
                          (0.4, "touch", "inward", ("stroke", 0.01, 0.03), "retreat", False, (0.035, 0.095))],
    # Pre-checks 2-3: presses released at 25-29 deg came back upright for outward leans (the finger's load
    # shifts the pivot) but a +x release fell flat, so outward gets most of the share; recovery is measured.
    # Full collection: 42/70 recovered; the 22 that fell were mostly +-x releases, so the top-up uses outward only.
    "negative|N2_rock": [(1.0, "rock", "outward", ("tilt", 20.0, 29.0), "retreat", False, (0.0, 0.0))],
    "negative|N3_near": [(1.0, "near", None, None, "hold", False, (0.03, 0.09))],
    "negative|N4_held_near": [(1.0, "held_near", None, None, "hold", False, (0.07, 0.12))],
}


# Attempts per cell for the real collection, sized from the three pre-checks (2026-09-23) so that every hard
# cell should reach about 45 measured events (15 bench-dev + 30 bench-test): attempts = 45 / estimated hit rate.
# Misses mostly land in a neighbouring severity cell, so the S1/S2 cells also collect overflow.
FULL_COUNTS_V2 = {
    "toward_camera|S1_flat": 60, "toward_camera|S2_lean": 60, "toward_camera|S3_just_over": 85,
    "away_camera|S1_flat": 60, "away_camera|S2_lean": 60, "away_camera|S3_just_over": 85,
    "lateral_outward|S1_flat": 60, "lateral_outward|S2_lean": 60, "lateral_outward|S3_just_over": 60,
    "lateral_inward|S3_just_over": 70,
    "lateral_inward|S1_flat": 40, "lateral_inward|S2_lean": 40,
    "negative|N1_touch": 50, "negative|N2_rock": 70, "negative|N3_near": 50, "negative|N4_held_near": 50,
}


def default_counts(scale):
    """precheck: 40 attempts (3 per hard cell, 1 per best-effort cell, 2 per negative);
    full: the design draft's flat 1200; full_v2: 960 attempts sized from the pre-check hit rates."""
    if scale == "precheck":
        hard, best, neg = 3, 1, 2
    elif scale == "full":
        hard, best, neg = 90, 60, 45
    elif scale == "full_v2":
        return dict(FULL_COUNTS_V2)
    else:
        raise ValueError("scale must be precheck, full or full_v2")
    return {**{c: hard for c in HARD_CELLS}, **{c: best for c in BEST_EFFORT_CELLS}, **{c: neg for c in NEGATIVE_CELLS}}


def lean_vector(lean, guard_index):
    """World direction the guard's TOP should move. Guard 1 sits at y<0, guard 2 at y>0."""
    if guard_index not in (1, 2):
        raise ValueError("guard_index must be 1 or 2")
    outer = -1.0 if guard_index == 1 else 1.0
    table = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "outward": (0.0, outer, 0.0), "inward": (0.0, -outer, 0.0)}
    if lean not in table:
        raise ValueError(f"Unknown lean {lean}")
    return np.array(table[lean])


def stop_reached(stop, travel_m, tilt_deg):
    """Pure stop rule: post-contact travel for stroke stops, measured tilt for tilt stops."""
    kind, low, high = stop
    if kind == "stroke":
        return bool(travel_m >= low - 0.0005)
    if kind == "tilt":
        return bool(tilt_deg >= low)
    raise ValueError(f"Unknown stop kind {kind}")


def validate_intent(intent):
    if not isinstance(intent, dict):
        raise ValueError("intent must be a dict")
    required = {"cell", "recipe", "guard_index", "lean", "stop", "after_event", "park_green", "height"}
    if set(intent) != required:
        raise ValueError(f"intent keys must be exactly {sorted(required)}")
    if intent["cell"] not in ALL_CELLS or intent["recipe"] not in RECIPES:
        raise ValueError("Unknown cell or recipe")
    if intent["guard_index"] not in (1, 2) or type(intent["guard_index"]) is not int:
        raise ValueError("guard_index must be the integer 1 or 2")
    if intent["lean"] is not None and intent["lean"] not in LEANS:
        raise ValueError("lean must be one of +x, -x, outward, inward or null")
    stop = intent["stop"]
    if stop is not None:
        if (not isinstance(stop, list) or len(stop) != 3 or stop[0] not in ("stroke", "tilt")
                or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in stop[1:]) or not 0 < stop[1] <= stop[2]):
            raise ValueError("stop must be null or [kind, low, high] with 0 < low <= high")
        if stop[0] == "stroke" and stop[2] > 0.15:
            raise ValueError("stroke stops above 15 cm are not allowed")
        if stop[0] == "tilt" and stop[2] >= 85:
            raise ValueError("tilt targets must stay below 85 degrees")
    if intent["after_event"] not in AFTER_EVENTS or type(intent["park_green"]) is not bool:
        raise ValueError("after_event/park_green invalid")
    height = intent["height"]
    if (not isinstance(height, list) or len(height) != 2 or not all(isinstance(v, (int, float)) for v in height)
            or not 0 <= height[0] <= height[1] <= 0.25):
        raise ValueError("height must be [low, high] inside [0, 0.25] m")
    recipe = intent["recipe"]
    if recipe in ("push", "touch", "slot_push", "carry_hit") and (stop is None or stop[0] != "stroke"):
        raise ValueError(f"{recipe} needs a stroke stop")
    if recipe in ("press", "carry_press", "rock") and (stop is None or stop[0] != "tilt"):
        raise ValueError(f"{recipe} needs a tilt stop")
    if recipe in ("push", "touch", "carry_hit", "press", "carry_press", "rock", "slot_push") and intent["lean"] is None:
        raise ValueError(f"{recipe} needs a lean direction")
    if recipe == "slot_push" and (intent["lean"] != "outward" or not intent["park_green"]):
        raise ValueError("slot_push pushes outward with the green block parked")
    if recipe in ("press", "rock") and intent["lean"] == "inward" and not intent["park_green"]:
        raise ValueError("pressing inward with the green block in place cannot tip the guard (park it first)")
    if intent["park_green"] and recipe not in ("push", "slot_push", "press"):
        raise ValueError("Only push/slot_push/press use the parked-green prologue")
    if intent["park_green"] and recipe in ("push", "press") and intent["lean"] != "inward":
        raise ValueError("Parked-green pushes/presses are for inward leans")
    return intent


def sample_intent(cell, rng):
    options = CELL_OPTIONS[cell]
    weights = np.array([o[0] for o in options], dtype=float)
    weight, recipe, lean, stop, after, park, height = options[int(rng.choice(len(options), p=weights / weights.sum()))]
    guard = int(rng.integers(1, 3))
    if lean == "random_xo":
        lean = str(rng.choice(["+x", "-x", "outward"]))
    if after == "random":
        after = str(rng.choice(AFTER_EVENTS))
    elif after == "random_cr":
        after = str(rng.choice(["continue", "retreat"]))
    intent = {"cell": cell, "recipe": recipe, "guard_index": guard, "lean": lean,
              "stop": None if stop is None else [stop[0], float(stop[1]), float(stop[2])],
              "after_event": after, "park_green": bool(park), "height": [float(height[0]), float(height[1])]}
    return validate_intent(intent)


def make_schedule(seed, counts):
    """Deterministic attempt list: cells expanded to counts, intents sampled, order shuffled."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    unknown = set(counts) - set(ALL_CELLS)
    if unknown or not counts or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError(f"counts must map known cells to non-negative integers; unknown={sorted(unknown)}")
    rng = np.random.default_rng(seed)
    entries = []
    for cell in ALL_CELLS:
        for _ in range(counts.get(cell, 0)):
            intent = sample_intent(cell, rng)
            entries.append({"mode": RECIPE_MODE[intent["recipe"]], "intent": intent})
    if not entries:
        raise ValueError("Empty schedule")
    order = rng.permutation(len(entries))
    entries = [{"id": i, **entries[j]} for i, j in enumerate(order)]
    return {"format": SCHEDULE_FORMAT, "seed": seed, "counts": {c: counts.get(c, 0) for c in ALL_CELLS},
            "interaction_revision": BENCH_INTERACTION_REVISION, "tip_angle_deg": TIP_ANGLE_DEG,
            "entries": entries}


def schedule_digest(schedule):
    body = {k: v for k, v in schedule.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_schedule(path, schedule):
    path = Path(path)
    payload = {**schedule, "sha256": schedule_digest(schedule)}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1, allow_nan=False)
        stream.write("\n")
    return payload["sha256"]


def read_schedule(path):
    schedule = json.loads(Path(path).read_text(encoding="utf-8"))
    if schedule.get("format") != SCHEDULE_FORMAT or schedule.get("sha256") != schedule_digest(schedule):
        raise ValueError("Schedule format unknown or file changed after it was written")
    entries = schedule["entries"]
    if [e["id"] for e in entries] != list(range(len(entries))):
        raise ValueError("Schedule ids must be 0..n-1 in order")
    for entry in entries:
        validate_intent(entry["intent"])
        if entry["mode"] != RECIPE_MODE[entry["intent"]["recipe"]]:
            raise ValueError("Schedule mode disagrees with recipe")
    return schedule


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", type=Path, required=True, help="NEW schedule json path")
    p.add_argument("--scale", choices=("precheck", "full", "full_v2"))
    p.add_argument("--counts", help='attempts per cell for a top-up: a JSON object or the path of a .json file holding one '
                                    '(cell names contain "|", so a file avoids shell quoting)')
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args(argv)
    if (a.scale is None) == (a.counts is None):
        p.error("Give exactly one of --scale or --counts")
    if a.scale:
        counts = default_counts(a.scale)
    else:
        text = Path(a.counts).read_text(encoding="utf-8") if a.counts.lower().endswith(".json") else a.counts
        counts = json.loads(text)
    schedule = make_schedule(a.seed, counts)
    digest = write_schedule(a.write, schedule)
    modes = {}
    for entry in schedule["entries"]:
        modes[entry["mode"]] = modes.get(entry["mode"], 0) + 1
    print(f"Wrote {len(schedule['entries'])} attempts to {a.write} (sha256 {digest[:12]}...)\nmodes: {modes}")


if __name__ == "__main__":
    main()
