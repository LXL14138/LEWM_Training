# LEWM_Training — how the S1 training data was made

This repository is the **data-generation pipeline** behind the `S1` world model released at
[LXL14138/LEWM_Manip](https://github.com/LXL14138/LEWM_Manip): the MuJoCo three-block scene, the scripted collectors,
the validators, the split tools and the cache builders that produced every training episode S1 saw. Every code file is
a byte-identical copy of the project file it came from, except four files edited for this release (every changed line
carries a `# release:` comment) and two written for it (`make_video.py`, the data viewer, and its test); `SOURCES.json`
lists the origin, the sha256 and the edit of each file. It contains
**no model weights, no training data, no planner and no model-training or evaluation code**. `bash quickstart.sh`
installs everything and re-runs the whole chain on a tiny scale (~5 minutes on two CPU cores).

## The scene

A UR5e arm with a Robotiq 2F-85 gripper (the CubeEnv of OGBench / stable-worldmodel) works on a table with three
4 × 4 × 10 cm boxes: a green block at [0.425, 0] m and two red pillars ("guards") at y = ±0.05 m, 1 cm on either side
of it. The task is to grasp the green block and carry it to a hover goal without toppling a pillar; a state is
**unsafe** when the worst pillar tilt exceeds 30° (`scene_config.UNSAFE_TILT_DEG`; a free pillar tips once past ≈ 22°).
Control runs at 20 Hz (one action per 0.05 s, physics at 500 Hz); an action is `[dx, dy, dz, yaw, dgripper]` in
[−1, 1], ±1 = ±5 cm commanded displacement of the pinch point per step, yaw fixed at 0, gripper ±1. A fixed front camera
(640 × 480, fovy 48°, shadows off) records one RGB frame per state; the cache builders LANCZOS-scale each frame to
224 × 168 and centre it on an ImageNet-mean 224 × 224 canvas ("letterbox"). Everything is simulation truth: no learned
policy, no model and no real robot is involved in making the data (`three_block_scene.py`, `scene_config.py`,
`task_progress.py`).

## The data S1 was trained on

Four sources, all produced by this code, 4,644,551 training windows in all (from the training run record shipped with
the model release).

| # | source | what it is | S1 training cache |
|---|---|---|---|
| 1 | guided random exploration | `collect_exploration_trajs.py`: scripted attempts in four modes — `free` (random moves), `grasp` (randomised grasp attempts), `carry` (grasp, carry, hover near the goal, maybe approach a pillar), `guard` (push or brush a pillar) — from random arm poses, ≤ 30 s each, failures kept. 5,000 episodes, split 4,500 / 250 / 250 by `wm_training/prepare_full.py` (the 250 test episodes were never read). | `wm_full_001`: 4,500 episodes, 1,983,477 windows (43 %) |
| 2 | targeted near-pillar attempts | `benchmark_plan.py` writes a schedule of intents (fall direction × severity, plus negatives: touch, rock, hover near); `collect_exploration_trajs.py --schedule` runs the recipes (push, press, carry-hit, slot push, rock …); `benchmark_measure.py` measures the stratum each episode actually reached; `benchmark_split.py` fixes a dev / test split per stratum; `wm_training/prepare_bench.py` builds the cache. | `wm_train_bench_001`: 961 episodes, 277,426 windows (6 %) |
| 3 | counterfactual branches | `collect_counterfactual_branches.py`: for a recorded fall event and a lead k, restore the MuJoCo state before the onset, replay the recording up to onset − k (bit-exact), then continue with a different law — `hold`, `retreat` (away from the leaning pillar) or `push_back` (against the lean) — so the same moment has several futures. 600 events × leads {4, 8, 15} × 3 branches. | `wm_cf_branches_001`: 5,400 branches, 1,123,200 windows (24 %) |
| 4 | planner-action-law branches | `rrt_action_law.py` is the planner's action vocabulary: 2 cm/s straight edges of 8 steps in a random 3-D direction, PI-tracked, then holds. `rrt_branches_plan.py` picks anchor frames from the training side in four strata (T: a pillar tilted 5–25° with the gripper or block near it; F: upright pillars, gripper low and near; G: block held near the goal; M: carrying elsewhere); `collect_rrt_branches.py` restores each anchor and runs 32-step rollouts under the law families (planner edges, hold, guard pushes …); `convert_rrt_branches.py` turns them into 35-state cache episodes and audits them. **No model runs during collection**: the data covers the commands a latent planner will later issue, before any planner exists. 5,340 anchors / 44,729 rollouts. | `wm_rrt_branches_001`: 39,389 cache episodes, 1,260,448 windows (27 %) |

`collect_random_branches.py` / `random_branches_plan.py` (the earlier random-action branch collector) are shipped only as
the library that `rrt_branches_plan.py` and `collect_rrt_branches.py` import (restore, capture and writing code); their own
datasets were evaluation pools of the earlier model B, not training data.

Validation data of the S1 run: 250 val episodes of source 1, 480 bench_dev episodes of source 2, 2,220 dev branches of
source 4, 540 of source 3. The hold-out danger benchmark used only for evaluation was made with the source-2 tools from a
different schedule and is kept apart by name (`danger_bench*` vs `danger_train*`); several tools refuse the wrong family.

Two earlier training artefacts are read **as records only** by the law-branch planner (source 4): the run record of
the safety classifier (which validation episodes are its dev / report halves) and the fine-tune record of model B (which
cache rows were its training material). They fix which episodes count as training-side; no weights are loaded. The
smoke chain writes stand-in records with just the keys the planners read.

## Pipeline

```text
collect_exploration_trajs.py ──► raw episodes (manifest.json, episode_XXXX/{images/*.png, trajectory.npz, metadata.json})
        │  (--schedule from benchmark_plan.py for source 2)                     │
        ▼                                                                       ▼
inspect_grasp_dataset.py (validate / replay)                 wm_training/prepare_full.py ──► wm_full_* (training cache, 90/5/5)
        │                                                                       │ reference (contract + normalisation)
        ▼                                                                       ▼
tools/bench_precheck_report.py ─► benchmark_split.py ─► wm_training/prepare_bench.py ──► wm_train_bench_* / wm_bench_*
                                        │
                                        ├─► collect_counterfactual_branches.py ─► counterfactual_split.py ─► prepare_bench ─► wm_cf_*
                                        └─► rrt_branches_plan.py ─► collect_rrt_branches.py ─► convert_rrt_branches.py ─► wm_rrt_branches_*
bench_restore_check.py: proves state restore ↔ recording (used by every branch collector)
calibrate_rrt_action_law.py: nominal plant models of the action law
make_video.py: GIFs + contact sheets of any raw episode, counterfactual grid, anchor or cache episode (no display needed)
```

Shared code: `exploration.py` (scene + controller), `collect_grasp_trajs.py` (episode recorder), `auto_grasp.py`
(scripted grasp), `random_starts.py`, `collection_workers.py`, `counterfactual_plan.py`, `benchmark_measure.py`,
`random_branches_plan.py` / `collect_random_branches.py` (see above), `rrt_pilot_yields.py` (the decision-point rules
`collect_rrt_branches.py` records), `wm_training/{prepare_data.py, danger_strata.py, label_audit.py, provenance.py}`.

## Directory layout

```text
collect_*.py  benchmark_*.py  *_plan.py  *_split.py  convert_rrt_branches.py  rrt_*.py  bench_restore_check.py
inspect_grasp_dataset.py  exploration_validation.py  calibrate_rrt_action_law.py  make_video.py (see the data as GIFs)
three_block_scene.py  scene_config.py  task_progress.py  exploration.py  auto_grasp.py  collect_grasp_trajs.py
random_starts.py  collection_workers.py
wm_training/   prepare_data.py, prepare_full.py, prepare_bench.py, dataset.py (cache window reader, used by the prepare
               tests), danger_strata.py, provenance.py, label_audit.py (the per-frame label audit convert_rrt_branches.py
               runs: a verbatim extract of the project's safety_classifier.py, whose classifier is not shipped)
tools/         bench_precheck_report.py, bench_cell_gallery.py
tests/, wm_training/tests/   unit tests
quickstart.sh  requirements.txt  SOURCES.json  SHA256SUMS  LICENSE
```

The scripts import each other by bare module name, so every command runs **from the repository root**. Most tools write
a human-readable `REPORT_ZH.md` (Chinese) next to their `summary.json`; every number in it is also in the JSON.

## Data formats

**Raw episode dataset** (sources 1–3): `<dataset>/manifest.json` (status, seed, software versions, source-code
sha256, one row per episode) and per episode `episode_XXXX/images/000000.png …` (640 × 480 RGB, one per state),
`trajectory.npz` (T actions, T+1 states; `state[i] -- actions[i] --> state[i+1]`) and `metadata.json` (start pose,
controller record, outcome). Main `trajectory.npz` keys: `actions (T,5)`, `physical_actions`, `time / qpos / qvel /
physics_state` (what the branch collectors restore), `object_com_xyz (T+1,3,3)`, `pinch_xyz`, `guard_tilt_deg (T+1,2)`,
contact forces and flags, `unsafe`, `grasped`, `hover_goal_xyz`, `class_label (T+1)` (0 = safe and hovering at the goal,
1 = a pillar beyond 30°, 2 = safe, not hovering). `metadata.status = complete` means the files are complete, not that
the task succeeded.

**Branch dataset** (source 4): `manifest.json` + `anchor_XXXX.npz` with `pixels (R, S, 224, 224, 3)` letterboxed
frames (R rollouts, S states, state 0 = the anchor), `actions (R, S−1, 5)`, `wm_action` (the commands in the world
model's normalisation), `prefix_pixels (11, 224, 224, 3)` (cache frames a−10..a), per-state `guard_tilt_deg / unsafe /
class_label / grasped / physics_state / contacts`, the law's bookkeeping and the anchor's provenance.

**HDF5 training cache** (`prepare_full`, `prepare_bench`, `convert_rrt_branches`): `index.json` (`format
three_blocks_lewm_v1`, the split with one row per episode and its `path`, `normalization` = mean / std of the action
columns fitted on the training part) and one file per episode with `pixels (T+1, 224, 224, 3)` uint8, `action (T, 4)`
float32 (dx, dy, dz, dgripper; yaw dropped), `diagnostics/*` (every numeric array of the raw episode) and the metadata
as attributes. A world model reads windows of 4 frames and 3 actions, never across episodes.

```python
import h5py, json
root = "smoke_out/data/wm_full_smoke_001/"
idx = json.load(open(root + "index.json"))
row = idx["episodes"]["train"][0]
f = h5py.File(root + row["path"], "r")
f["pixels"].shape, f["action"].shape, f["diagnostics/guard_tilt_deg"][:], f.attrs["alignment"]
```

## Quick start

```bash
git clone https://github.com/LXL14138/LEWM_Training.git && cd LEWM_Training   # Windows: inside WSL (Ubuntu)
bash quickstart.sh                                                             # [--with-tests] [--cuda] [--skip-install] [--help]
```

`quickstart.sh` finds a Python 3.10 (an existing `./.venv`, `$PYTHON310`, `python3.10` on PATH, or one provisioned
through `uv`), creates `./.venv`, installs the torch CPU wheels and `requirements.txt` (~2 GB), verifies `SHA256SUMS`,
checks the free disk space (≥ 22 GiB: the counterfactual collector refuses to start below its 20 GiB reserve), probes a
MuJoCo rendering backend (`egl`, `osmesa`, `glfw`, or your exported `MUJOCO_GL`) and runs the chain with tiny counts
into a new `./smoke_out/`: 8 exploration episodes → validation → training cache → 6 targeted attempts → measured
strata and split → bench cache → a hold-out mini benchmark → restore check → counterfactual branches → planner-law
branches (plan, collect) → conversion check (the converter's post-conversion audit needs the training modules, which are
not shipped; it is skipped and recorded as such in the conversion's `summary.json`) → videos of the data just made
(`make_video.py`, next section: GIFs in `smoke_out/videos/`, no display needed). Every step prints `OK` or a `FAIL`
line that says what is wrong and what to do; re-running reuses the venv. Measured on WSL2 (2 cores, NVIDIA EGL): about
5 minutes in all (312 s with `--with-tests`), 330 MB of outputs. `--with-tests` adds the 13 unit-test suites that need
no simulator (~20 s). Linux / WSL only.

## See the data you generated

Nothing here needs a display. `make_video.py` (numpy + Pillow + h5py only) turns the data into GIFs with a label bar
above every frame — the state index and time, the worst pillar tilt in degrees (red with `UNSAFE` when the recorded flag
is set), grasped yes / no, and for a branch its name and lead — plus a contact-sheet PNG of 8 evenly spaced states.
`quickstart.sh` renders three examples into `smoke_out/videos/` (WSL: `explorer.exe smoke_out/videos`; Linux desktop:
`xdg-open smoke_out/videos/<name>.gif`; headless: copy the files to any machine with a browser).

```bash
.venv/bin/python make_video.py smoke_out/data/exploration_smoke_001/episode_0000 --out videos       # one raw episode
.venv/bin/python make_video.py smoke_out/data/cf_branches_smoke_001 --grid --all --out videos        # branches side by side
.venv/bin/python make_video.py smoke_out/data/rrt_branches_smoke_001/anchor_0000.npz --out videos  # one anchor, all rollouts
.venv/bin/python make_video.py smoke_out/data/wm_full_smoke_001/episodes/episode_0000.h5 --out videos  # a cache episode
```

- An **episode directory** (sources 1–3: `images/` + `trajectory.npz`) gives `<dataset>__<episode>.gif` — every 2nd state
  at 10 fps, i.e. real time for a 20 Hz episode (`--every`, `--fps`; `--scale 0.5` = 320 × 240 frames by default) — and
  `<dataset>__<episode>_strip.png`.
- A **dataset directory** (`manifest.json`) renders its first 3 episodes (`--limit N` or `--all`). For a counterfactual
  dataset, `--grid` groups the branches that share a source event and a lead and plays them side by side, frames aligned
  by state index — the same moment, several futures (`hold | retreat | push_back`), each panel with its own tilt overlay.
- An **anchor npz** (source 4) renders all rollouts of the anchor as a grid of 224 px panels (8 rollouts → 4 × 2), each
  labelled with its action-law family; `--rollout R` renders one rollout alone.
- A **cache episode `.h5`** is rendered from `pixels` with `diagnostics/guard_tilt_deg` and `diagnostics/unsafe` when present.

The tool prints every output path with its size and never overwrites a file without `--force`. With a display,
`inspect_grasp_dataset.py <dataset> --replay <index>` is the interactive alternative (a pygame window).

## Scaling up

The real campaigns use the same commands with larger counts and `--workers 1–4` (one simulator process
each): e.g. 5,000 exploration episodes, the full targeted schedule (`benchmark_plan.py --scale full`), the counterfactual
campaign `--events 600 --leads 4,8,15 --branches hold,retreat,push_back --hold-steps 200`, and the law-branch campaign
with its anchor quotas per stratum (`rrt_branches_plan.py --quota`; the project fixed them in a frozen plan from pilot
yields, and neither that plan nor the quota planner is shipped). The collectors refuse to overwrite an existing output
directory and record the code hashes and the rendering backend in every manifest.

## Tests

`python -m unittest <module>` from the repository root with `.venv/bin/python`. `quickstart.sh --with-tests` runs the
13 suites that need no simulator (122 tests, 3 of them skip without the project's real data; `tests.test_make_video`
renders synthetic frames). The 17 simulator suites
(`tests.test_scene`, `test_exploration`, `test_rrt_branches`, …) take ~5 minutes with `MUJOCO_GL` exported;
`tests.test_tilt_safety` has two historically failing subtests that are shipped as they are. Tests that need the
project's real data skip by themselves.

## Limitations

- **Rendering backend.** The campaigns were rendered with `MUJOCO_GL=glfw` (WSL2 + NVIDIA, d3d12); the script defaults
  to `egl`. Frames from different backends differ at the few-levels pixel scale (physics identical), so data made here
  is not bit-identical to the original data; every manifest records the backend.
- **Determinism** holds within one build on one CPU (the collectors verify it); another machine produces different
  contact events for the same seeds.
- **Not shipped** (deliberately): the S1 model and its training / evaluation code (the project's `wm_training` training
  and evaluation modules; the post-conversion audit of `convert_rrt_branches.py`, which imported them, is skipped and
  recorded as such), the campaign quota planner and the frozen campaign plan (`rrt_branches_plan.py` hard-codes its
  sha256, so the `dev` / `confirm` roles and the production conversion names cannot be exercised here), the
  re-collection comparison tool, the goal images and the planner.
- **Tiny-data yield.** With 8 episodes and 6 targeted attempts the chain needs at least one fall event; the `FAIL`
  messages say which count to raise. The campaigns' counts make this certain.
- **Disk reserves** are built into the collectors (20 GiB for the counterfactual collector, 5 GiB for the others) and
  cannot be switched off.

## License

MIT (`LICENSE`) for the code in this repository. The scene is a subclass of the CubeEnv of stable-worldmodel
(https://github.com/galilai-group/stable-worldmodel, installed from the pinned git commit) and uses the OGBench manipspace
robot and physics (https://github.com/seohongpark/ogbench), MuJoCo and dm_control; these are installed from their own
distributions by `requirements.txt`, each under its own open-source licence, and are not redistributed here. If you use
this pipeline, please cite the model release LXL14138/LEWM_Manip, which holds the S1 weights, its training run record
(the source of the counts above) and the parity-checked loader.
