#!/usr/bin/env bash
# quickstart.sh - one command that sets up this repository and runs the data-generation pipeline end to end on a
# tiny scale, from the repository root:
#     bash quickstart.sh [--with-tests] [--cuda] [--skip-install] [--help]
# It finds a Python 3.10, creates ./.venv, installs torch (CPU wheel unless --cuda) + requirements.txt, verifies
# SHA256SUMS, checks the free disk space, probes a MuJoCo rendering backend, then runs the smoke chain into a NEW
# directory ./smoke_out (smoke_out_2, ... on re-runs): collect 8 exploration episodes -> validate -> training cache;
# a 6-attempt near-pillar collection -> measured cells -> split -> bench cache; a 3-attempt hold-out collection -> its
# cache; the state-restore check; counterfactual branches -> split -> cache; planner-action-law branches (plan +
# collect + cache conversion check); videos of the data (make_video.py: GIFs with a tilt overlay, no display needed,
# in smoke_out/videos). Every step prints a heading, then OK, or FAIL with what is wrong and what to do. Re-running
# reuses the venv (installs are skipped when already satisfied) and runs the chain again.
# --with-tests also runs the unit tests that need no simulator (122 tests, ~20 s more). Measured on WSL2 (Ubuntu
# 24.04, 2 CPU cores, NVIDIA EGL): 312 s in all with --with-tests and a warm pip cache (the first run also
# downloads ~2 GB of packages), of which the chain itself (steps 7-21, videos included) is ~4.5 min.
# Linux only (prepare_full uses fcntl, the branch collectors resource and /proc);
# on Windows run it inside WSL from a clone made inside WSL. Needs: bash 4+, git, network, sha256sum; timeout (coreutils).
IFS= read -r _l < "$0"; case "$_l" in *$'\r'*) echo "FAIL quickstart.sh has CRLF line endings (checked out with Git for Windows / core.autocrlf=true?)"; echo "  -> clone inside WSL or Linux, or run: git config core.autocrlf false && git checkout -- .   then re-run"; exit 1;; esac #
set -u

TORCH_PINS="torch==2.14.0 torchvision==0.29.0"
WITH_TESTS=0; CUDA=0; SKIP_INSTALL=0; SUMMARY=(); T0=$SECONDS; GL=""; NO_TIMEOUT=0
for arg in "$@"; do
    case "$arg" in
        --with-tests) WITH_TESTS=1 ;;  --cuda) CUDA=1 ;;  --skip-install) SKIP_INSTALL=1 ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "FAIL unknown option '$arg' (see: bash quickstart.sh --help)"; exit 2 ;;
    esac
done

step() { STEP_NAME="$2"; STEP_T=$SECONDS; echo; echo "== [$1] $2${3:+: $3}"; }
ok()   { echo "OK ${1:-}"; SUMMARY+=("$(printf '  %-34s OK  %4d s  %s' "$STEP_NAME" $((SECONDS - STEP_T)) "${1:-}")"); }
fail() { echo "FAIL $1"; echo "  -> $2"; [ -z "${OUT:-}" ] || echo "  (logs: $PWD/$OUT/logs)"; exit 1; }
run_logged() { local log=$1; shift; "$@" 2>&1 | tee "$log"; return "${PIPESTATUS[0]}"; }   # show + keep a copy
if ! command -v timeout >/dev/null 2>&1; then
    if command -v gtimeout >/dev/null 2>&1; then timeout() { gtimeout "$@"; }
    else timeout() { shift; "$@"; }; NO_TIMEOUT=1; fi
fi
TMPLOG=$(mktemp -t quickstart.XXXXXX) || { echo "FAIL cannot create a temp file"; exit 1; }
trap 'rm -f "$TMPLOG"' EXIT
# jcheck FILE EXPR WHAT: evaluate a python expression on the JSON file (as d); the step fails unless it is True
jcheck() {
    local out
    out=$("$VPY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2" 2>&1) ||
        { echo "   cannot evaluate '$3' on $1: $(echo "$out" | tail -1)"; return 1; }
    [ "$out" = "True" ] || { echo "   check failed: $3 (got $out) in $1"; return 1; }
}

# ---- 0. location -----------------------------------------------------------------------------------------------
step 0 "location check"
[ -f collect_exploration_trajs.py ] && [ -f requirements.txt ] && [ -f wm_training/prepare_full.py ] ||
    fail "collect_exploration_trajs.py / requirements.txt / wm_training/prepare_full.py are not in the current directory ($PWD)" \
         "cd into the repository root (the directory that contains collect_exploration_trajs.py), then: bash quickstart.sh"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) fail "native Windows shell detected" "run this script inside WSL (Ubuntu)";;
    Linux) ;;
    *) echo "   note: only Linux / WSL is supported (prepare_full uses fcntl, the branch collectors resource and /proc); continuing anyway";;
esac
command -v git >/dev/null 2>&1 || fail "git is not installed (pip needs it for stable-worldmodel)" "Debian/Ubuntu: sudo apt install git; then re-run"
[ $NO_TIMEOUT = 0 ] || echo "   note: no 'timeout' command, so a step that hangs cannot be detected"
ok "$PWD"

# ---- 1. Python 3.10 --------------------------------------------------------------------------------------------
step 1 "Python 3.10"
pyver() { "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null; }
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python; HOW="existing .venv"
elif [ -n "${PYTHON310:-}" ]; then
    PY=$PYTHON310; HOW="\$PYTHON310"
elif command -v python3.10 >/dev/null 2>&1; then
    PY=$(command -v python3.10); HOW="python3.10 on PATH"
else
    UV=$(command -v uv 2>/dev/null || true); [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV=$HOME/.local/bin/uv
    [ -n "$UV" ] || fail "no python3.10 on PATH and no uv (Python 3.12 etc. are untested, 3.10 is required)" \
        "either  curl -LsSf https://astral.sh/uv/install.sh | sh  and re-run (the script then gets 3.10 through uv), or  pyenv install 3.10  and  export PYTHON310=/path/to/python3.10"
    PY=$("$UV" python find 3.10 2>/dev/null) || { echo "   uv: installing CPython 3.10 ..."; "$UV" python install 3.10 ||
        fail "uv python install 3.10 failed" "check the network, or export PYTHON310=/path/to/python3.10 and re-run"; PY=$("$UV" python find 3.10) ||
        fail "uv cannot find the 3.10 it just installed" "run  uv python find 3.10  and export PYTHON310=<that path>"; }
    HOW="uv"
fi
VER=$(pyver "$PY") || fail "$PY does not run ($HOW)" "export PYTHON310=/path/to/a/working/python3.10 (or rm -rf .venv) and re-run"
case "$VER" in 3.10.*) ;; *) fail "$PY is Python $VER, not 3.10 ($HOW)" \
    "export PYTHON310=/path/to/python3.10 (or rm -rf .venv if it was made with another version) and re-run";; esac
ok "$PY  (CPython $VER, $HOW)"

# ---- 2. virtual environment ------------------------------------------------------------------------------------
step 2 "virtual environment ./.venv"
[ -x .venv/bin/python ] || "$PY" -m venv .venv ||
    fail "$PY -m venv .venv failed" "Debian/Ubuntu: sudo apt install python3.10-venv; then rm -rf .venv and re-run"
VPY=.venv/bin/python; PIP=(.venv/bin/python -m pip)
"${PIP[@]}" --version >/dev/null 2>&1 || "$VPY" -m ensurepip --upgrade >/dev/null 2>&1 ||
    fail ".venv has no pip" "rm -rf .venv and re-run (Debian/Ubuntu: sudo apt install python3.10-venv first)"
ok "$VPY (CPython $(pyver "$VPY")); the script calls .venv/bin/python directly, no activation needed"

# ---- 3. packages -----------------------------------------------------------------------------------------------
step 3 "packages" "torch $([ $CUDA = 1 ] && echo CUDA || echo CPU) + requirements.txt"
pins_satisfied() {  # exit 0 when every  name==version  /  name @ url  line of the given files is installed
    "$VPY" - "$@" <<'PYEOF'
import importlib.metadata as m, re, sys
bad = []
for path in sys.argv[1:]:
    for line in open(path):
        line = line.split("#")[0].split(";")[0].strip()
        if not line or line.startswith("-"): continue
        name, want = (line.split("@")[0], None) if "@" in line else (line.split("==")[0], line.split("==")[1].strip())
        name = re.sub(r"\[.*\]", "", name).strip()
        try: have = m.version(name)
        except m.PackageNotFoundError: bad.append(name + " missing"); continue
        if want and have.split("+")[0] != want: bad.append(f"{name} {have}!={want}")
print("   not yet satisfied: " + ", ".join(bad) if bad else "   all pins already installed"); sys.exit(1 if bad else 0)
PYEOF
}
if [ $SKIP_INSTALL = 1 ]; then
    echo "   --skip-install: pip not run"
else
    "${PIP[@]}" install --quiet --upgrade pip || fail "pip self-upgrade failed" "check the network / proxy and re-run (or --skip-install if everything is installed)"
    if "$VPY" -c 'import torch, torchvision' >/dev/null 2>&1 && pins_satisfied <(printf '%s\n' $TORCH_PINS) >/dev/null; then
        echo "   torch $("$VPY" -c 'import torch; print(torch.__version__)') already installed, keeping it"
    elif [ $CUDA = 1 ]; then
        "${PIP[@]}" install --quiet $TORCH_PINS || fail "pip install $TORCH_PINS (CUDA wheels from PyPI) failed" "check the network and re-run (or without --cuda)"
    else
        "${PIP[@]}" install --quiet $TORCH_PINS --index-url https://download.pytorch.org/whl/cpu ||
            fail "pip install of the torch CPU wheels failed" "check that download.pytorch.org is reachable and re-run"
    fi
    pins_satisfied requirements.txt ||
        { echo "   pip install -r requirements.txt ..."; "${PIP[@]}" install --quiet -r requirements.txt ||
          fail "pip install -r requirements.txt failed" "read pip's message above (network? git for stable-worldmodel? disk space?), fix it and re-run"; }
    "${PIP[@]}" check || fail "pip check found broken requirements" "rm -rf .venv and re-run (the pins are mutually consistent in a clean venv)"
fi
export PYGAME_HIDE_SUPPORT_PROMPT=1 PYTHONDONTWRITEBYTECODE=1
MUJOCO_GL=disable "$VPY" -c 'import numpy, PIL, h5py, torch, pygame, mujoco, ogbench, stable_worldmodel, dm_control, gymnasium' 2>"$TMPLOG" ||
    fail "the merged stack does not import: $(tail -1 "$TMPLOG")" "re-run without --skip-install (or rm -rf .venv and re-run)"
"$VPY" -c 'import collect_exploration_trajs, collect_counterfactual_branches, collect_random_branches, collect_rrt_branches, convert_rrt_branches, rrt_branches_plan, random_branches_plan, wm_training.prepare_full, wm_training.prepare_bench' 2>"$TMPLOG" ||
    fail "a repository module does not import: $(tail -1 "$TMPLOG")" "run from the repository root; re-run without --skip-install"
ok "torch $("$VPY" -c 'import torch; print(torch.__version__)'), h5py $("$VPY" -c 'import h5py; print(h5py.__version__)'), mujoco $("$VPY" -c 'import mujoco; print(mujoco.__version__)'); $("${PIP[@]}" list 2>/dev/null | tail -n +3 | wc -l | tr -d ' ') packages"

# ---- 4. checksums ----------------------------------------------------------------------------------------------
step 4 "checksums" "SHA256SUMS"
if [ ! -f SHA256SUMS ]; then
    echo "   SHA256SUMS is not present (a working copy before release packaging?); nothing verified"
    ok "skipped (no SHA256SUMS file)"
else
    if command -v sha256sum >/dev/null 2>&1; then SUMCMD=(sha256sum -c SHA256SUMS)
    elif command -v shasum >/dev/null 2>&1; then SUMCMD=(shasum -a 256 -c SHA256SUMS)
    else fail "neither sha256sum nor shasum is available" "install coreutils (Linux) or perl (shasum) and re-run"; fi
    "${SUMCMD[@]}" >"$TMPLOG" 2>&1 || { grep -v ': OK$' "$TMPLOG" | sed 's/^/   /';
        fail "some files are modified or missing (FAILED lines above)" "restore them from the repository (git checkout -- <file>, or re-clone) and re-run"; }
    ok "$(grep -c ': OK$' "$TMPLOG") files verified with ${SUMCMD[0]}"
fi

# ---- 5. disk space ---------------------------------------------------------------------------------------------
step 5 "free disk space" "the counterfactual collector refuses to start below 20 GiB free, the branch collectors below 5 GiB"
FREE_GIB=$(df -Pk . | awk 'NR==2 {printf "%d", $4 / 1048576}')
[ "$FREE_GIB" -ge 22 ] || fail "only $FREE_GIB GiB free on the file system of $PWD (the smoke outputs need ~1 GB, but collect_counterfactual_branches.py checks for 20 GiB + its estimate)" \
    "free some space or clone the repository on a bigger disk and re-run"
ok "$FREE_GIB GiB free (smoke outputs ~1 GB)"

# ---- 6. rendering backend --------------------------------------------------------------------------------------
GLDESC="probing egl, osmesa, glfw"; [ -z "${MUJOCO_GL:-}" ] || GLDESC="exported MUJOCO_GL=$MUJOCO_GL"
step 6 "MuJoCo rendering backend" "$GLDESC"
GLPROBE='import mujoco; m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"1 1 1\"/></worldbody></mujoco>"); r = mujoco.Renderer(m, 32, 32); r.update_scene(mujoco.MjData(m)); r.render(); print("ok")'
for cand in ${MUJOCO_GL:-egl osmesa glfw}; do
    if MUJOCO_GL=$cand timeout 60 "$VPY" -c "$GLPROBE" >"$TMPLOG" 2>&1; then GL=$cand; break; fi
    echo "   $cand: no ($(grep -Ev '^ |^Traceback' "$TMPLOG" | tail -1 | cut -c1-150))"
done
[ -n "$GL" ] || fail "no working MuJoCo rendering backend${MUJOCO_GL:+ (your exported MUJOCO_GL=$MUJOCO_GL fails; unset it to let the script probe)}" \
    "egl needs a GPU driver that exposes EGL (Debian/Ubuntu: sudo apt install libegl1 + NVIDIA/Mesa driver); osmesa needs sudo apt install libosmesa6; glfw needs a display (DISPLAY / WSLg)"
export MUJOCO_GL=$GL     # fixed for the WHOLE chain: the branch collectors compare re-rendered frames with the cache pixels bit for bit
ok "MUJOCO_GL=$GL (kept for every step below)"

# ================================================================================================================
# The smoke chain. Everything goes under $OUT (new directory); every script refuses to overwrite existing output.
# ================================================================================================================
OUT=smoke_out; n=2; while [ -e "$OUT" ]; do OUT=smoke_out_$n; n=$((n + 1)); done
mkdir -p "$OUT/logs" "$OUT/data" "$OUT/runs" "$OUT/reports" "$OUT/schedules" "$OUT/stubs" "$OUT/checks"
echo; echo "== smoke chain -> $PWD/$OUT   (MUJOCO_GL=$GL, one simulator process at a time)"
D=$OUT/data; R=$OUT/runs

# ---- 7. exploration episodes (data source 1) --------------------------------------------------------------------
step 7 "collect exploration episodes" "collect_exploration_trajs.py, 8 episodes x 15 s (2 per mode)"
run_logged "$OUT/logs/07_exploration.log" timeout 1800 "$VPY" collect_exploration_trajs.py --output-dir "$D/exploration_smoke_001" \
    --n-trajectories 8 --max-seconds 15 --workers 1 --seed 20260920; RC=$?
[ $RC = 0 ] && jcheck "$D/exploration_smoke_001/manifest.json" 'd["status"] == "complete" and d["interaction_revision"] == 2 and sum(e.get("status") == "complete" for e in d["episodes"]) == 8' "8 complete episodes, interaction_revision 2" ||
    fail "collect_exploration_trajs.py did not produce 8 complete episodes (exit $RC)" "read the traceback above / $OUT/logs/07_exploration.log; a rendering error -> export MUJOCO_GL=<egl|osmesa|glfw> and re-run"
ok "$("$VPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["episodes"]), "complete episodes (the real campaign: exploration_5000_001, 5,000 episodes)")' "$D/exploration_smoke_001/manifest.json")"

# ---- 8. validate ---------------------------------------------------------------------------------------------------
step 8 "validate the exploration episodes" "inspect_grasp_dataset.py --report"
run_logged "$OUT/logs/08_inspect_exploration.log" timeout 600 "$VPY" inspect_grasp_dataset.py "$D/exploration_smoke_001" --report "$OUT/reports/exploration_smoke_validation.json"; RC=$?
[ $RC = 0 ] && jcheck "$OUT/reports/exploration_smoke_validation.json" 'd["valid"] is True' "report valid" ||
    fail "the validator rejected the exploration episodes (exit $RC)" "read the errors in $OUT/reports/exploration_smoke_validation.json"
ok "valid: true"

# ---- 9. training cache (the wm_full_001 recipe) ------------------------------------------------------------------
step 9 "training cache" "wm_training.prepare_full --full (fixed 90/5/5 split, TRAIN-only action normalisation)"
run_logged "$OUT/logs/09_prepare_full.log" timeout 1800 "$VPY" -m wm_training.prepare_full --source "$D/exploration_smoke_001" \
    --output "$D/wm_full_smoke_001" --full --workers 1 --seed 20260921; RC=$?
[ $RC = 0 ] && jcheck "$D/wm_full_smoke_001/index.json" 'd["status"] == "complete" and len(d["episodes"]["train"]) >= 2 and len(d["episodes"]["val"]) >= 1 and len(d["split_ids"]["test"]) >= 1' "complete index with train >= 2, val >= 1, test >= 1" ||
    fail "prepare_full did not write a complete cache (exit $RC)" "read $OUT/logs/09_prepare_full.log (disk budget? h5py?)"
ok "$("$VPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("train", len(d["episodes"]["train"]), "/ val", len(d["episodes"]["val"]), "/ test", len(d["split_ids"]["test"]), "(test rows never trained on)")' "$D/wm_full_smoke_001/index.json")"

# ---- 10. near-pillar schedule + collection (data source 2) -------------------------------------------------------
step 10 "near-pillar training collection" "benchmark_plan.py schedule (6 attempts) -> collect_exploration_trajs.py --schedule"
printf '%s\n' '{"toward_camera|S1_flat": 2, "away_camera|S1_flat": 2, "lateral_outward|S1_flat": 1, "negative|N3_near": 1}' > "$OUT/schedules/train_counts.json"
run_logged "$OUT/logs/10a_benchmark_plan.log" timeout 120 "$VPY" benchmark_plan.py --write "$OUT/schedules/danger_train_smoke.json" --counts "$OUT/schedules/train_counts.json" --seed 1 ||
    fail "benchmark_plan.py failed" "read the traceback above"
run_logged "$OUT/logs/10b_collect_danger_train.log" timeout 1800 "$VPY" collect_exploration_trajs.py --schedule "$OUT/schedules/danger_train_smoke.json" \
    --output-dir "$D/danger_train_smoke_001" --max-seconds 20 --workers 1 --seed 1; RC=$?
[ $RC = 0 ] && jcheck "$D/danger_train_smoke_001/manifest.json" 'd["status"] == "complete" and d["interaction_revision"] == 3 and sum(e.get("status") == "complete" for e in d["episodes"]) == 6' "6 complete attempts, interaction_revision 3" ||
    fail "the scheduled collection did not complete (exit $RC)" "read $OUT/logs/10b_collect_danger_train.log"
run_logged "$OUT/logs/10c_inspect_danger_train.log" timeout 600 "$VPY" inspect_grasp_dataset.py "$D/danger_train_smoke_001" --report "$OUT/reports/danger_train_smoke_validation.json" ||
    fail "the validator rejected the near-pillar episodes" "read $OUT/reports/danger_train_smoke_validation.json"
ok "6 attempts, validated (the real campaign: danger_train_001/002, 961 bench_test + 480 bench_dev episodes)"

# ---- 11. measured cells ------------------------------------------------------------------------------------------
step 11 "measured strata" "tools/bench_precheck_report.py (intended vs measured direction|severity)"
run_logged "$OUT/logs/11_precheck.log" timeout 600 "$VPY" tools/bench_precheck_report.py "$D/danger_train_smoke_001" --output "$OUT/reports/danger_train_smoke_cells"; RC=$?
[ $RC = 0 ] && jcheck "$OUT/reports/danger_train_smoke_cells/precheck_report.json" 'any("|" in r["measured_cell"] for r in d["rows"])' "at least one measured fall event (direction|severity)" ||
    fail "no attempt produced a measured fall event (exit $RC); the later branch steps need one" \
         "the push recipes fall in ~90 % of attempts, so this is bad luck or a physics difference: raise the S1_flat counts in step 10 (train_counts.json), or --max-seconds, and re-run"
ok "$("$VPY" -c 'import json,sys,collections; d=json.load(open(sys.argv[1])); print(dict(collections.Counter(r["measured_cell"] for r in d["rows"])))' "$OUT/reports/danger_train_smoke_cells/precheck_report.json")"

# ---- 12. split + bench cache -------------------------------------------------------------------------------------
step 12 "training-bench split + cache" "benchmark_split.py -> wm_training.prepare_bench (the wm_train_bench_001 recipe)"
run_logged "$OUT/logs/12a_benchmark_split.log" timeout 600 "$VPY" benchmark_split.py --dataset "$D/danger_train_smoke_001" --seed 0 --dev-fraction 0.34 --output "$D/danger_train_split_smoke_001"; RC=$?
[ $RC = 0 ] && jcheck "$D/danger_train_split_smoke_001/benchmark_split.json" 'any("|" in k and not k.startswith("negative") and len(v["test"]) >= 1 for k, v in d["strata"].items())' "a fall-event stratum with a bench_test member" ||
    fail "benchmark_split.py gave no bench_test fall event (exit $RC)" "see step 11; re-run with more S1_flat attempts"
run_logged "$OUT/logs/12b_prepare_bench_train.log" timeout 1800 "$VPY" -m wm_training.prepare_bench --split "$D/danger_train_split_smoke_001/benchmark_split.json" \
    --reference-cache "$D/wm_full_smoke_001" --output "$D/wm_train_bench_smoke_001" --workers 1; RC=$?
[ $RC = 0 ] && jcheck "$D/wm_train_bench_smoke_001/index.json" 'd["status"] == "complete" and d["normalization"]["copied_from_reference"] is True and len(d["episodes"]["val"]) == 6' "complete eval cache, normalisation copied, 6 rows" ||
    fail "prepare_bench did not write a complete cache (exit $RC)" "read $OUT/logs/12b_prepare_bench_train.log"
ok "$("$VPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("bench_dev", len(d["bench_split"]["bench_dev"]), "/ bench_test", len(d["bench_split"]["bench_test"]), "rows")' "$D/wm_train_bench_smoke_001/index.json")"

# ---- 13. hold-out benchmark (evaluation data; here only so the planners have a hold-out cache) --------------------
step 13 "hold-out benchmark" "3 attempts -> split -> cache (the danger_bench_001/002 -> wm_bench_001 recipe)"
printf '%s\n' '{"toward_camera|S1_flat": 2, "negative|N3_near": 1}' > "$OUT/schedules/bench_counts.json"
run_logged "$OUT/logs/13a_benchmark_plan.log" timeout 120 "$VPY" benchmark_plan.py --write "$OUT/schedules/danger_bench_smoke.json" --counts "$OUT/schedules/bench_counts.json" --seed 2 ||
    fail "benchmark_plan.py failed" "read the traceback above"
run_logged "$OUT/logs/13b_collect_danger_bench.log" timeout 1800 "$VPY" collect_exploration_trajs.py --schedule "$OUT/schedules/danger_bench_smoke.json" \
    --output-dir "$D/danger_bench_smoke_001" --max-seconds 20 --workers 1 --seed 2 ||
    fail "the hold-out collection did not complete" "read $OUT/logs/13b_collect_danger_bench.log"
run_logged "$OUT/logs/13c_inspect_danger_bench.log" timeout 600 "$VPY" inspect_grasp_dataset.py "$D/danger_bench_smoke_001" --report "$OUT/reports/danger_bench_smoke_validation.json" ||
    fail "the validator rejected the hold-out episodes" "read $OUT/reports/danger_bench_smoke_validation.json"
run_logged "$OUT/logs/13d_benchmark_split.log" timeout 600 "$VPY" benchmark_split.py --dataset "$D/danger_bench_smoke_001" --seed 0 --dev-fraction 0.34 --output "$D/danger_bench_split_smoke_001" ||
    fail "benchmark_split.py failed on the hold-out collection" "read $OUT/logs/13d_benchmark_split.log"
run_logged "$OUT/logs/13e_prepare_bench_holdout.log" timeout 1800 "$VPY" -m wm_training.prepare_bench --split "$D/danger_bench_split_smoke_001/benchmark_split.json" \
    --reference-cache "$D/wm_full_smoke_001" --output "$D/wm_bench_smoke_001" --workers 1; RC=$?
[ $RC = 0 ] && jcheck "$D/wm_bench_smoke_001/index.json" 'd["status"] == "complete" and len(d["episodes"]["val"]) == 3' "complete hold-out cache with 3 rows" ||
    fail "prepare_bench did not write the hold-out cache (exit $RC)" "read $OUT/logs/13e_prepare_bench_holdout.log"
HOLDOUT_EVENT=$(jcheck "$D/danger_bench_split_smoke_001/benchmark_split.json" 'any("|" in k and not k.startswith("negative") and len(v["test"]) >= 1 for k, v in d["strata"].items())' x >/dev/null && echo 1 || echo 0)
ok "3 attempts -> wm_bench_smoke_001 (bench_test fall event: $([ $HOLDOUT_EVENT = 1 ] && echo yes || echo no))"

# ---- 14. state restore check ---------------------------------------------------------------------------------------
step 14 "state restore check" "bench_restore_check.py --mode both (replay == recording, re-rendered pixels, no-return leads)"
if [ $HOLDOUT_EVENT = 1 ]; then
    run_logged "$OUT/logs/14_bench_restore.log" timeout 1200 "$VPY" bench_restore_check.py --split "$D/danger_bench_split_smoke_001/benchmark_split.json" \
        --output "$R/bench_restore_smoke_001" --mode both --events 1 --leads 1,2,4 --hold-steps 20 --no-retreat --render --seed 0; RC=$?
    [ $RC = 0 ] && jcheck "$R/bench_restore_smoke_001/summary.json" 'd["determinism"]["max_qpos_diff"] == 0 and d["determinism"]["pixels"]["identical_frames"] == d["determinism"]["pixels"]["frames_compared"] >= 1' "replay reproduces the recording bit for bit and the re-rendered frames equal the PNGs" ||
        fail "bench_restore_check.py did not reproduce the recording (exit $RC)" "read $OUT/logs/14_bench_restore.log; a different physics build (mujoco != 3.11.0) breaks state restore; different pixels -> the renderer is not the one the PNGs were made with"
    ok "$("$VPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); a=d["determinism"]; print("max qpos diff", a["max_qpos_diff"], "| re-rendered frames identical", a["pixels"]["identical_frames"], "/", a["pixels"]["frames_compared"], "| no-return rollouts at leads 1,2,4")' "$R/bench_restore_smoke_001/summary.json")"
else
    echo "   skipped: the 3-attempt hold-out collection has no bench_test fall event this time (needs one with an onset)"
    ok "skipped (no hold-out event)"
fi

# ---- 15. counterfactual branches (data source 3) ---------------------------------------------------------------------
step 15 "counterfactual branches" "collect_counterfactual_branches.py: 1 training fall event x leads 4,8 x {hold, push_back}"
run_logged "$OUT/logs/15a_collect_cf.log" timeout 1800 "$VPY" collect_counterfactual_branches.py --split "$D/danger_train_split_smoke_001/benchmark_split.json" \
    --output "$D/cf_branches_smoke_001" --part test --events 1 --order listed --leads 4,8 --branches hold,push_back --hold-steps 20 --prefix 5 --move-steps 5 --workers 1 --seed 0; RC=$?
[ $RC = 0 ] && jcheck "$D/cf_branches_smoke_001/manifest.json" 'd["status"] == "complete" and d["interaction_revision"] == 4 and sum(e.get("status") == "complete" for e in d["episodes"]) == 4 and d["summary"]["inexact_replays"] == 0' "4 complete branches, interaction_revision 4, exact replays" ||
    fail "collect_counterfactual_branches.py did not produce 4 exact branches (exit $RC)" "read $OUT/logs/15a_collect_cf.log (less than 20 GiB free? the event's onset < 11 frames?)"
run_logged "$OUT/logs/15b_inspect_cf.log" timeout 600 "$VPY" inspect_grasp_dataset.py "$D/cf_branches_smoke_001" --report "$OUT/reports/cf_branches_smoke_validation.json" ||
    fail "the validator rejected the counterfactual branches" "read $OUT/reports/cf_branches_smoke_validation.json"
ok "4 branch episodes x 26 states, replays exact, validated (the real campaign: 5,400 branches)"

# ---- 16. counterfactual split + cache ---------------------------------------------------------------------------------
step 16 "counterfactual split + cache" "counterfactual_split.py -> wm_training.prepare_bench (the wm_cf_branches_001 recipe)"
run_logged "$OUT/logs/16a_cf_split.log" timeout 600 "$VPY" counterfactual_split.py --dataset "$D/cf_branches_smoke_001" --output "$D/cf_split_smoke_001" ||
    fail "counterfactual_split.py failed" "read $OUT/logs/16a_cf_split.log"
run_logged "$OUT/logs/16b_prepare_bench_cf.log" timeout 1800 "$VPY" -m wm_training.prepare_bench --split "$D/cf_split_smoke_001/benchmark_split.json" \
    --reference-cache "$D/wm_full_smoke_001" --output "$D/wm_cf_smoke_001" --workers 1; RC=$?
[ $RC = 0 ] && jcheck "$D/wm_cf_smoke_001/index.json" 'd["status"] == "complete" and len(d["episodes"]["val"]) == 4' "complete cache with 4 rows" ||
    fail "prepare_bench did not write the counterfactual cache (exit $RC)" "read $OUT/logs/16b_prepare_bench_cf.log"
ok "wm_cf_smoke_001: 4 rows, normalisation copied from wm_full_smoke_001"

# ---- 17. stand-in run records ---------------------------------------------------------------------------------------
step 17 "stand-in run records" "two JSON stubs for records that training steps (not in this repository) would produce"
"$VPY" - "$D" "$OUT/stubs" <<'PYEOF' || fail "could not write the stubs" "read the traceback above"
import hashlib, json, os, sys
data, stubs = sys.argv[1], sys.argv[2]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
full = json.load(open(os.path.join(data, "wm_full_smoke_001", "index.json")))
val_ids = [int(r["id"]) for r in full["episodes"]["val"]]
# classifier run: the planners only read which val episodes are the classifier's dev half (held out) and report half
json.dump({"note": "stand-in for runs/safety_classifier_001/run.json: only dev_episode_ids / report_episode_ids are read",
           "dev_episode_ids": [], "report_episode_ids": val_ids}, open(os.path.join(stubs, "classifier_run.json"), "w"), indent=2)
# fine-tune run record of model B: binds the caches by sha256 and names the bench rows B was trained on
bench = os.path.join(data, "wm_train_bench_smoke_001")
json.dump({"note": "stand-in for runs/lewm_ft_B_001/run.json: only the keys below are read",
           "cache_index_sha256": sha(os.path.join(data, "wm_full_smoke_001", "index.json")),
           "bench_cache": os.path.abspath(bench), "bench_cache_index_sha256": sha(os.path.join(bench, "index.json")),
           "finetune": {"data": {"bench_rows_used_for_training": "bench_test"}}}, open(os.path.join(stubs, "ft_run.json"), "w"), indent=2)
print("   classifier_run.json: report ids", val_ids, "| ft_run.json: bench_test rows of wm_train_bench_smoke_001")
PYEOF
ok "$OUT/stubs/classifier_run.json, ft_run.json"

# ---- 18. planner-action-law branches: plan (data source 4, stage 1) ---------------------------------------------------
step 18 "action-law branch plan" "rrt_branches_plan.py (anchors T/F/G/M from the training cache + training-bench rows)"
run_logged "$OUT/logs/18a_rrt_plan.log" timeout 600 "$VPY" rrt_branches_plan.py --output "$R/rrt_branches_plan_smoke_001" --full-cache "$D/wm_full_smoke_001" \
    --bench-cache "$D/wm_train_bench_smoke_001" --bench-split "$D/danger_train_split_smoke_001/benchmark_split.json" --ft-run "$OUT/stubs/ft_run.json" \
    --classifier-run "$OUT/stubs/classifier_run.json" --quota T=2,F=2,G=2,M=2 --workers-for-estimate 1 --seed 20260928; RC=$?
[ $RC = 0 ] && jcheck "$R/rrt_branches_plan_smoke_001/summary.json" 'd["anchors"] >= 1' "at least one anchor" ||
    fail "rrt_branches_plan.py selected no anchor (exit $RC)" "read $OUT/logs/18a_rrt_plan.log; with 6 training episodes no frame met the T/F/G/M rules -> re-run with another --seed in step 7 or more episodes"
run_logged "$OUT/logs/18b_rrt_capacity.log" timeout 600 "$VPY" rrt_branches_plan.py --output "$R/rrt_anchor_capacity_smoke_001" --full-cache "$D/wm_full_smoke_001" \
    --bench-cache "$D/wm_train_bench_smoke_001" --bench-split "$D/danger_train_split_smoke_001/benchmark_split.json" --ft-run "$OUT/stubs/ft_run.json" \
    --classifier-run "$OUT/stubs/classifier_run.json" --capacity-only --workers-for-estimate 1 --seed 20260928 ||
    fail "rrt_branches_plan.py --capacity-only failed" "read $OUT/logs/18b_rrt_capacity.log"
ok "$("$VPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["anchors"], "anchors, by stratum", d["counts"]["by_stratum"])' "$R/rrt_branches_plan_smoke_001/summary.json")"

# ---- 19. collect the law rollouts (stage 2) ------------------------------------------------------------------------------
step 19 "action-law branch collection" "collect_rrt_branches.py (8 rollouts x 32 steps per anchor, prefix pixels checked against the cache)"
run_logged "$OUT/logs/19_collect_rrt.log" timeout 2400 "$VPY" collect_rrt_branches.py --plan "$R/rrt_branches_plan_smoke_001" --output "$D/rrt_branches_smoke_001" \
    --report "$R/rrt_branches_smoke_001" --workers 1; RC=$?
[ $RC = 0 ] && jcheck "$D/rrt_branches_smoke_001/manifest.json" 'd["status"] == "complete" and d["format"] == "rrt_branches_v1" and len(d["anchors"]) >= 1' "complete rrt_branches_v1 dataset" &&
    jcheck "$R/rrt_branches_smoke_001/summary.json" 'd["sanity"]["recorded_exact_anchors"] == d["sanity"]["anchors"] and d["sanity"]["prefix_frames_cache_max_abs_diff"] == 0' "recorded rollouts exact, prefix frames equal the cache pixels" ||
    fail "collect_rrt_branches.py did not complete or its sanity checks failed (exit $RC)" "read $OUT/logs/19_collect_rrt.log and $R/rrt_branches_smoke_001/REPORT_ZH.md"
ok "$("$VPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2]))["sanity"]; print(len(d["anchors"]), "anchors,", sum(len(a["rollouts"]) for a in d["anchors"]), "rollouts x 32 steps; recorded exact", s["recorded_exact_anchors"], "/", s["anchors"], "; prefix frames == cache", s["prefix_frames_equal_cache"], "(the real campaign: 5,340 anchors / 44,729 rollouts)")' "$D/rrt_branches_smoke_001/manifest.json" "$R/rrt_branches_smoke_001/summary.json")"

# ---- 20. convert to a fine-tune cache (check-only) ---------------------------------------------------------------------------
step 20 "law-branch cache conversion" "convert_rrt_branches.py --check-only (the wm_rrt_branches_001 recipe; its post-conversion audit needs the training modules, which are not shipped, and is skipped)"
run_logged "$OUT/logs/20_convert_rrt.log" timeout 1800 "$VPY" convert_rrt_branches.py --role train --dataset "$D/rrt_branches_smoke_001" --report "$R/rrt_branches_smoke_001" \
    --output "$OUT/checks/wm_rrt_smoke_check_001" --reference-cache "$D/wm_full_smoke_001" --bench-cache "$D/wm_train_bench_smoke_001" --holdout-cache "$D/wm_bench_smoke_001" \
    --bench-split "$D/danger_train_split_smoke_001/benchmark_split.json" --classifier-run "$OUT/stubs/classifier_run.json" --check-only --workers 1; RC=$?
[ $RC = 0 ] && jcheck "$OUT/checks/wm_rrt_smoke_check_001/summary.json" 'd["status"] == "complete" and d["check_only"] is True and d["episodes"] >= 1 and d["verification"]["upstream_raw_audit"]["rows"] == d["episodes"] and d["verification"]["post_conversion_audit"] == "skipped: training modules not shipped in this release"' "complete check conversion; every row passed the label audit and the upstream raw audit; post-conversion audit recorded as skipped" ||
    fail "convert_rrt_branches.py failed or its checks did not pass (exit $RC)" "read $OUT/logs/20_convert_rrt.log and $OUT/checks/wm_rrt_smoke_check_001/FAILED.json"
ok "$("$VPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); v=d["verification"]; print(d["episodes"], "cache episodes x 35 states from", d["anchors"], "anchors; label audit + upstream raw audit passed; post-conversion audit", v["post_conversion_audit"])' "$OUT/checks/wm_rrt_smoke_check_001/summary.json")"

# ---- 21. render videos (see the data without a display) -------------------------------------------------------------------
step 21 "render videos" "make_video.py: the first exploration episode, the counterfactual branches side by side (--grid), the first law-branch anchor as a grid"
V=$OUT/videos
run_logged "$OUT/logs/21a_video_episode.log" timeout 600 "$VPY" make_video.py "$D/exploration_smoke_001/episode_0000" --out "$V" ||
    fail "make_video.py failed on the exploration episode" "read $OUT/logs/21a_video_episode.log"
run_logged "$OUT/logs/21b_video_cf_grid.log" timeout 600 "$VPY" make_video.py "$D/cf_branches_smoke_001" --grid --all --out "$V" ||
    fail "make_video.py --grid failed on the counterfactual branches" "read $OUT/logs/21b_video_cf_grid.log"
ANCHOR=$(find "$D/rrt_branches_smoke_001" -maxdepth 1 -name 'anchor_*.npz' | LC_ALL=C sort | head -1)
[ -n "$ANCHOR" ] || fail "no anchor_*.npz in $D/rrt_branches_smoke_001" "step 19 writes one file per anchor; read $OUT/logs/19_collect_rrt.log"
run_logged "$OUT/logs/21c_video_anchor.log" timeout 600 "$VPY" make_video.py "$ANCHOR" --out "$V" ||
    fail "make_video.py failed on the law-branch anchor" "read $OUT/logs/21c_video_anchor.log"
for pat in 'exploration_smoke_001__episode_0000.gif' 'cf_branches_smoke_001__src*_grid.gif' 'rrt_branches_smoke_001__anchor_*.gif'; do
    f=$(find "$V" -maxdepth 1 -name "$pat" | LC_ALL=C sort | head -1)
    [ -n "$f" ] && [ "$(wc -c < "$f")" -gt 102400 ] || fail "no GIF matching $pat above 100 kB in $V" "read $OUT/logs/21*_video_*.log"
done
ok "$(find "$V" -maxdepth 1 -name '*.gif' | wc -l) GIFs + $(find "$V" -maxdepth 1 -name '*_strip.png' | wc -l) contact sheets in $V"

# ---- 22. unit tests (optional) ---------------------------------------------------------------------------------------------
if [ $WITH_TESTS = 1 ]; then
step 22 "unit tests" "the 13 suites that need no simulator (~20 s; real-data tests skip)"
TESTS="tests.test_task_progress tests.test_benchmark_plan tests.test_benchmark_split tests.test_bench_precheck_report tests.test_rrt_action_law tests.test_rrt_pilot_yields tests.test_rrt_plan_cache_pixels tests.test_convert_rrt_branches tests.test_collect_counterfactual_branches tests.test_make_video wm_training.tests.test_data wm_training.tests.test_parallel_prepare wm_training.tests.test_danger_strata"
run_logged "$OUT/logs/22_unittest.log" timeout 2400 "$VPY" -m unittest $TESTS; RC=$?
[ $RC = 0 ] || fail "some unit tests failed (exit $RC)" "read $OUT/logs/22_unittest.log; the simulator suites (tests.test_exploration, tests.test_rrt_branches, ...) are listed in README 'Tests'"
ok "$(grep -E '^Ran [0-9]+ tests' "$OUT/logs/22_unittest.log" | tail -1) - $(grep -E '^(OK|FAILED)' "$OUT/logs/22_unittest.log" | tail -1) (real-data tests skip)"
fi

# ---- summary ----------------------------------------------------------------------------------------------------------------
echo; echo "== summary (total $((SECONDS - T0)) s)"
printf '%s\n' "${SUMMARY[@]}"
echo "  python: $PWD/$VPY   MUJOCO_GL=$GL"
echo "  outputs: $PWD/$OUT  (data/ raw episodes, caches and branch datasets; runs/ plans and reports; reports/ validation; logs/)"
echo "  videos:  $PWD/$V  (one .gif + one _strip.png contact sheet per clip; python make_video.py --help renders more)"
echo "           to watch a .gif:  WSL: explorer.exe $V    Linux desktop: xdg-open $V/<name>.gif    no display: copy the files to a machine with a browser"
[ $WITH_TESTS = 1 ] || echo "  unit tests not run (add --with-tests, ~20 s; the simulator suites are listed in README 'Tests')"
echo "ALL STEPS OK"
