#!/usr/bin/env bash
# Kill test pipeline (PHASE1_KILL_TEST.md). Usage:
#   bash scripts/run_kill_test.sh [--tiny] [--model olmoe|qwen3|both] [--force] [--mem-gb N] [--stop-after k0]
# --stop-after k0 runs data preparation and the K0 sanity checks only, then stops for the K0 report (CLAUDE.md:
# stop at every gate). Rerun without it to continue; finished steps are skipped.
# Chains: prepare_data, sanity_real_model (K0, stops on failure), calibrate, record_analysis,
# run_layer_eval (K1, K2), tune_tau, run_downstream (K3), make_kill_report.
# Every step skips finished outputs unless --force, so the script resumes after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

TINY=""; MODEL="both"; PASS=(); STOP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiny) TINY="--tiny"; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --force) PASS+=("--force"); shift ;;
    --mem-gb) PASS+=("--mem-gb" "$2"); shift 2 ;;
    --stop-after) STOP="$2"; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done

if [[ -n "${PYTHON:-}" ]]; then PY="$PYTHON"
elif [[ -x .venv/bin/python ]]; then PY=".venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then PY=".venv/Scripts/python.exe"
else PY="python"; fi
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
[[ -n "$TINY" ]] && export HF_HUB_OFFLINE=1

case "$MODEL" in
  olmoe) KEYS=(olmoe_1b_7b) ;;
  qwen3) KEYS=(qwen3_30b_a3b) ;;
  both)  KEYS=(olmoe_1b_7b qwen3_30b_a3b) ;;
  *) echo "--model must be olmoe, qwen3 or both"; exit 2 ;;
esac
[[ -z "$STOP" || "$STOP" == "k0" ]] || { echo "--stop-after takes k0"; exit 2; }

run() { echo ">>> $*"; "$PY" "$@"; }
# expand an empty array safely under set -u
P() { if [[ ${#PASS[@]} -gt 0 ]]; then echo "${PASS[@]}"; fi; }

for KEY in "${KEYS[@]}"; do
  A=(--model "$KEY" $TINY $(P))
  run scripts/prepare_data.py "${A[@]}"
  run scripts/sanity_real_model.py "${A[@]}" || { echo "K0 failed for $KEY: stop (PHASE1_KILL_TEST.md Section 3)"; exit 1; }
  if [[ "$STOP" == "k0" ]]; then echo "K0 done for $KEY (--stop-after k0)"; continue; fi
  run scripts/calibrate.py "${A[@]}" --target stats
  run scripts/calibrate.py "${A[@]}" --target task
  run scripts/record_analysis.py "${A[@]}" --target stats
  run scripts/record_analysis.py "${A[@]}" --target task
  run scripts/run_layer_eval.py "${A[@]}"
  run scripts/tune_tau.py "${A[@]}" --target task
  run scripts/run_downstream.py "${A[@]}"
done
if [[ "$STOP" == "k0" ]]; then echo "stopped after K0: report K0, then rerun without --stop-after"; exit 0; fi
run scripts/make_kill_report.py $TINY
if [[ -n "$TINY" ]]; then echo "done: results/tiny/KILL_REPORT.md"; else echo "done: results/kill_test/KILL_REPORT.md"; fi
