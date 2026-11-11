#!/usr/bin/env bash
# P1 on a fresh VM, up to the K0 gate (then stop and report; CLAUDE.md).
# Usage, from the repository root on the VM:
#   mkdir -p logs && nohup bash scripts/p1_start.sh --qwen bf16 > logs/p1_start.log 2>&1 &   (or --qwen fp8: decision D7)
# Steps: environment and CPU tests; Qwen3 download in the background; OLMoE downloads; data and K0 for OLMoE;
# K0 for Qwen3 once its download finishes. Every GPU step waits for free memory first (shared GPU).
# Memory caps (GB): OLMOE_MEM_GB (default 40: K0.1 runs OLMoE in fp32), QWEN_MEM_GB (default 75 bf16, 45 fp8).
set -euo pipefail
cd "$(dirname "$0")/.."
QWEN=""
while [[ $# -gt 0 ]]; do
  case "$1" in --qwen) QWEN="$2"; shift 2 ;; *) echo "unknown argument $1"; exit 2 ;; esac
done
[[ "$QWEN" == "bf16" || "$QWEN" == "fp8" ]] || { echo "give --qwen bf16 or --qwen fp8 (decision D7)"; exit 2; }
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
OLMOE_MEM_GB="${OLMOE_MEM_GB:-40}"
if [[ "$QWEN" == "bf16" ]]; then QWEN_MEM_GB="${QWEN_MEM_GB:-75}"; NEED_GB=150; FP8=""; else QWEN_MEM_GB="${QWEN_MEM_GB:-45}"; NEED_GB=120; FP8="--fp8"; fi
mkdir -p logs "$HF_HOME"
stamp() { echo "== $(date -u +%FT%TZ) $*"; }

stamp "start: git $(git rev-parse --short HEAD), D7 = $QWEN, caps OLMoE $OLMOE_MEM_GB GB, Qwen3 $QWEN_MEM_GB GB"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
free_gb=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc 0-9)
[[ "$free_gb" -ge "$NEED_GB" ]] || { echo "only ${free_gb} GB free under HF_HOME, need about ${NEED_GB} GB"; exit 1; }

stamp "environment and CPU tests"
bash scripts/vm_setup.sh
PY=.venv/bin/python

stamp "Qwen3 download ($QWEN) in the background: logs/download_qwen3.log"
$PY scripts/download_models.py --model qwen3_30b_a3b $FP8 --fill > logs/download_qwen3.log 2>&1 &
QPID=$!

stamp "OLMoE downloads"
$PY scripts/download_models.py --model olmoe_1b_7b --fill
$PY scripts/download_models.py --model olmoe_1b_7b_instruct --fill

stamp "OLMoE: data and K0"
bash scripts/run_kill_test.sh --model olmoe --stop-after k0 --mem-gb "$OLMOE_MEM_GB"

stamp "waiting for the Qwen3 download"
wait "$QPID" || { echo "Qwen3 download failed: see logs/download_qwen3.log"; exit 1; }
tail -3 logs/download_qwen3.log

stamp "Qwen3: data and K0"
bash scripts/run_kill_test.sh --model qwen3 --stop-after k0 --mem-gb "$QWEN_MEM_GB"

stamp "K0 done for both models: stop and report (results/kill_test/, logs/). Then pull results to the laptop."
