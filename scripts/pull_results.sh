#!/usr/bin/env bash
# Copy results, logs and the filled model configs from the VM to the laptop (run on the laptop).
# vast.ai instances often have no persistent volume, so run this after every step that produces numbers.
# Usage: bash scripts/pull_results.sh <ssh port> <user@host> [--artifacts]
# Files land in vm_pull/<UTC timestamp>/ (never overwrites the working tree; merge by hand).
# --artifacts also copies artifacts/ (calibration and per-token records, about 1 to 2 GB per model).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:?ssh port}"; HOST="${2:?user@host}"; WHAT="results logs configs/models"
[[ "${3:-}" == "--artifacts" ]] && WHAT="$WHAT artifacts"
REMOTE_DIR="${REMOTE_DIR:-/workspace/MoE_CPAL}"
DEST="vm_pull/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$DEST"
ssh -p "$PORT" -o ServerAliveInterval=30 "$HOST" \
  "cd $REMOTE_DIR && tar czf - --ignore-failed-read $WHAT 2>/dev/null" | tar xzf - -C "$DEST"
echo "pulled into $DEST:"; du -sh "$DEST"/* 2>/dev/null || true
