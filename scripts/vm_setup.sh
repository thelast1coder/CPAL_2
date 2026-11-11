#!/usr/bin/env bash
# VM bootstrap (PHASE0_SETUP.md Section 6): HF venv (.venv), versions, GPU and disk report, then the CPU tests.
# The vLLM venv (.venv-vllm) is opt-in (WITH_VLLM=1): it is used only from check V1 in P2.
# Run from the repository root: bash scripts/vm_setup.sh
# Uses uv when available (vast.ai images ship it), else pip. Safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_HOME:?set HF_HOME to the big disk first}"
PYBIN="${PYBIN:-python3}"

echo "== GPU (shared with other projects: never kill processes we did not start)"
nvidia-smi || echo "nvidia-smi not available"
echo "== free disk under HF_HOME"
mkdir -p "$HF_HOME"; df -h "$HF_HOME"
if command -v vast-capabilities >/dev/null 2>&1; then
  vol=$(vast-capabilities 2>/dev/null | "$PYBIN" -c "import json,sys; print(json.load(sys.stdin).get('instance',{}).get('workspace_is_volume'))" 2>/dev/null || echo unknown)
  echo "== vast.ai workspace_is_volume: $vol"
  [[ "$vol" == "True" ]] || echo "WARNING: no persistent volume. Pull results to the laptop often (scripts/pull_results.sh)."
fi

make_venv() {  # make_venv <dir> <requirements file>
  if command -v uv >/dev/null 2>&1; then
    [[ -x "$1/bin/python" ]] || uv venv -q "$1" --python "$PYBIN"
    uv pip install -q --python "$1/bin/python" -r "$2"
    uv pip install -q --python "$1/bin/python" -e . --no-deps
  else
    [[ -x "$1/bin/python" ]] || "$PYBIN" -m venv "$1"
    "$1/bin/python" -m pip install -q --upgrade pip
    PIP_NO_CACHE_DIR=1 "$1/bin/python" -m pip install -q -r "$2"
    "$1/bin/python" -m pip install -q -e . --no-deps
  fi
}

echo "== HF venv"
make_venv .venv requirements.txt
if [[ "${WITH_VLLM:-0}" == "1" ]]; then
  echo "== vLLM venv (separate: vLLM pins its own transformers)"
  make_venv .venv-vllm requirements-vllm.txt
fi

echo "== versions"
.venv/bin/python - <<'PY'
import sys, torch, transformers
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "transformers", transformers.__version__, "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if torch.cuda.is_available():  # catches a wheel without kernels for this GPU's architecture
    print("cuda op ok:", (torch.ones(4, device="cuda") * 2).sum().item() == 8.0)
PY
[[ -x .venv-vllm/bin/python ]] && { .venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)" || echo "vllm import failed (P2 only)"; }

echo "== CPU tests (must all pass before anything else)"
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
