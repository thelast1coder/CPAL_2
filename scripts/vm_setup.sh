#!/usr/bin/env bash
# VM bootstrap (PHASE0_SETUP.md Section 6): HF venv (.venv) and vLLM venv (.venv-vllm), installs, versions,
# GPU and disk report, then the CPU tests. Run from the repository root: bash scripts/vm_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_HOME:?set HF_HOME to the big disk first}"
PYBIN="${PYBIN:-python3}"

echo "== GPU (shared with Mirror_CPAL and a CVPR project: never kill processes we did not start)"
nvidia-smi || echo "nvidia-smi not available"
echo "== free disk under HF_HOME"
mkdir -p "$HF_HOME"; df -h "$HF_HOME"

echo "== HF venv"
"$PYBIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
PIP_NO_CACHE_DIR=1 .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps

echo "== vLLM venv (separate: vLLM pins its own transformers)"
"$PYBIN" -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade pip
PIP_NO_CACHE_DIR=1 .venv-vllm/bin/python -m pip install -r requirements-vllm.txt
.venv-vllm/bin/python -m pip install -e . --no-deps

echo "== versions"
.venv/bin/python - <<'EOF'
import sys, torch, transformers
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "transformers", transformers.__version__, "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
EOF
.venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)" || echo "vllm import failed (check later, P2 only)"

echo "== CPU tests (must all pass before anything else)"
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
