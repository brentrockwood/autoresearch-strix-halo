#!/bin/bash
# autoresearch Strix Halo setup script
# Run from inside the cloned autoresearch repo root.
# Rockwood Lab LLC — https://rockwoodlab.com

set -euo pipefail

ROCM_INDEX="https://repo.amd.com/rocm/whl/gfx1151"
TORCH_VERSION="2.9.1+rocm7.10.0"

# ── Preflight ────────────────────────────────────────────────────────────────

echo "=== Checking prerequisites ==="

if ! command -v uv &>/dev/null; then
    echo "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! rocminfo &>/dev/null; then
    echo "ERROR: rocminfo not found. Is ROCm installed?"
    echo "Run scripts/install_rocm.sh first, or install ROCm manually."
    exit 1
fi

if ! rocminfo 2>/dev/null | grep -q "gfx1151"; then
    echo "WARNING: gfx1151 not detected by rocminfo. Proceeding anyway."
fi

if [[ ! -f "train.py" ]]; then
    echo "ERROR: train.py not found. Run this from the autoresearch repo root."
    exit 1
fi

echo "Prerequisites OK."

# ── Python venv ──────────────────────────────────────────────────────────────

echo ""
echo "=== Creating venv and installing base dependencies ==="

# Pin Python 3.12 (required for ROCm gfx1151 wheels)
echo "3.12" > .python-version

uv venv --python 3.12
uv sync --extra rocm-gfx1151 || {
    echo "uv sync with rocm extra failed — falling back to cpu extra for non-torch deps"
    uv sync --extra cpu
}

# Seed pip into the uv venv (uv doesn't include it by default)
.venv/bin/python -m ensurepip --upgrade

# ── ROCm PyTorch ─────────────────────────────────────────────────────────────

echo ""
echo "=== Installing ROCm PyTorch wheel ==="
echo "Source: $ROCM_INDEX"
echo "Version: $TORCH_VERSION"

# Remove whatever uv installed (likely the CUDA wheel)
.venv/bin/python -m pip uninstall -y torch torchvision pytorch-triton 2>/dev/null || true

# Install ROCm torch — must use --no-cache-dir to avoid pip serving cached CUDA wheel
.venv/bin/python -m pip install "torch==$TORCH_VERSION" \
    --index-url "$ROCM_INDEX" \
    --pre --no-deps --no-cache-dir

# Install torchvision (rocm7.10.0 build)
.venv/bin/python -m pip install "torchvision" \
    --index-url "$ROCM_INDEX" \
    --pre --no-deps --no-cache-dir

# Install pytorch-triton-rocm (must match torch ROCm version)
.venv/bin/python -m pip install pytorch-triton-rocm \
    --index-url "$ROCM_INDEX" \
    --pre --no-deps --no-cache-dir

# ── Verify ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Verifying installation ==="

.venv/bin/python -c "
import torch
print('torch version:  ', torch.__version__)
print('hip version:    ', torch.version.hip)
print('GPU available:  ', torch.cuda.is_available())
print('device count:   ', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device name:    ', torch.cuda.get_device_name(0))
    print('memory (GB):    ', torch.cuda.get_device_properties(0).total_memory / 1e9)
else:
    print()
    print('ERROR: GPU not visible to PyTorch.')
    print('Check: groups (need render, video) | rocm-smi | rocminfo')
    exit(1)
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Run data prep (one-time):"
echo "     TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/bin/python prepare.py"
echo ""
echo "  2. Run a baseline experiment:"
echo "     TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/bin/python train.py"
echo ""
echo "  3. Start the agent loop:"
echo "     Point Claude Code at program.md"
echo "     Prompt: 'Have a look at program.md and kick off a new experiment.'"
echo ""
echo "  NOTE: Always use .venv/bin/python, not uv run."
echo "        uv's lockfile references the CUDA wheel and will override ROCm."
