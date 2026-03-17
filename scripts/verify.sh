#!/bin/bash
# Verify autoresearch ROCm setup is working correctly.
# Run from the autoresearch repo root.
# Rockwood Lab LLC — https://rockwoodlab.com

set -euo pipefail

PASS=0
FAIL=0

check() {
    local label="$1"
    local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo "  ✓ $label"
        ((PASS++)) || true
    else
        echo "  ✗ $label"
        ((FAIL++)) || true
    fi
}

echo ""
echo "=== autoresearch ROCm verification ==="
echo ""

echo "System:"
check "rocm-smi available"        "command -v rocm-smi"
check "rocminfo available"        "command -v rocminfo"
check "gfx1151 detected"          "grep -q gfx1151 < <(rocminfo)"
check "/dev/kfd exists"           "test -e /dev/kfd"
check "/dev/dri/renderD128 exists" "test -e /dev/dri/renderD128"
check "user in render group"      "groups | grep -q render"
check "user in video group"       "groups | grep -q video"

echo ""
echo "Python environment:"
check ".venv exists"              "test -d .venv"
check ".venv/bin/python exists"   "test -f .venv/bin/python"
check "ROCm torch installed"      ".venv/bin/python -c 'import torch; assert torch.version.hip is not None'"
check "GPU visible to PyTorch"    ".venv/bin/python -c 'import torch; assert torch.cuda.is_available()'"
check "train.py present"          "test -f train.py"
check "prepare.py present"        "test -f prepare.py"
check "SDPA patch applied"        "grep -q 'scaled_dot_product_attention' train.py"
check "WINDOW_PATTERN=L"          "grep -q 'WINDOW_PATTERN = \"L\"' train.py"
check "ROCm env var in train.py"  "grep -q 'TORCH_ROCM_AOTRITON' train.py"

echo ""
echo "Data:"
check "tokenizer exists"          "test -f ~/.cache/autoresearch/tokenizer/tokenizer.pkl"
check "data shards exist"         "test -d ~/.cache/autoresearch/data"

echo ""

if [[ $FAIL -eq 0 ]]; then
    echo "All checks passed ($PASS/$((PASS+FAIL))). Ready to train."
    echo ""
    echo "Run:"
    echo "  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/bin/python train.py"
else
    echo "$FAIL check(s) failed. See above."
    if ! .venv/bin/python -c "import torch; assert torch.version.hip" &>/dev/null; then
        echo ""
        echo "ROCm torch not installed. Run: bash scripts/setup.sh"
    fi
    if ! .venv/bin/python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
        echo ""
        echo "GPU not visible. Check groups (render, video) and reboot if recently added."
    fi
fi

echo ""
echo "=== GPU info ==="
.venv/bin/python -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  Device:  {torch.cuda.get_device_name(0)}')
    print(f'  Memory:  {props.total_memory / 1e9:.1f} GB')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  HIP:     {torch.version.hip}')
else:
    print('  GPU not available')
" 2>/dev/null || true
echo ""
