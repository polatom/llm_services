#!/bin/bash
# ─────────────────────────────────────────────────────────────
# One-time setup: create your own Python venv with vLLM for
# AMD MI210 GPUs (ROCm) on the UFAL LRC cluster.
#
# WHY: Hrabal's pre-compiled vLLM env works (Option A in README),
# but if you want your own independent environment — e.g. to pin
# a specific version or not depend on someone else's home dir —
# this script creates one under /lnet/work/people/$USER/.
#
# IMPORTANT: Run this on a GPU node (inside an srun job), NOT on
# the login node. The ROCm libraries at /opt/rocm are only
# available on GPU nodes, and some pip packages need them at
# install time to detect the AMD GPU backend.
#
# Usage:
#   1. Get on a GPU node:
#      srun -p gpu-amd -c 16 -G 2 --mem=32G -t 2:00:00 --pty bash
#   2. Run this script:
#      bash setup_env.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
# Everything goes under /lnet/work (50 GB quota) instead of $HOME
# to avoid filling the home directory quota.

WORK_BASE="/lnet/work/people/$USER"
VENV_DIR="${VENV_DIR:-$WORK_BASE/.venvs/sprint-vllm-rocm}"

export PIP_CACHE_DIR="$WORK_BASE/.cache/pip"
export HF_HOME="$WORK_BASE/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

echo "======================================================"
echo "  SPRINT vLLM — ROCm Environment Setup"
echo "  venv:      $VENV_DIR"
echo "  pip cache: $PIP_CACHE_DIR"
echo "  HF cache:  $HF_HOME"
echo "======================================================"

# ── Step 0: Verify we're on a GPU node ───────────────────────

if [ ! -d /opt/rocm ]; then
    echo ""
    echo "ERROR: /opt/rocm not found. You're probably on the login node."
    echo "       Run this inside an interactive GPU job first:"
    echo "       srun -p gpu-amd -c 16 -G 2 --mem=32G -t 2:00:00 --pty bash"
    exit 1
fi

# ── Step 1: Set ROCm paths ───────────────────────────────────

export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64/:"${LD_LIBRARY_PATH:-}"
export PATH=/opt/rocm/bin:"$PATH"

echo ""
echo "ROCm: $(cat /opt/rocm/.info/version 2>/dev/null || echo 'version unknown')"
rocm-smi --showid 2>/dev/null | head -5 || echo "(rocm-smi not available)"

# ── Step 2: Check disk usage ─────────────────────────────────

echo ""
echo "Disk usage under $WORK_BASE:"
du -sh "$WORK_BASE" 2>/dev/null || echo "  (could not check)"
echo ""

# ── Step 3: Check Python ─────────────────────────────────────

PYTHON="${PYTHON:-python3}"
PY_VERSION=$($PYTHON --version 2>&1)
echo "Python: $PY_VERSION"

$PYTHON -c "import sys; assert sys.version_info >= (3, 9), 'Python 3.9+ required'" || {
    echo "ERROR: Python 3.9+ is required."
    exit 1
}

# ── Step 4: Create or update venv ─────────────────────────────

if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "venv already exists at $VENV_DIR"
    read -p "Recreate from scratch? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing venv. Upgrading vLLM..."
        source "$VENV_DIR/bin/activate"
        pip install --upgrade vllm
        echo "Done. vLLM: $(python3 -c 'import vllm; print(vllm.__version__)')"
        exit 0
    fi
fi

echo ""
echo "Creating venv at $VENV_DIR ..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ── Step 5: Install PyTorch for ROCm ─────────────────────────
# Standard PyTorch wheels are CUDA-only. For AMD GPUs we need the
# ROCm build. The --index-url points to PyTorch's ROCm wheel repo.
# rocm6.2 is the closest match to the cluster's ROCm 6.4.1.

echo ""
echo "Installing PyTorch for ROCm..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# ── Step 6: Install vLLM ─────────────────────────────────────
# pip install vllm may install a CUDA wheel. If it doesn't detect
# ROCm correctly, you'll need to build from source (see below).

echo ""
echo "Installing vLLM..."
pip install vllm

# ── Step 7: Verify ───────────────────────────────────────────

echo ""
echo "======================================================"
VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>&1) || VLLM_VER="IMPORT FAILED"
TORCH_VER=$(python3 -c "import torch; print(f'{torch.__version__}, ROCm: {torch.version.hip}')" 2>&1) || TORCH_VER="IMPORT FAILED"

echo "  vLLM:    $VLLM_VER"
echo "  PyTorch: $TORCH_VER"
echo "  venv:    $VENV_DIR"
echo ""

if [[ "$VLLM_VER" == *"FAILED"* ]] || [[ "$TORCH_VER" == *"FAILED"* ]]; then
    echo "  WARNING: Something failed. vLLM may need to be built from source."
    echo "  Try:"
    echo "    pip install wheel packaging ninja cmake"
    echo "    git clone https://github.com/vllm-project/vllm.git /tmp/vllm-src"
    echo "    cd /tmp/vllm-src"
    echo "    VLLM_TARGET_DEVICE=rocm pip install -e ."
    echo ""
    echo "  This will take 30+ minutes but produces a ROCm-native build."
else
    echo "  Setup complete! To use:"
    echo "    bash run_vllm.sh --env $VENV_DIR"
fi
echo "======================================================"
