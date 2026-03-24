#!/bin/bash
# ─────────────────────────────────────────────────────────────
# One-time setup: create your own Python venv with vLLM for
# AMD MI210 GPUs (ROCm) on the UFAL LRC cluster.
#
# Creates an independent venv under /lnet/work/people/$USER/
# with PyTorch (ROCm) and vLLM compiled for AMD GPUs.
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
#   3. Then launch vLLM:
#      bash run_vllm.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Logging ────────────────────────────────────────────────────
# Tee all output (stdout + stderr) to a timestamped log file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log file: $LOG_FILE"

# ── Paths ────────────────────────────────────────────────────
# Everything goes under /lnet/work (50 GB quota) instead of $HOME
# to avoid filling the home directory quota.

WORK_BASE="/lnet/work/people/$USER"
VENV_DIR="${VENV_DIR:-$WORK_BASE/.venvs/llm-services-vllm-rocm}"

export PIP_CACHE_DIR="$WORK_BASE/.cache/pip"
export HF_HOME="$WORK_BASE/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

echo "======================================================"
echo "  LLM Services — ROCm Environment Setup"
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

# ── Step 5: Detect ROCm version ───────────────────────────────
# Match PyTorch ROCm wheel to the installed ROCm version.

ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | tr -d '[:space:]')
ROCM_MAJOR_MINOR=$(echo "$ROCM_VER" | grep -oP '^\d+\.\d+')

echo ""
echo "ROCm version: $ROCM_VER (major.minor: $ROCM_MAJOR_MINOR)"

# Map cluster ROCm to closest available PyTorch ROCm wheel.
# PyTorch publishes wheels for specific ROCm versions (6.2, 6.3, etc.).
# Pick the closest match that doesn't exceed the installed version.
case "$ROCM_MAJOR_MINOR" in
    6.4|6.5|6.6) PYTORCH_ROCM="rocm6.4" ;;
    6.3)         PYTORCH_ROCM="rocm6.3" ;;
    6.2)         PYTORCH_ROCM="rocm6.2" ;;
    6.1|6.0)     PYTORCH_ROCM="rocm6.2" ;;
    *)           PYTORCH_ROCM="rocm6.2"
                 echo "WARNING: Unrecognized ROCm $ROCM_MAJOR_MINOR, defaulting to rocm6.2 wheels" ;;
esac

echo "Using PyTorch index: $PYTORCH_ROCM"

# ── Step 6: Install PyTorch for ROCm ─────────────────────────
# Standard PyTorch wheels are CUDA-only. For AMD GPUs we need the
# ROCm build from PyTorch's dedicated wheel index.

echo ""
echo "Installing PyTorch for ROCm ($PYTORCH_ROCM)..."
pip install torch torchvision --index-url "https://download.pytorch.org/whl/$PYTORCH_ROCM"

# Verify PyTorch sees ROCm
TORCH_HIP=$(python3 -c "import torch; print(torch.version.hip or 'None')" 2>&1) || TORCH_HIP="FAILED"
if [[ "$TORCH_HIP" == "None" ]] || [[ "$TORCH_HIP" == "FAILED" ]]; then
    echo ""
    echo "ERROR: PyTorch installed but does not see ROCm (torch.version.hip=$TORCH_HIP)."
    echo "       The wrong wheel may have been installed."
    echo "       Try: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"
    exit 1
fi
echo "PyTorch ROCm backend: $TORCH_HIP ✓"

# ── Step 7: Install vLLM ─────────────────────────────────────
# WARNING: `pip install vllm` pulls in CUDA PyTorch as a dependency,
# overwriting the ROCm version we just installed. We fix this by
# reinstalling ROCm PyTorch afterwards to reclaim the correct backend.

echo ""
echo "Installing vLLM (this will temporarily install CUDA PyTorch)..."
pip install vllm

echo ""
echo "Reinstalling PyTorch for ROCm (overwriting CUDA version pulled by vLLM)..."
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_ROCM" --force-reinstall --no-deps

# Verify ROCm PyTorch is back
TORCH_HIP=$(python3 -c "import torch; print(torch.version.hip or 'None')" 2>&1) || TORCH_HIP="FAILED"
if [[ "$TORCH_HIP" == "None" ]] || [[ "$TORCH_HIP" == "FAILED" ]]; then
    echo "ERROR: ROCm PyTorch reinstall failed (torch.version.hip=$TORCH_HIP)."
    exit 1
fi
echo "PyTorch ROCm backend restored: $TORCH_HIP ✓"

# ── Step 8: Verify ───────────────────────────────────────────

echo ""
echo "======================================================"
VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>&1) || VLLM_VER="IMPORT FAILED"
TORCH_VER=$(python3 -c "import torch; print(f'{torch.__version__}, ROCm: {torch.version.hip}')" 2>&1) || TORCH_VER="IMPORT FAILED"

echo "  vLLM:    $VLLM_VER"
echo "  PyTorch: $TORCH_VER"
echo "  venv:    $VENV_DIR"
echo ""

TORCH_HIP_FINAL=$(python3 -c "import torch; print(torch.version.hip or 'None')" 2>&1) || TORCH_HIP_FINAL="FAILED"

if [[ "$VLLM_VER" == *"FAILED"* ]]; then
    echo "  ERROR: vLLM import failed. Manual debugging needed."
    echo "  Try: VLLM_TARGET_DEVICE=rocm pip install vllm --no-build-isolation"
    exit 1
fi

if [[ "$TORCH_HIP_FINAL" == "None" ]] || [[ "$TORCH_HIP_FINAL" == "FAILED" ]]; then
    echo "  ERROR: PyTorch ROCm backend missing (torch.version.hip=$TORCH_HIP_FINAL)."
    echo "  The ROCm PyTorch reinstall may have failed. Try manually:"
    echo "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$PYTORCH_ROCM --force-reinstall --no-deps"
    exit 1
fi

echo ""
echo "  Setup complete! To launch vLLM:"
echo "    bash run_vllm.sh --env $VENV_DIR"
echo ""
echo "  Or set as default (no --env needed):"
echo "    export VLLM_ENV=$VENV_DIR"
echo "======================================================"
