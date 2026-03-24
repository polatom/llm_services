#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Launch multiple independent vLLM servers, one per GPU.
#
# Workaround for vLLM's --data-parallel-size being broken on
# ROCm (GPU assignment via CUDA_VISIBLE_DEVICES doesn't work
# with HIP). Instead, we launch N separate vLLM processes,
# each pinned to one GPU via HIP_VISIBLE_DEVICES, on
# consecutive ports (8421, 8422, ...).
#
# Usage (on the GPU node):
#   bash run_vllm_multi.sh                                    # 8 servers, ports 8421-8428
#   bash run_vllm_multi.sh --num-gpus 4 --base-port 8421     # 4 servers
#   bash run_vllm_multi.sh --model google/gemma-3-12b-it      # different model
#
# Stop all servers:
#   kill $(jobs -p) 2>/dev/null    # if running in foreground
#   pkill -f "vllm serve"          # kill all vLLM processes
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Configuration ────────────────────────────────────────────

MODEL="${MODEL:-swiss-ai/Apertus-8B-Instruct-2509}"
BASE_PORT="${BASE_PORT:-8421}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
DTYPE="${DTYPE:-auto}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --base-port)   BASE_PORT="$2"; shift 2 ;;
        --num-gpus)    NUM_GPUS="$2"; shift 2 ;;
        --max-len)     MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-seqs)    MAX_NUM_SEQS="$2"; shift 2 ;;
        --dtype)       DTYPE="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── ROCm environment ────────────────────────────────────────

if [ ! -d /opt/rocm ]; then
    echo "ERROR: /opt/rocm not found — this script is for AMD ROCm GPUs only."
    echo "       For NVIDIA, use run_vllm.sh with --dp (DP works on CUDA)."
    exit 1
fi

export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64/:"${LD_LIBRARY_PATH:-}"
export PATH=/opt/rocm/bin:"$PATH"
export VLLM_TARGET_DEVICE=rocm
export TORCHDYNAMO_DISABLE=1

# ── Python environment ───────────────────────────────────────

WORK_BASE="/lnet/work/people/$USER"
OWN_VENV="$WORK_BASE/.venvs/llm-services-vllm-rocm"
HRABAL_VENV="/lnet/work/home-students-external/hrabal/uv_venv/3.13_vllm0.13_rocm6.4.1"

if [ -d "$OWN_VENV" ]; then
    VENV_PATH="$OWN_VENV"
elif [ -d "$HRABAL_VENV" ]; then
    VENV_PATH="$HRABAL_VENV"
else
    echo "ERROR: No venv found. Run setup_env.sh first."
    exit 1
fi

# Activate venv
PYVENV_CFG="$VENV_PATH/pyvenv.cfg"
if [ -f "$PYVENV_CFG" ]; then
    PYHOME=$(grep -E '^home\s*=' "$PYVENV_CFG" | sed 's/^home\s*=\s*//' | tr -d '[:space:]')
    if [ -n "$PYHOME" ] && [ -d "$PYHOME" ]; then
        export PATH="$PYHOME:$PATH"
    fi
fi
source "$VENV_PATH/bin/activate" 2>/dev/null || true
export PATH="$VENV_PATH/bin:$PATH"

# HuggingFace cache
export HF_HOME="$WORK_BASE/.cache/huggingface"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
mkdir -p "$HF_HOME"

# ── Print configuration ─────────────────────────────────────

LAST_PORT=$((BASE_PORT + NUM_GPUS - 1))
echo "======================================================"
echo "  vLLM Multi-Server (1 per GPU)"
echo "  Node:       $(hostname)"
echo "  Slurm Job:  ${SLURM_JOB_ID:-interactive}"
echo "  Model:      $MODEL"
echo "  GPUs:       $NUM_GPUS (one server each)"
echo "  Ports:      $BASE_PORT — $LAST_PORT"
echo "  Max length: $MAX_MODEL_LEN tokens"
echo "  Max seqs:   $MAX_NUM_SEQS (per server)"
echo "  Python:     $(which python3) ($(python3 --version 2>&1))"
echo "  vLLM:       $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '?')"
echo "  Started:    $(date)"
echo "======================================================"
echo ""

# ── Write connection info ────────────────────────────────────

CONN_FILE="$WORK_BASE/.sprint_vllm_connection"
HOSTNAME=$(hostname)
cat > "$CONN_FILE" <<EOF
VLLM_HOST=$HOSTNAME
VLLM_PORTS=$BASE_PORT-$LAST_PORT
VLLM_NUM_SERVERS=$NUM_GPUS
VLLM_MODEL=$MODEL
SLURM_JOB_ID=${SLURM_JOB_ID:-interactive}
EOF

echo "Connection info: $CONN_FILE"
echo ""

# ── Launch servers ───────────────────────────────────────────

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    GPU_LOG="$LOG_DIR/vllm_gpu${i}_port${PORT}_$(date +%Y%m%d_%H%M%S).log"

    echo "Starting server $i: GPU=$i, port=$PORT, log=$GPU_LOG"

    # Pin GPU via env command for a clean environment.
    # HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES both work on ROCm.
    # CUDA_VISIBLE_DEVICES does NOT work on ROCm (ignored).
    env \
        HIP_VISIBLE_DEVICES="$i" \
        ROCR_VISIBLE_DEVICES="$i" \
        vllm serve "$MODEL" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --tensor-parallel-size 1 \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --dtype "$DTYPE" \
            --enforce-eager \
            --trust-remote-code \
            > "$GPU_LOG" 2>&1 &

    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"

    # Stagger launches to reduce NFS and memory contention during model loading
    if [ "$i" -lt $((NUM_GPUS - 1)) ]; then
        sleep 5
    fi
done

echo ""
echo "======================================================"
echo "  All $NUM_GPUS servers launched."
echo "  Ports: $BASE_PORT — $LAST_PORT"
echo "  PIDs:  ${PIDS[*]}"
echo ""
echo "  Monitor:"
echo "    tail -f $LOG_DIR/vllm_gpu0_port${BASE_PORT}_*.log"
echo "    tail -f $LOG_DIR/vllm_gpu*_*.log"
echo ""
echo "  Test (from lrc1):"
echo "    python3 test_endpoint.py --url http://$HOSTNAME:$BASE_PORT --mode health --model $MODEL"
echo "    python3 test_endpoint.py --mode ponk --model $MODEL \\"
echo "      --urls http://$HOSTNAME:${BASE_PORT}"
for i in $(seq 1 $((NUM_GPUS - 1))); do
    P=$((BASE_PORT + i))
    echo "      --urls http://$HOSTNAME:${P}"
done
echo ""
echo "  Stop all:"
echo "    kill ${PIDS[*]}"
echo "    # or: pkill -f 'vllm serve'"
echo "======================================================"
echo ""
echo "Waiting for servers to exit (Ctrl+C to stop all)..."

# Trap Ctrl+C to kill all child processes
cleanup() {
    echo ""
    echo "Stopping all vLLM servers..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    echo "All servers stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for any server to exit (if one crashes, report it)
while true; do
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            EXIT_CODE=$?
            PORT=$((BASE_PORT + i))
            if [ "$EXIT_CODE" -ne 0 ]; then
                echo "WARNING: Server on GPU $i (port $PORT, PID $pid) exited with code $EXIT_CODE"
                echo "         Check log: $LOG_DIR/vllm_gpu${i}_port${PORT}_*.log"
            else
                echo "Server on GPU $i (port $PORT) exited normally."
            fi
            # Remove from array to avoid re-checking
            unset 'PIDS[i]'
        fi
    done
    # If all servers are gone, exit
    if [ ${#PIDS[@]} -eq 0 ]; then
        echo "All servers have exited."
        exit 0
    fi
    sleep 5
done
