#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Submit a vLLM server job to Slurm (non-interactive).
#
# This wraps run_vllm.sh in an sbatch script so the server runs
# in the background without needing an interactive terminal.
#
# Usage:
#   bash submit_vllm.sh                          # defaults: H100 1 GPU
#   bash submit_vllm.sh --partition gpu-troja --node tdll-4gpu1 --gpus 1
#   bash submit_vllm.sh --partition gpu-amd --gpus 8 --tp 2 --dp 4
#   bash submit_vllm.sh --time 4:00:00           # 4-hour time limit
#   bash submit_vllm.sh --partition gpu-ms --gpus 2 --tp 2 --dp 1
#
# The server's hostname:port is written to a connection file
# so connect.sh and test_endpoint.py can find it automatically.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────
PARTITION="${PARTITION:-gpu-amd}"
NODE="${NODE:-}"                        # empty = let Slurm pick
GPUS="${GPUS:-8}"
CPUS="${CPUS:-16}"
MEM="${MEM:-0}"                        # 0 = all available (DP>1 loads multiple model copies concurrently)
TIME="${TIME:-30-00}"                    # 30 days default (persistent serving)
JOB_NAME="${JOB_NAME:-vllm-serve}"

# vLLM parameters (passed through to run_vllm.sh)
MODEL="${MODEL:-google/gemma-3-12b-it}"
PORT="${PORT:-8421}"
TP="${TP:-1}"
DP="${DP:-8}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition|-p) PARTITION="$2"; shift 2 ;;
        --node|-w)      NODE="$2"; shift 2 ;;
        --gpus|-G)      GPUS="$2"; shift 2 ;;
        --cpus|-c)      CPUS="$2"; shift 2 ;;
        --mem)          MEM="$2"; shift 2 ;;
        --time|-t)      TIME="$2"; shift 2 ;;
        --job-name|-J)  JOB_NAME="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --tp)           TP="$2"; shift 2 ;;
        --dp)           DP="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash submit_vllm.sh [--partition P] [--node N] [--gpus G] [--tp T] [--dp D] [--time HH:MM:SS]"
            exit 1
            ;;
    esac
done

# Build node constraint if specified
NODE_FLAG=""
if [ -n "$NODE" ]; then
    NODE_FLAG="#SBATCH -w $NODE"
fi

# Exclude dll-4gpu5 only on AMD partition (it only has 4 GPUs, not enough for DP=8)
EXCLUDE_FLAG=""
if [[ "$PARTITION" == "gpu-amd" ]]; then
    EXCLUDE_FLAG="#SBATCH -x dll-4gpu5"
fi

# ── Generate sbatch script ────────────────────────────────────
SBATCH_SCRIPT=$(mktemp /tmp/vllm_sbatch_XXXXXX.sh)

cat > "$SBATCH_SCRIPT" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p ${PARTITION}
#SBATCH -G ${GPUS}
#SBATCH -c ${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
${EXCLUDE_FLAG}
#SBATCH -o ${SCRIPT_DIR}/logs/vllm_%j.out
#SBATCH -e ${SCRIPT_DIR}/logs/vllm_%j.err
${NODE_FLAG}

# Run the vLLM server
bash ${SCRIPT_DIR}/run_vllm.sh \\
    --model ${MODEL} \\
    --port ${PORT} \\
    --tp ${TP} \\
    --dp ${DP}
SBATCH_EOF

# ── Submit ─────────────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/logs"

echo "======================================================"
echo "  Submitting vLLM Server Job"
echo "  Partition: $PARTITION"
echo "  Node:      ${NODE:-'(any)'}"
echo "  GPUs:      $GPUS"
echo "  CPUs:      $CPUS"
echo "  Memory:    $MEM"
echo "  Time:      $TIME"
echo "  Model:     $MODEL"
echo "  TP×DP:     ${TP}×${DP}"
echo "  Port:      $PORT"
echo "======================================================"
echo ""

JOBID=$(sbatch --parsable "$SBATCH_SCRIPT")
echo "Submitted job: $JOBID"
echo "  Logs: ${SCRIPT_DIR}/logs/vllm_${JOBID}.out"
echo "        ${SCRIPT_DIR}/logs/vllm_${JOBID}.err"
echo ""
echo "Monitor:"
echo "  squeue -j $JOBID"
echo "  tail -f ${SCRIPT_DIR}/logs/vllm_${JOBID}.out"
echo ""
echo "Once running, find the node:"
echo "  squeue -j $JOBID -o '%N'"
echo ""
echo "Test:"
echo "  python3 ${SCRIPT_DIR}/test_endpoint.py --mode health --url http://\$(squeue -j $JOBID -o '%N' --noheader):${PORT}"
echo ""
echo "Cancel:"
echo "  scancel $JOBID"

# Clean up temp script
rm -f "$SBATCH_SCRIPT"
