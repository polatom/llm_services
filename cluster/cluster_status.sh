#!/bin/bash
# ─────────────────────────────────────────────────────────────
# cluster_status.sh — Show GPU node availability at a glance
#
# Usage:  bash cluster/cluster_status.sh
#         bash cluster/cluster_status.sh --free    # only show nodes with free GPUs
# ─────────────────────────────────────────────────────────────

ONLY_FREE=false
if [[ "${1:-}" == "--free" ]]; then
    ONLY_FREE=true
fi

# Colors (if terminal supports it)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; DIM=''; RESET=''
fi

printf "\n${BOLD}GPU Cluster Status${RESET}  ($(date '+%Y-%m-%d %H:%M'))\n"
printf "══════════════════════════════════════════════════════════════════════════════════════════════\n"
printf "${BOLD}%-16s %-7s %-18s %6s %10s %8s  %-7s  %s${RESET}\n" \
    "Node" "Vendor" "GPU Model" "GPUMem" "GPUs" "State" "Usable?" "Jobs / Notes"
printf "──────────────────────────────────────────────────────────────────────────────────────────────\n"

# Gather node info from scontrol, parse into table
# We query all GPU partitions
scontrol show node -o 2>/dev/null | while IFS= read -r line; do
    node=$(echo "$line" | grep -oP 'NodeName=\K\S+')
    state=$(echo "$line" | grep -oP 'State=\K\S+')
    # Skip non-GPU nodes and DOWN nodes
    gres=$(echo "$line" | grep -oP 'Gres=\K\S+')
    [[ -z "$gres" || "$gres" == "(null)" ]] && continue

    partitions=$(echo "$line" | grep -oP 'Partitions=\K\S+')
    # Only GPU partitions
    echo "$partitions" | grep -qE 'gpu-amd|gpu-troja|gpu-ms' || continue

    # Parse GPU info
    # Gres format: gpu:type_name:COUNT(S:0-1)  e.g. gpu:amd_mi210:8(S:0-1)
    # GresUsed:    gpu:type_name:COUNT(IDX:...)  e.g. gpu:amd_mi210:3(IDX:0-2)
    gres_used=$(echo "$line" | grep -oP 'GresUsed=\K\S+')
    total_gpus=$(echo "$gres" | grep -oP 'gpu:[^:]+:\K[0-9]+' | head -1)
    used_gpus=$(echo "$gres_used" | grep -oP 'gpu:[^:]+:\K[0-9]+' | head -1)
    [ -z "$total_gpus" ] && total_gpus=0
    [ -z "$used_gpus" ] && used_gpus=0
    free_gpus=$((total_gpus - used_gpus))

    # Determine vendor and GPU model from partition and features
    features=$(echo "$line" | grep -oP 'AvailableFeatures=\K\S+')
    if echo "$partitions" | grep -q 'gpu-amd'; then
        vendor="AMD"
        gpu_model="MI210"
    else
        vendor="NVIDIA"
        # Guess model from features and node name
        if echo "$features" | grep -q 'gpuram95G'; then
            gpu_model="H100"
        elif echo "$features" | grep -q 'gpuram48G'; then
            # Distinguish A40 vs L40
            if [[ "$node" == "dll-4gpu3" ]]; then
                gpu_model="L40"
            else
                gpu_model="A40"
            fi
        elif echo "$features" | grep -q 'gpuram40G'; then
            gpu_model="A100 40GB"
        elif echo "$features" | grep -q 'gpuram24G'; then
            gpu_model="A30"
        elif echo "$features" | grep -q 'gpuram16G'; then
            if echo "$gres" | grep -q 'quadro'; then
                gpu_model="Quadro P5000"
            else
                gpu_model="RTX A4000"
            fi
        elif echo "$features" | grep -q 'gpuram11G'; then
            gpu_model="GTX 1080 Ti"
        else
            gpu_model="Unknown"
        fi
    fi

    # GPU memory from features
    gpu_mem=$(echo "$features" | grep -oP 'gpuram\K[0-9]+G' || echo "?")

    # State display
    clean_state=$(echo "$state" | sed 's/\*//g')  # remove * suffix
    case "$clean_state" in
        IDLE)       state_color="${GREEN}IDLE${RESET}" ;;
        MIXED)      state_color="${YELLOW}MIXED${RESET}" ;;
        ALLOCATED)  state_color="${RED}FULL${RESET}" ;;
        DOWN*|DRAIN*) state_color="${DIM}DOWN${RESET}" ;;
        *)          state_color="${DIM}${clean_state}${RESET}" ;;
    esac

    # GPU usage bar
    if [ "$total_gpus" -gt 0 ]; then
        gpu_str="${used_gpus}/${total_gpus}"
        if [ "$free_gpus" -gt 0 ]; then
            gpu_str="${GREEN}${free_gpus} free${RESET} (${used_gpus}/${total_gpus})"
        else
            gpu_str="${RED}0 free${RESET} (${used_gpus}/${total_gpus})"
        fi
    else
        gpu_str="N/A"
    fi

    # Usable for LLM serving? (need >= 24GB GPU mem)
    mem_num=$(echo "$gpu_mem" | grep -oP '[0-9]+' || echo "0")
    if [ "$mem_num" -ge 40 ]; then
        usable="${GREEN}YES${RESET}"
    elif [ "$mem_num" -ge 24 ]; then
        usable="${YELLOW}maybe${RESET}"
    else
        usable="${RED}no${RESET}"
    fi

    # Jobs on this node
    jobs=$(squeue -w "$node" --noheader -o "%u/%j" 2>/dev/null | head -3 | tr '\n' ' ')
    [ -z "$jobs" ] && jobs="-"

    # Skip if --free and no free GPUs
    if $ONLY_FREE && [ "$free_gpus" -eq 0 ]; then
        continue
    fi
    # Skip DOWN nodes unless they're just draining
    if echo "$clean_state" | grep -qiE '^DOWN'; then
        continue
    fi

    printf "%-16s ${CYAN}%-7s${RESET} %-18s %4sGB  " "$node" "$vendor" "$gpu_model" "$mem_num"
    printf "%-22b" "$gpu_str"
    printf " %-16b" "$state_color"
    printf " %-14b" "$usable"
    printf " %s" "$jobs"
    printf "\n"
done

# Summary: queue depth
printf "\n──────────────────────────────────────────────────────────────────────────────────────────────\n"
printf "${BOLD}Queue summary:${RESET}\n"
for part in gpu-amd gpu-troja gpu-ms; do
    running=$(squeue -p "$part" -t RUNNING --noheader 2>/dev/null | wc -l)
    pending=$(squeue -p "$part" -t PENDING --noheader 2>/dev/null | wc -l)
    printf "  %-12s  %d running, %d pending\n" "$part" "$running" "$pending"
done

# Your jobs
your_jobs=$(squeue -u "$USER" --noheader 2>/dev/null | wc -l)
if [ "$your_jobs" -gt 0 ]; then
    printf "\n${BOLD}Your jobs:${RESET}\n"
    squeue -u "$USER" 2>/dev/null | sed 's/^/  /'
fi

printf "\n${DIM}Tip: bash cluster/cluster_status.sh --free  (show only nodes with free GPUs)${RESET}\n\n"
