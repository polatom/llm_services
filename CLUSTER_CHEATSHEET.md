# UFAL Cluster Cheatsheet

Quick reference for the UFAL Linguistic Research Cluster (LRC).
For vLLM-specific deployment, see `cluster/README.md`.

---

## Connecting

```bash
ssh lrc1.ufal.hide.ms.mff.cuni.cz    # login/submit node 1
ssh lrc2.ufal.hide.ms.mff.cuni.cz    # login/submit node 2
```

**Never run heavy computation on login nodes.** Use `srun` or `sbatch`.

---

## GPU Hardware Inventory

Each machine has **system RAM** (shared by CPU tasks) and **GPU memory** (per GPU card,
used to hold model weights). For LLM serving, the bottleneck is GPU memory — the model
must fit entirely in GPU memory. System RAM is used for data loading and tokenization.

### AMD GPUs (gpu-amd partition) — less contested

| Node | GPUs | GPU Model | GPU Mem (each) | System RAM | Notes |
|---|---|---|---|---|---|
| `tdll-8gpu5` | 8× | MI210 | 64 GB | ~512 GB | Best for LLM serving (8 GPUs) |
| `tdll-8gpu6` | 8× | MI210 | 64 GB | ~512 GB | Best for LLM serving (8 GPUs) |
| `tdll-8gpu7` | 8× | MI210 | 64 GB | ~512 GB | Best for LLM serving (8 GPUs) |
| `dll-4gpu5` | 4× | MI210 | 64 GB | ~256 GB | Only 4 GPUs — excluded by `-x` in srun |

**Total: 28 AMD GPUs.** Uses ROCm stack (AMD's CUDA equivalent).

### NVIDIA GPUs (gpu-troja partition) — often busy

| Node | GPUs | GPU Model | GPU Mem (each) | System RAM | Notes |
|---|---|---|---|---|---|
| `tdll-4gpu1` | 4× | H100 | 95 GB | ~1 TB | Fastest. Very contested. |
| `tdll-8gpu1` | 8× | A100 | 40 GB | ~512 GB | Good for LLMs at TP=2 |
| `tdll-8gpu2` | 8× | A100 | 40 GB | ~512 GB | Good for LLMs at TP=2 |
| `tdll-3gpu[1-4]` | 3× | A40 | 48 GB | ~256 GB | 4 nodes, 12 GPUs total |
| `tdll-8gpu[3-4]` | 8× | Quadro P5000 | 16 GB | ~256 GB | Too small for LLMs |

### NVIDIA GPUs (gpu-ms partition) — often busy

| Node | GPUs | GPU Model | GPU Mem (each) | System RAM | Notes |
|---|---|---|---|---|---|
| `dll-3gpu[1-5]` | 3× | A40 | 48 GB | ~256 GB | 5 nodes, 15 GPUs total |
| `dll-4gpu3` | 4× | L40 | 48 GB | ~256 GB | Newer than A40 |
| `dll-4gpu4` | 4× | A40 | 48 GB | ~256 GB | |
| `dll-8gpu[1-2]` | 8× | A30 | 24 GB | ~512 GB | Too small for 27B models |
| `dll-8gpu4` | 8× | RTX A4000 | 16 GB | ~256 GB | Too small for LLMs |
| `dll-8gpu5` | 8× | Quadro RTX 5000 | 16 GB | ~256 GB | Too small for LLMs |
| `dll-10gpu2` | 10× | GeForce GTX 1080 Ti | 11 GB | ~256 GB | Too small for LLMs |

### Which GPU for which model?

| Model | Size (bf16) | Min GPU Mem | Example configs |
|---|---|---|---|
| Gemma 3 27B | ~54 GB | 2× 40GB+ | H100 TP=1, A100 TP=2, MI210 TP=2, A40 TP=2 |
| Gemma 3 12B | ~24 GB | 1× 48GB+ | Any 48GB+ GPU at TP=1, MI210 TP=1 |
| 8B models | ~16 GB | 1× 24GB+ | Any 24GB+ GPU at TP=1 |
| 70B models | ~140 GB | 4× 40GB+ | H100 TP=2, A100 TP=4, MI210 TP=4 |

---

## Checking Hardware & Queue

### See all GPU partitions and node states

```bash
sinfo -o "%20P %10G %15N %5D %10C %10m %5t" -p gpu-troja,gpu-ms,gpu-amd
```

**Output fields:**
- **PARTITION** — job queue name (`gpu-troja`, `gpu-ms`, `gpu-amd`)
- **GRES** — generic resources (GPU type and count)
- **NODELIST** — node hostnames
- **NODES** — number of nodes matching this line
- **CPUS(A/I/O/T)** — CPUs: Allocated/Idle/Other/Total
- **MEMORY** — total RAM in MB
- **STATE** — `idle` (free), `mix` (partially used), `alloc` (full), `down`

### Detailed node info (GPU type, VRAM, features)

```bash
sinfo -N -o "%20N %10P %20G %10m %20f" | grep -i gpu
```

**Output fields:**
- **NODELIST** — node hostname
- **PARTITION** — partition name
- **GRES** — GPU model and count, e.g. `gpu:nvidia_h100:4(S:0)`
- **MEMORY** — total system RAM in MB
- **AVAIL_FEATURES** — tags like `gpuram48G`, `gpu_cc8.6` (compute capability)

### What's idle right now?

```bash
# Show only idle or partially-used nodes
sinfo -p gpu-troja,gpu-ms,gpu-amd -t idle,mix -o "%20N %10P %10G %10t %10f"
```

### Check the job queue

```bash
# All jobs in a partition
squeue -p gpu-amd -o "%.8i %.10u %.20j %.3t %.10M %.5D %.4C %.6b %R"

# Just your jobs
squeue -u $USER

# Jobs on specific nodes
squeue -w tdll-8gpu5,tdll-8gpu6,tdll-8gpu7 -o "%.8i %.10u %.20j %.3t %.10M %.4C %.6b"

# Count pending jobs in a partition
squeue -p gpu-amd -t PENDING --noheader | wc -l
```

**Queue output fields:**
- **JOBID** — unique job identifier (use with `scancel`, `scontrol`)
- **USER** — who submitted the job
- **NAME** — job name (set by `-J` flag or script `#SBATCH -J`)
- **ST** — state: `R` (running), `PD` (pending/queued), `CG` (completing)
- **TIME** — elapsed wall time (e.g. `1-23:07:46` = 1 day 23h 7m 46s)
- **NODES** — number of nodes used
- **CPUS** — CPUs allocated
- **TRES_PER_NODE** — trackable resources (GPUs) — shows `gres:gpu:N` or `N/A`
- **NODELIST(REASON)** — which node, or why pending: `(Resources)`, `(Priority)`

### Pending reasons

| Reason | Meaning |
|---|---|
| `(Resources)` | Not enough free GPUs/CPUs/RAM on any node |
| `(Priority)` | Resources available but lower-priority jobs are ahead |
| `(ReqNodeNotAvail)` | Requested node is down or reserved |
| `(QOSMaxGRESPerUser)` | You hit the per-user GPU limit |

### Detailed node info

```bash
# Full details for a specific node
scontrol show node tdll-8gpu5

# GPU-relevant fields across all nodes
scontrol show node | grep -E "NodeName|Gres=|Features|RealMemory|AllocTRES" | head -100
```

**Key fields from `scontrol show node`:**
- **Gres** — total GPUs on the node
- **GresUsed** — how many GPUs are currently allocated
- **AllocTRES** — what's allocated (CPUs, memory, GPUs)
- **State** — `IDLE`, `MIXED`, `ALLOCATED`, `DOWN`

---

## Getting a GPU Node

### Interactive (for testing)

```bash
# AMD MI210 — 8 GPUs, all on one node
srun -p gpu-amd -c 16 -G 8 --mem=64G -x dll-4gpu5 -t 8:00:00 --pty bash

# AMD MI210 — specific node
srun -p gpu-amd -w tdll-8gpu5 -c 16 -G 8 --mem=64G -t 8:00:00 --pty bash

# NVIDIA H100 — 1 GPU (95GB, enough for 27B)
srun -p gpu-troja -w tdll-4gpu1 -c 4 -G 1 --mem=16G -t 4:00:00 --pty bash

# NVIDIA A40 — 2 GPUs (48GB each)
srun -p gpu-ms --constraint=gpuram48G -c 8 -G 2 --mem=32G -t 4:00:00 --pty bash
```

**`srun` flags:**
- `-p <partition>` — which partition
- `-w <node>` — request a specific node
- `-x <node>` — exclude a node
- `-c <N>` — number of CPU cores
- `-G <N>` — number of GPUs
- `--mem=<N>G` — system RAM
- `-t <time>` — max wall time (`HH:MM:SS` or `D-HH:MM`)
- `--constraint=<feature>` — require a feature (e.g. `gpuram48G`)
- `--pty bash` — interactive shell

### Non-interactive (sbatch)

```bash
# Submit and forget — see cluster/submit_vllm.sh for vLLM-specific wrapper
sbatch my_script.sh

# Check status
squeue -u $USER

# Cancel
scancel <jobid>

# View output
tail -f slurm-<jobid>.out
```

---

## Job Management

```bash
squeue -u $USER                     # your running/pending jobs
scancel <jobid>                     # cancel a job
scancel -u $USER                    # cancel ALL your jobs (careful!)
scontrol show job <jobid>           # detailed job info
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,MaxVMSize  # resource usage
```

---

## Storage

| Path | Quota | Shared across nodes? | Use for |
|---|---|---|---|
| `$HOME` (~) | **Small (~5 GB)** | Yes (NFS) | Config, dotfiles only |
| `/lnet/work/people/$USER/` | ~50 GB | Yes (NFS) | Model weights, venvs, pip cache, repos |
| `/tmp/` on GPU node | Large | No (local) | Temp build files (lost on job end) |

### Check your quota

```bash
# NFS quota (shows $HOME and /lnet/work)
quota -s
# Fields: blocks = used, quota = soft limit, limit = hard limit
# If 'blocks' is near 'limit', you're out of space

# If quota command isn't available, use df:
df -h ~                             # $HOME filesystem usage
df -h /lnet/work/people/$USER/      # /lnet/work filesystem usage
```

### Check what's using space

```bash
# $HOME — should be < 5 GB total
du -sh ~ 2>/dev/null                          # total $HOME usage
du -sh ~/.cache ~/.local ~/.pip ~/.*cache* 2>/dev/null | sort -rh | head -10

# /lnet/work — should be < 50 GB total
du -sh /lnet/work/people/$USER/               # total work usage
du -sh /lnet/work/people/$USER/*/ 2>/dev/null | sort -rh | head -10
du -sh /lnet/work/people/$USER/.cache/huggingface/   # model weights (biggest)
du -sh /lnet/work/people/$USER/.venvs/*/      # venvs

# Find biggest files anywhere under your directories
find ~ -maxdepth 4 -type f -size +50M 2>/dev/null | head -20
find /lnet/work/people/$USER/ -maxdepth 4 -type f -size +100M 2>/dev/null | head -20
```

### Common $HOME space hogs (safe to delete)

```bash
# pip cache (often lands in $HOME by default!)
rm -rf ~/.cache/pip
rm -rf ~/.local/share/pip

# HuggingFace cache (should be on /lnet/work, not $HOME)
rm -rf ~/.cache/huggingface

# Python package installs that ended up in $HOME
rm -rf ~/.local/lib/python*/site-packages

# Failed pip builds / temp files
rm -rf /tmp/pip-* ~/.tmp/pip-*

# Old Python bytecode
find ~ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Git garbage collection (reduce .git bloat)
cd /lnet/work/people/$USER/llm_services && git gc --aggressive
```

### Prevent future quota issues (add to ~/.bashrc)

```bash
# Redirect everything large to /lnet/work
export TMPDIR=/lnet/work/people/$USER/.tmp
export PIP_CACHE_DIR=/lnet/work/people/$USER/.cache/pip
export HF_HOME=/lnet/work/people/$USER/.cache/huggingface
export XDG_CACHE_HOME=/lnet/work/people/$USER/.cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"
```

---

## Quick Checks

```bash
# Am I on a GPU node?
hostname                           # should be tdll-* or dll-*, not lrc*
echo $SLURM_STEP_GPUS             # should show GPU IDs

# NVIDIA GPU status
nvidia-smi                         # GPU utilization, memory, temperature

# AMD GPU status
rocm-smi                           # GPU utilization, memory, temperature
rocm-smi --showid --showtemp --showuse

# What Python am I using?
which python3 && python3 --version

# What PyTorch backend?
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('ROCm:', hasattr(torch.version, 'hip') and torch.version.hip is not None)"
```
