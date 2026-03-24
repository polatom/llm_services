# LLM Deployment on the UFAL Cluster

Self-hosted LLM inference for SPRINT and PONK module 3, running on the UFAL
compute cluster with AMD Instinct MI210 GPUs.

---

## Goals & Status

We're deploying a self-hosted LLM so that our apps (SPRINT, PONK module 3)
can evaluate Czech legal/academic texts without relying on external APIs
(OpenAI, etc.) — no per-token cost, no external dependency.

### Three goals

1. **Deploy Gemma 3 27B** on the cluster with `vllm serve`
2. **Make it available** on an endpoint apps can reach
3. **Test it** with `test_endpoint.py` (SPRINT-like prompts)

### Current status

- [x] Cluster access (ssh lrc1.ufal.hide.ms.mff.cuni.cz)
- [x] Deployment scripts ready (run_vllm.sh, setup_env.sh, connect.sh)
- [x] AMD MI210 / ROCm environment tested
- [x] test_endpoint.py created
- [ ] **Deploy Gemma 3 27B** — start vLLM, verify with test_endpoint.py
- [ ] **App endpoint** — confirm SPRINT/PONK VMs can reach `<gpu-node>:8421` directly
- [ ] PONK throughput test — 40k-char document in ≤30s (via chunk splitting)
- [ ] Production integration (SPRINT/PONK .env pointing at the endpoint)

---

## Architecture

### How it works

Since SPRINT and PONK are on UFAL VMs (inside the UFAL network), they can
reach the cluster GPU nodes directly — no SSH tunnel or reverse proxy needed.

```
┌──────────────┐                           ┌─────────────────────────┐
│  SPRINT /    │     HTTP (UFAL network)   │  UFAL cluster GPU node  │
│  PONK app    │ ────────────────────────►  │  vLLM + Gemma 27B       │
│  (UFAL VM)   │   <gpu-node>:8421         │  8× AMD MI210 (64GB ea) │
└──────────────┘                           └─────────────────────────┘

┌──────────────┐     SSH tunnel            ┌─────────────────────────┐
│  developer   │ ──────────────────────►   │  same GPU node          │
│  (laptop)    │  lrc1 → <gpu-node>:8421   │  for testing only       │
└──────────────┘                           └─────────────────────────┘
```

**What's what:**

- **vLLM** — a high-performance LLM serving engine. It loads a model into
  GPU memory and exposes an OpenAI-compatible REST API (`/v1/chat/completions`).
  Our apps already speak this protocol, so no code changes are needed — just
  point them at the new endpoint.

- **Gemma 3 27B IT** — a 27-billion parameter multilingual model by Google
  with good Czech support. "IT" means instruction-tuned (it follows prompts).
  At ~54 GB, it needs 2 GPUs per copy. With 8 GPUs, we run 4 parallel copies.

- **AMD MI210** — the cluster's GPU hardware. Each MI210 has 64 GB of
  HBM2e memory. They use **ROCm** (AMD's equivalent of NVIDIA's CUDA).

- **UFAL network** — both the app VMs and the cluster are on the UFAL
  network, so the apps can call `http://<gpu-node>:8421/v1/...` directly.
  An SSH tunnel is only needed for developer laptops outside the network.

### Why self-hosted?

| | External API (GPT-4o) | Self-hosted (vLLM) |
|---|---|---|
| Cost | Per-token billing | Free (UFAL hardware) |
| Privacy | Data leaves UFAL | Data stays on premises |
| Availability | Depends on OpenAI | Depends on cluster uptime |
| Speed | ~2-5s per request | ~3-8s (depends on model/GPU) |

---

## Quick Start

If you've done this before and just need the commands:

```bash
# 1. Get a GPU node
ssh lrc1.ufal.hide.ms.mff.cuni.cz
srun -p gpu-amd -c 16 -G 8 --mem=64G -x dll-4gpu5 -t 30-00 --pty bash

# 2. Start vLLM (on the GPU node)
cd /lnet/work/people/$USER/llm_services/cluster
bash run_vllm.sh
# Wait for: "Uvicorn running on http://0.0.0.0:8421"

# 3. Test it (second terminal on same GPU node, or from lrc1)
python3 test_endpoint.py --url http://<gpu-node>:8421 --mode single

# 4. SSH tunnel (from your local machine or app VM)
ssh -N -L 8421:<gpu-node>:8421 lrc1.ufal.hide.ms.mff.cuni.cz
# Now http://localhost:8421/v1/chat/completions is available
```

---

## Detailed Deployment Guide

### Step 1: SSH into the cluster login node

```bash
ssh lrc1.ufal.hide.ms.mff.cuni.cz
```

This is a **login/submit node** — no heavy computation here.
It's used to submit jobs to GPU nodes via Slurm (the cluster's job scheduler).

### Step 2: Request an interactive GPU job

```bash
srun -p gpu-amd -c 16 -G 8 --mem=64G -x dll-4gpu5 -t 30-00 --pty bash
```

What each flag means:
- `-p gpu-amd` — use the AMD GPU partition
- `-c 16` — 16 CPU cores (for data loading / tokenization)
- `-G 8` — 8 GPUs (all MI210s on a single node)
- `--mem=64G` — 64 GB system RAM
- `-x dll-4gpu5` — exclude this node (only has 4 GPUs, not 8)
- `-t 30-00` — max wall time: 30 days (job killed after this)
- `--pty bash` — interactive bash shell on the GPU node

You'll get a prompt on a GPU node (e.g. `tdll-8gpu5`). Remember this
hostname — you'll need it for the SSH tunnel.

> **Fewer GPUs:** For smaller models, change `-G 8` to `-G 2` or `-G 4`.
> Adjust the vLLM parallelism arguments accordingly (see Step 4).

### Step 3: Set up the environment (one-time)

We need Python + vLLM compiled for ROCm (AMD's GPU stack).

#### Option A: Use Hrabal's pre-compiled env (recommended)

Miroslav Hrabal maintains a vLLM environment compiled for ROCm 6.4.1.
The `run_vllm.sh` script uses this by default — nothing to install.

```bash
# To verify manually:
source /lnet/work/home-students-external/hrabal/uv_venv/3.13_vllm0.13_rocm6.4.1/bin/activate
python3 -c "import vllm; print(vllm.__version__)"  # should print 0.13.x
```

#### Option B: Your own environment

For independence from Hrabal's env (e.g. to pin a specific vLLM version):

```bash
cd /lnet/work/people/$USER/llm_services/cluster
bash setup_env.sh
# Then launch with: bash run_vllm.sh --env /lnet/work/people/$USER/.venvs/sprint-vllm-rocm
```

See `setup_env.sh` for full details (PyTorch ROCm wheels, building from source, etc.).

### Step 4: Start the vLLM server

```bash
cd /lnet/work/people/$USER/llm_services/cluster
bash run_vllm.sh
```

This runs `vllm serve` with these defaults (override via `--flags`):

| Parameter | Default | What it does |
|---|---|---|
| `--model` | `google/gemma-3-27b-it` | HuggingFace model ID |
| `--tp` | `2` | Tensor parallelism — GPUs per replica. 27B needs 2 GPUs. |
| `--dp` | `4` | Data parallelism — number of replicas. 4 = 4 concurrent request streams. |
| `--port` | `8421` | HTTP port. Non-default to avoid conflicts. |
| `--max-len` | `16000` | Max tokens (prompt + response). Our prompts are ~1-3K tokens. |
| `--max-seqs` | `256` | Max concurrent sequences across all replicas. |

**GPU math:** TP × DP = 2 × 4 = 8 GPUs total.

Examples:
```bash
bash run_vllm.sh                                         # Gemma 27B, 8 GPUs (default)
bash run_vllm.sh --model swiss-ai/Apertus-8B-Instruct-2509 --tp 1 --dp 4   # 8B model, 4 GPUs
bash run_vllm.sh --port 8422                             # different port
```

> **First run** downloads model weights from HuggingFace (~54 GB for 27B).
> Cached afterwards in `$HF_HOME` (default: `/lnet/work/people/$USER/.cache/huggingface`).

Wait for this line:
```
INFO:     Uvicorn running on http://0.0.0.0:8421
```

### Step 5: Test the endpoint

From a second terminal on the GPU node (or from `lrc1`):

```bash
# Quick health check
python3 test_endpoint.py --url http://localhost:8421 --mode health

# Send a single SPRINT-like evaluation (1 rule, 5 sentences)
python3 test_endpoint.py --url http://localhost:8421 --mode single

# Concurrent load test (10 parallel requests)
python3 test_endpoint.py --url http://localhost:8421 --mode concurrent --requests 10

# Full SPRINT simulation (6 rules × all sentence batches)
python3 test_endpoint.py --url http://localhost:8421 --mode sprint
```

You can also use curl:
```bash
curl -s http://localhost:8421/v1/models | python3 -m json.tool

curl -s http://localhost:8421/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "messages": [{"role": "user", "content": "Řekni ahoj česky."}],
    "temperature": 0,
    "max_tokens": 100
  }' | python3 -m json.tool
```

---

## Connecting Apps to the Endpoint

Both SPRINT and PONK module 3 run on UFAL VMs inside the UFAL network.
They can reach the GPU node directly — no tunnel or proxy needed.

### From UFAL VMs (SPRINT, PONK — production)

Just point the app at the GPU node hostname:

```env
# In the app's .env:
AIUFAL_ENDPOINT=http://<gpu-node>:8421/v1/chat/completions
AIUFAL_MODEL=google/gemma-3-27b-it
AIUFAL_API_KEY=dummy
```

Replace `<gpu-node>` with the actual hostname (e.g. `tdll-8gpu5`).

> **Caveat:** If the Slurm job restarts on a different node, the hostname
> changes. For production stability, ask IT for a DNS alias or use a
> long-running job (`-t 30-00`).

### From your laptop (development/testing)

Your laptop is outside the cluster network, so you need an SSH tunnel:

```bash
ssh -N -L 8421:<gpu-node>:8421 lrc1.ufal.hide.ms.mff.cuni.cz
```

Or use the helper script:
```bash
bash connect.sh <gpu-node> 8421
```

Then `http://localhost:8421/v1/...` works locally.

---

## test_endpoint.py Reference

Standalone test script — zero pip dependencies (uses Python stdlib only).
Sends the same prompts as SPRINT: a linguistic rule + Czech legal sentences
→ expects structured JSON back.

### Modes

| Mode | What it does |
|---|---|
| `health` | GET `/v1/models` — checks if server is alive and model is loaded |
| `single` | Sends 1 SPRINT request (1 rule, 5 sentences), shows full response + accuracy |
| `concurrent` | Sends N SPRINT requests in parallel, reports latency stats |
| `sprint` | Full SPRINT simulation: 6 rules × sentence batches |
| `ponk` | **PONK throughput test:** 40k-char document split into chunks, sent concurrently. Reports wall time vs 30s target. |

### Arguments

```
--url              vLLM base URL (default: http://localhost:8421)
--model            Model name (default: google/gemma-3-27b-it)
--api-key          API key (default: dummy — vLLM needs no auth)
--mode             Test mode: health, single, concurrent, sprint, ponk
--requests         Number of requests for concurrent mode (default: 10)
--batch-size       Sentences per request for sprint mode (default: 15)
--doc-chars        Document size for ponk mode (default: 40000)
--chunk-chars      Chunk size for ponk mode (default: 5000)
--target-seconds   Wall time target for ponk mode (default: 30)
```

### What it checks

For each response, the script verifies:
1. **Valid JSON** — the model returned parseable JSON (not markdown or prose)
2. **Correct structure** — SPRINT: `sent_id`+`violation`; PONK: `annotations` array with `start`/`end`/`label`
3. **Accuracy** — in `single` mode, compares SPRINT violations against ground truth
4. **Throughput** — in `ponk` mode, checks if wall time meets the target (default: 30s)

### PONK throughput requirement

PONK module 3 must process documents up to **40,000 characters** in ~30 seconds.
A single LLM call for 40k chars would take 60-120s (too slow), so the app will
split documents into chunks (~5k chars each) and send them concurrently.

With 8 chunks and DP=4 (4 vLLM replicas), the model processes ~4 chunks in
parallel. The `--mode ponk` test verifies this works within the time budget.

Tuning knobs if the target isn't met:
- **Smaller chunks** (`--chunk-chars 3000`) — less output per chunk, faster
- **More DP replicas** — fewer sequential rounds
- **Lower `--max-tokens`** in the prompt — cap output size
- **Simpler model** (8B) — faster but lower quality

---

## Model Options

Models tested or recommended, sized for MI210 GPUs (64 GB each):

| Model | Size | GPUs (TP) | Replicas on 8 GPUs (DP) | Notes |
|---|---|---|---|---|
| **Gemma 3 27B IT** | 27B | 2 | 4 | Good Czech, confirmed working. Default. |
| Apertus 8B | 8B | 1 | 8 | Fastest, but lower quality. |
| Apertus 70B FP8 | 70B | 4 | 2 | Highest quality, may have Triton/ROCm issues. |

> **Triton warning:** Some models fail on ROCm due to a Triton build issue.
> Gemma 3 27B works. If a model won't start, this may be why.

---

## Reference

### Useful Slurm commands

```bash
sinfo -p gpu-amd -o "%N %G %c %m %t"   # available GPUs
squeue -u $USER                          # your running jobs
scancel <jobid>                          # cancel a job
scontrol show job <jobid>                # job details
```

### Disk usage

Large files live under `/lnet/work/people/$USER/` (50 GB quota):
- **Model weights:** `~/.cache/huggingface/` — Gemma 27B is ~54 GB
- **venv:** `~/.venvs/sprint-vllm-rocm/` — ~2–5 GB
- **pip cache:** `~/.cache/pip/` — ~1–3 GB

```bash
du -sh /lnet/work/people/$USER/.cache/huggingface
du -sh /lnet/work/people/$USER/
```

### Troubleshooting

**"No GPUs visible"** — You're on the login node, not a GPU node. Run `srun` first.
```bash
echo $SLURM_STEP_GPUS    # should show GPU IDs like 0,1,2,...
rocm-smi                  # should list MI210 GPUs
```

**GPU out of memory** — Lower `--max-len` (8192), `--max-seqs` (64),
or reduce DP replicas. Add `--kv-cache-dtype fp8` for KV cache quantization.

**Model download fails** — Some models are gated (accept license on HuggingFace).
Set `export HF_TOKEN=hf_xxxxx` before starting vLLM.

**Requests fail** — Check model name matches (`curl .../v1/models`).
Check vLLM terminal for errors.

**Port in use** — Another user on the same node. Pick another: `--port 8422`.

---

## Files

| File | Description |
|---|---|
| `README.md` | This file — plan, guide, reference |
| `run_vllm.sh` | Sets up ROCm env + starts vLLM server |
| `setup_env.sh` | One-time setup: creates your own venv with vLLM for ROCm |
| `connect.sh` | SSH tunnel helper (run from your local machine) |
| `test_endpoint.py` | Tests the endpoint with SPRINT-like Czech legal prompts |
