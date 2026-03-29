# LLM Deployment on the UFAL Cluster

Self-hosted LLM inference for SPRINT and PONK module 3, running on the UFAL
compute cluster. Primary target: AMD MI210 GPUs (ROCm), with NVIDIA fallback.
See also: `../CLUSTER_CHEATSHEET.md` for general cluster commands and hardware inventory.

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
- [x] Deployment scripts ready (run_vllm.sh, setup_env.sh, connect.sh, submit_vllm.sh)
- [x] AMD MI210 / ROCm environment tested
- [x] test_endpoint.py created (health, single, concurrent, sprint, ponk modes)
- [x] **Gemma 3 27B deployed on MI210** — vLLM starts, health check OK, single requests work
- [x] ROCm workarounds applied (see "Known Issues" below)
- [ ] ~~PONK throughput test~~ — **FAILED** with Gemma 27B on MI210 (all 9 chunks timed out at 300s in eager mode)
- [ ] **Next: Deploy Gemma 3 12B on MI210** — TP=1, DP=8 for ~4× throughput
- [ ] Re-run PONK throughput test with Gemma 12B
- [ ] **App endpoint** — confirm SPRINT/PONK VMs can reach `<gpu-node>:8421` directly
- [ ] Production integration (SPRINT/PONK .env pointing at the endpoint)
- [ ] **Backlog:** Get `torch.compile` working on ROCm (see "Known Issues")

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
# 1. Get a GPU node (AMD MI210 — less contested than NVIDIA)
ssh lrc1.ufal.hide.ms.mff.cuni.cz
srun -p gpu-amd -c 16 -G 8 --mem=0 -x dll-4gpu5 -t 8:00:00 --pty bash

# 2. Start vLLM (on the GPU node)
cd /lnet/work/people/$USER/llm_services/cluster
git pull

# Gemma 12B — recommended (faster, fits 1 GPU, 8 replicas)
bash run_vllm.sh --model google/gemma-3-12b-it --tp 1 --dp 8

# Or Gemma 27B — higher quality but slower (needs 2 GPUs per replica)
# bash run_vllm.sh --model google/gemma-3-27b-it --tp 2 --dp 4

# Wait for: "Uvicorn running on http://0.0.0.0:8421"

# 3. Test it (from lrc1 or another terminal)
python3 test_endpoint.py --url http://<gpu-node>:8421 --mode health
python3 test_endpoint.py --url http://<gpu-node>:8421 --mode ponk

# 4. Or submit as a background job (no interactive terminal needed)
bash submit_vllm.sh --partition gpu-amd --gpus 8 --tp 1 --dp 8 \
     --model google/gemma-3-12b-it --time 8:00:00

# 5. SSH tunnel (from your laptop, outside UFAL network)
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
srun -p gpu-amd -c 16 -G 8 --mem=0 -x dll-4gpu5 -t 30-00 --pty bash
```

What each flag means:
- `-p gpu-amd` — use the AMD GPU partition
- `-c 16` — 16 CPU cores (for data loading / tokenization)
- `-G 8` — 8 GPUs (all MI210s on a single node)
- `--mem=0` — all available system RAM (DP=8 loads 8 model copies concurrently, needs ~128 GB+)
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

**Note:** AMD MI210 is currently **blocked** by two issues (see Known Issues):
1. `torch.compile` is disabled — models run in slow eager mode
2. vLLM `0.18.1rc1.dev70` produces garbage output on ROCm even in eager mode

### Models tested on AMD MI210 (8× 64 GB, ROCm, eager mode)

| Model | Size | TP | DP | Health | Output quality | PONK result | Verdict |
|---|---|---|---|---|---|---|---|
| Gemma 3 27B IT | ~54 GB | 2 | 4 | ✅ | ❌ garbage (vLLM bug) | ❌ 300s timeout | vLLM broken on ROCm |
| Gemma 3 12B IT | ~24 GB | 1 | 8 | ✅ | ❌ garbage (vLLM bug) | ❌ all chunks bad | vLLM broken on ROCm |
| Apertus 8B | ~16 GB | 1 | 8 | ✅ | ❌ garbage (vLLM bug) | ❌ 5/32 valid | vLLM broken on ROCm |

### Detailed test results (March 2026)

**Gemma 3 27B IT** on tdll-8gpu6 (TP=2, DP=4, eager):
- Health check: ✅
- Single request: ✅ (~16 tok/s per replica)
- PONK (9 chunks, 5k chars): ❌ All chunks timed out at 300s
- Note: tested before garbage-output bug was isolated

**Apertus 8B** on tdll-8gpu6 (TP=1, DP=8, eager):
- Health check: ✅
- Aggregate throughput: ~262 tok/s across 8 replicas
- PONK (32 chunks, 1.5k chars): ❌ Only 5/32 valid JSON
  - Output was repetitive garbage (`.Qt.Qt.Qt...`) or multilingual noise
  - Root cause: vLLM 0.18.1rc1 ROCm bug (confirmed — see below)

**Gemma 3 12B IT** on tdll-8gpu6 (TP=1, DP=8, eager):
- Health check: ✅
- Simple `2+2` query: ❌ Returns `'4 Centres dinero LABELity HS gcc...'` (nonsense)
- PONK (9 chunks, 5k chars): ❌ All 9 chunks bad JSON, multilingual garbage output
- **HuggingFace transformers test (same node, same weights):** ✅ Returns `'4'` correctly
- **Conclusion:** Model weights and ROCm runtime are fine. vLLM 0.18.1rc1.dev70 is broken on ROCm.

### Recommendations

**For AMD MI210** — currently **blocked** until vLLM ROCm issue is resolved:
- The model weights are correct, ROCm runtime is functional
- vLLM 0.18.1rc1.dev70 (built from main branch) is broken on AMD ROCm
- Fix: rebuild vLLM at a known-good ROCm-tested version (e.g. v0.6.x or v0.7.x)
- See `docs/LLM_on_AMD.md` for IT consultation notes

**For NVIDIA nodes** (torch.compile works, 2-3× faster, no vLLM bug):

| Model | Size | TP | DP (8× A100) | DP (4× H100) | Gated? | PONK estimate |
|---|---|---|---|---|---|---|
| **Gemma 3 12B IT** | ~24 GB | 1 | 8 | 4 | Yes (HF token) | **~10-20s** ✓ |
| Qwen 2.5 14B Instruct | ~30 GB | 1 | 8 | 4 | No | ~15-25s (est.) |
| Gemma 3 27B IT | ~54 GB | 2 | 4 | 2 | Yes | ~30-60s (borderline) |

> **Triton warning:** Some models fail on ROCm due to a Triton build issue.
> Gemma 3 27B and 12B work. Apertus 8B works. If a model won't start, this may be why.

---

## Known Issues & Backlog

### vLLM 0.18.1rc1 produces garbage output on ROCm (CRITICAL — currently blocking all AMD use)

**Status:** Confirmed broken as of 2025-03-29. Root cause isolated.

**Symptom:** All model responses are multilingual nonsense (e.g. `'4 Centres dinero LABELity HS gcc Conexion 이론 Flame'` for `2+2`). Both `torch.compile` enabled and eager mode affected.

**Root cause:** vLLM `0.18.1rc1.dev70+gce57fd555` (built from main branch HEAD) has a broken ROCm inference path. The model weights themselves are fine — a direct HuggingFace transformers inference test on the same node with the same weights returns correct output (`4`).

**Evidence:**
- vLLM (eager mode, `TORCHDYNAMO_DISABLE=1`, `enforce_eager=True`): ❌ garbage output
- HuggingFace transformers (same GPU, same weights, same dtype): ✅ correct output
- ROCm runtime and model weights verified working

**Fix required:** Rebuild vLLM from a version with known working ROCm support. The current install was built from the main branch tip, which is a pre-release. Stable ROCm support in vLLM was documented around v0.6–v0.7. See `docs/LLM_on_AMD.md` for IT consultation details.

**Workaround:** None known — the bug is inside vLLM's ROCm attention/sampling kernels.

### `torch.compile` broken on ROCm (secondary issue)

**Status:** Patched but `torch.compile` corrupts model outputs anyway. Workaround applied.

**Problem:** PyTorch's `torch._inductor` autotuner crashes on ROCm MI210 with:
```
'KernelMetadata' object has no attribute 'cluster_dims'
```
This is an NVIDIA Hopper/Blackwell feature that leaked into shared ROCm codepaths.

**Patch applied:** `cluster/patches/fix_rocm_cluster_dims.py` — makes `cluster_dims` access safe.
However, even with the patch, `torch.compile` was found to corrupt model outputs.

**Current workaround:**
- `TORCHDYNAMO_DISABLE=1` env var (set in `run_vllm.sh`) — disables `torch.compile`
- `--enforce-eager` flag — disables CUDA graphs
- Models run in **eager mode**, which is ~2-3× slower

**Check Hrabal's env** — his venv (`/lnet/work/home-students-external/hrabal/uv_venv/3.13_vllm0.13_rocm6.4.1`) uses vLLM 0.13 which may not have the garbage-output bug and may have working ROCm support.

### ROCm workarounds applied

These are already handled by `run_vllm.sh` and `setup_env.sh`:

1. **Build vLLM from source** — PyPI wheel is CUDA-only.
   `VLLM_TARGET_DEVICE=rocm pip install --no-build-isolation .`
2. **Install amdsmi from ROCm bundle** — PyPI version incompatible.
   Copies from `/opt/rocm/share/amd_smi/`
3. **Set `TORCHDYNAMO_DISABLE=1`** — disables torch.compile globally
4. **Set `VLLM_TARGET_DEVICE=rocm`** — runtime platform detection
5. **Patch `cuda_communicator.py`** — FlashInfer/CustomAllreduce/QuickAllReduce
   are CUDA-only, wrapped in try/except. Script: `patches/fix_rocm_communicators.py`
6. **Redirect pip/temp to `/lnet/work`** — `$HOME` has ~5 GB quota, not enough
   for vLLM install. Use `PIP_CACHE_DIR` and `TMPDIR` env vars.
7. **HF_TOKEN** — needed for gated models (Gemma 27B). Store at
   `/lnet/work/people/$USER/.cache/huggingface/token`

### NVIDIA alternative (tested, deprioritized)

NVIDIA nodes (`gpu-troja`, `gpu-ms`) support `torch.compile` natively and would
be ~3-5× faster, but they're heavily contested:
- H100 (`tdll-4gpu1`): 4-day vLLM job by another user, couldn't get a slot
- A100 (`tdll-8gpu[1-2]`): 7+ concurrent jobs, fully allocated
- A40 nodes: also busy, and `pip install vllm` hits `$HOME` quota

The `run_vllm.sh` script auto-detects NVIDIA vs ROCm and works on both.
Revisit NVIDIA when queue pressure drops or for production deployment.

### Resource management strategy

Our use case (large model, occasional high-demand bursts) means we shouldn't
hold GPUs 24/7. Current approach:

- **Always set a time limit** (`-t 8:00:00`) — job auto-terminates, frees GPUs
- **Use `submit_vllm.sh`** for background jobs — no interactive terminal needed
- **Model startup is ~2-5 min** — acceptable for on-demand use
- **AMD partition is less contested** — fewer queuing delays

---

## Reference

### Useful Slurm commands

See `../CLUSTER_CHEATSHEET.md` for comprehensive cluster commands.

```bash
sinfo -p gpu-amd -o "%N %G %c %m %t"   # available AMD GPUs
squeue -u $USER                          # your running jobs
scancel <jobid>                          # cancel a job
scontrol show job <jobid>                # job details
```

### Disk usage

Large files live under `/lnet/work/people/$USER/` (~50 GB quota).
**`$HOME` has only ~5 GB** — never install packages there.

- **Model weights:** `/lnet/work/people/$USER/.cache/huggingface/` — 12B is ~24 GB, 27B is ~54 GB
- **venv (ROCm):** `/lnet/work/people/$USER/.venvs/llm-services-vllm-rocm/` — ~5 GB
- **venv (CUDA):** `/lnet/work/people/$USER/.venvs/llm-services-vllm-cuda/` — ~3 GB
- **pip cache:** `/lnet/work/people/$USER/.cache/pip/` — ~1–3 GB

```bash
du -sh /lnet/work/people/$USER/.cache/huggingface
du -sh /lnet/work/people/$USER/

# IMPORTANT: redirect temp/pip to /lnet/work to avoid $HOME quota
export TMPDIR=/lnet/work/people/$USER/.tmp
export PIP_CACHE_DIR=/lnet/work/people/$USER/.cache/pip
```

### Troubleshooting

**"No GPUs visible"** — You're on the login node, not a GPU node. Run `srun` first.
```bash
echo $SLURM_STEP_GPUS    # should show GPU IDs like 0,1,2,...
rocm-smi                  # should list MI210 GPUs (AMD)
nvidia-smi                # should list GPUs (NVIDIA)
```

**GPU out of memory** — Lower `--max-len` (8192), `--max-seqs` (64),
or reduce DP replicas. Add `--kv-cache-dtype fp8` for KV cache quantization.

**Model download fails** — Some models are gated (accept license on HuggingFace).
Set `export HF_TOKEN=hf_xxxxx` before starting vLLM.

**Disk quota exceeded** — You're writing to `$HOME`. Set `TMPDIR` and
`PIP_CACHE_DIR` to `/lnet/work/people/$USER/...` (see Disk usage above).

**Requests fail** — Check model name matches (`curl .../v1/models`).
Check vLLM terminal for errors.

**Port in use** — Another user on the same node. Pick another: `--port 8422`.

**torch.compile crash on ROCm** — See "Known Issues" above. `TORCHDYNAMO_DISABLE=1`
is already set by `run_vllm.sh`.

---

## Files

| File | Description |
|---|---|
| `README.md` | This file — deployment guide, test results, known issues |
| `run_vllm.sh` | Launches vLLM server (auto-detects NVIDIA/ROCm) |
| `submit_vllm.sh` | Submits vLLM as a background Slurm job (sbatch) |
| `setup_env.sh` | One-time setup: creates venv with vLLM for ROCm |
| `connect.sh` | SSH tunnel helper (run from your local machine) |
| `test_endpoint.py` | Tests endpoint: health, single, concurrent, sprint, ponk |
| `patches/fix_rocm_communicators.py` | Patches CUDA-only imports for ROCm |
| `../CLUSTER_CHEATSHEET.md` | General cluster commands and hardware inventory |
