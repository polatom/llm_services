# LLM Serving on AMD MI210 — Status & IT Consultation

## Summary

We are trying to self-host an LLM (Gemma 3 12B or similar) on the UFAL LRC
AMD MI210 nodes (`tdll-8gpu[5-7]`) for two applications:

- **PONK app3** — Czech legal document speech-act annotation (40k chars, ≤30s target)
- **SPRINT** — Czech legal text evaluation against linguistic rules (batch, structured JSON)

We need a dedicated endpoint because the shared UFAL endpoint
(`ai.ufal.mff.cuni.cz`) is used by other projects and becomes slow under load.

## What We've Accomplished

We set up a full vLLM serving pipeline for AMD ROCm:

- Scripts to launch vLLM with data parallelism (DP=8, 8 replicas on 8× MI210)
- Workarounds for ROCm-specific issues: amdsmi version mismatch, CUDA-only imports, disk quota management
- Patch for a `torch.compile` crash (`cluster_dims` attribute error in PyTorch Inductor)
- Slurm job submission and monitoring scripts

All code is in the `cluster/` directory of this repo.

## Current Blocker: vLLM Garbage Output on ROCm

**The model weights are fine. The ROCm runtime is fine. vLLM is broken.**

We confirmed this with a controlled test on `tdll-8gpu6`:

| Test | Tool | Result |
|---|---|---|
| `2+2 = ?` | vLLM 0.18.1rc1 serving | ❌ `'4 Centres dinero LABELity HS gcc Conexion 이론 Flame'` |
| `2+2 = ?` | HuggingFace `transformers` direct | ✅ `'4'` |

Both tests: same GPU (`tdll-8gpu6`), same model weights (`google/gemma-3-12b-it`),
same dtype (`bfloat16`), same ROCm runtime, `TORCHDYNAMO_DISABLE=1` (eager mode).

### vLLM version we built

```
vLLM 0.18.1rc1.dev70+gce57fd555
PyTorch 2.9.1+rocm6.4
ROCm 6.4.1
```

This was built from the **main branch HEAD** in late March 2026 — a pre-release.
The ROCm inference path in this version is clearly broken (corrupted attention outputs,
broken sampling, or similar low-level kernel bug).

## What We Think Would Fix It

### Option 1: Use an older stable vLLM (most likely to work)

Rebuild our venv using a vLLM release that had documented stable ROCm support.
**Hrabal's venv** (`/lnet/work/home-students-external/hrabal/uv_venv/3.13_vllm0.13_rocm6.4.1`)
uses **vLLM 0.13** — this predates the garbage-output regression and may work correctly.

Questions for IT:
- Can we reuse Hrabal's venv or install from the same pinned vLLM version?
- Which vLLM release/commit does the UFAL shared endpoint (`ai.ufal.mff.cuni.cz`) use for AMD?
  That one is known to work — we could match it.

### Option 2: AMD's ROCm-specific vLLM Docker image

AMD publishes ROCm-optimized Docker images at `rocm/vllm` on Docker Hub,
tested against specific MI300X/MI210 hardware. These are separate from the
upstream vLLM release cycle and often more stable on AMD.

If Singularity/Apptainer is available on the cluster, we could pull and run one of these.

### Option 3: SGLang instead of vLLM

[SGLang](https://github.com/sgl-project/sglang) is an alternative serving engine
with active ROCm/AMD support. It may not have the same regression.
Requires a separate venv build but has an OpenAI-compatible API endpoint.

## Secondary Issue: `torch.compile` Broken on ROCm

Even if we fix the garbage-output bug, `torch.compile` is disabled because:
1. PyTorch Inductor crashes with `'KernelMetadata' object has no attribute 'cluster_dims'`
2. Even with our patch, `torch.compile` corrupts model outputs

Without `torch.compile`, models run in **eager mode** (~2-3× slower than compiled NVIDIA).
This means even a correctly-serving Gemma 12B may be too slow for PONK's 30s wall-clock target.

If `torch.compile` / CUDA graphs can be made to work on MI210, throughput would improve
significantly and the 30s target becomes achievable.

## Hardware & Setup Reference

| Item | Value |
|---|---|
| Nodes | `tdll-8gpu[5-7]` (8× MI210 64GB each), `dll-4gpu5` (4× MI210 64GB) |
| Partition | `gpu-amd` |
| ROCm version | 6.4.1 |
| Our venv | `/lnet/work/people/tpolak/.venvs/llm-services-vllm-rocm` |
| Model weights | `/lnet/work/people/tpolak/.cache/huggingface/hub/models--google--gemma-3-12b-it` |
| Repo | `/lnet/work/people/tpolak/llm_services` (also on GitHub: `polatom/llm_services`) |
| Scripts | `cluster/run_vllm.sh`, `cluster/submit_vllm.sh`, `cluster/setup_env.sh` |

## Interim Option: NVIDIA A40 While AMD Is Blocked

While waiting for the AMD vLLM fix, we can run on NVIDIA A40 nodes which:
- Support `torch.compile` natively (2-3× faster than AMD eager mode)
- Use the existing CUDA venv (`/lnet/work/people/tpolak/.venvs/llm-services-vllm-cuda`)
- Require no additional setup — `run_vllm.sh` auto-detects NVIDIA vs ROCm

Nodes with A40 48GB (fit Gemma 12B at TP=1) that were **idle as of 2026-03-29**:
- `tdll-3gpu[2-3]` (gpu-troja) — 3× A40 each
- `dll-3gpu5` (gpu-ms) — 3× A40

**Expected performance on A40 TP=1 DP=3 with torch.compile:**
~60-80 tok/s per replica × 3 replicas = ~180-240 tok/s total.
PONK: 9 chunks × ~500 output tokens ÷ 180 tok/s ≈ **~25s** — should meet the 30s target.

**Command to try (run from login node):**
```bash
cd /lnet/work/people/tpolak/llm_services && git pull
bash cluster/submit_vllm.sh \
  --partition gpu-troja \
  --gpus 3 \
  --cpus 6 \
  --mem 64G \
  --time 2:00:00 \
  --tp 1 \
  --dp 3

# Once all 3 "Application startup complete" messages appear:
JOBID=$(squeue -u $USER --noheader -o '%i' | head -1)
NODE=$(squeue -j $JOBID -o '%N' --noheader)
python3 cluster/test_endpoint.py \
  --url http://$NODE:8421 \
  --model google/gemma-3-12b-it \
  --mode ponk
```

## Specific Questions for IT

1. **Which vLLM version/commit does the UFAL AMD endpoint use?** We want to match it.
2. **Can we use or copy Hrabal's venv** (`vllm0.13_rocm6.4.1`)? Or get the same install?
3. **Is Singularity/Apptainer available** on the GPU nodes for running ROCm Docker images?
4. **Is there a known-working vLLM + ROCm 6.4 combination** you're aware of?
5. **Can `torch.compile` be made to work on MI210?** We hit `cluster_dims` crashes and output
   corruption even with a patch. Is there a ROCm-patched PyTorch build available?
