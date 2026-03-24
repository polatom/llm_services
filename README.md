# UFAL LLM Services

Self-hosted LLM inference for UFAL applications — currently serving
**SPRINT** (Czech legal text evaluation) and **PONK module 3** (speech act annotation).

## What is this?

This repo contains everything needed to deploy, test, and manage LLM models
on the UFAL compute cluster. It's a shared service: multiple apps point at
the same vLLM endpoint.

```
┌──────────────┐
│  SPRINT app  │──┐
└──────────────┘  │    HTTP (UFAL network)    ┌─────────────────────────┐
                  ├──────────────────────────► │  UFAL cluster GPU node  │
┌──────────────┐  │   <gpu-node>:8421          │  vLLM + Gemma 3 27B     │
│  PONK app 3  │──┘                           │  8× AMD MI210 (64GB ea) │
└──────────────┘                              └─────────────────────────┘
```

## Quick start

See [`cluster/README.md`](cluster/README.md) for the full deployment guide.

```bash
# On cluster login node:
ssh lrc1.ufal.hide.ms.mff.cuni.cz
srun -p gpu-amd -c 16 -G 8 --mem=64G -x dll-4gpu5 -t 30-00 --pty bash

# On GPU node:
cd /path/to/llm_services/cluster
bash run_vllm.sh

# Test (second terminal):
python3 cluster/test_endpoint.py --url http://localhost:8421 --mode health
python3 cluster/test_endpoint.py --url http://localhost:8421 --mode ponk
```

## Repo structure

```
llm_services/
├── README.md               ← this file
├── cluster/                ← deployment scripts for UFAL LRC cluster
│   ├── README.md           ← detailed deployment guide & plan
│   ├── run_vllm.sh         ← launch vLLM server (AMD MI210 / ROCm)
│   ├── setup_env.sh        ← one-time venv setup
│   ├── connect.sh          ← SSH tunnel helper
│   └── test_endpoint.py    ← test script (SPRINT + PONK modes)
├── stress_test.py          ← full stress test suite (async, multi-user)
├── config.yaml             ← model definitions & test scenarios
├── sample_prompts.json     ← prompt templates & ground truth data
├── results/                ← stress test output files
└── docs/
    ├── LRC.md              ← UFAL cluster intro
    └── IT_REQUEST.md       ← IT infrastructure request (historical)
```

## Apps served

### SPRINT
Czech legal text evaluation — sends sentences + linguistic rules to LLM,
gets back structured JSON violation judgments. Many small concurrent requests.

### PONK module 3
Speech act annotation of Czech legal documents — sends document text (up to
40k chars) to LLM, gets back character-offset annotations. Needs chunking
for large documents to meet the 30-second target.

## Configuring apps to use the endpoint

Both apps run on UFAL VMs (inside UFAL network) and can reach the GPU node
directly:

```env
# PONK module 3 (.env)
AIUFAL_ENDPOINT=http://<gpu-node>:8421/v1/chat/completions
AIUFAL_MODEL=google/gemma-3-27b-it
AIUFAL_API_KEY=dummy

# SPRINT (.env)
LLM_PROVIDER=UFAL_TP
UFAL_TP_ENDPOINT=http://<gpu-node>:8421/v1/chat/completions
UFAL_TP_MODEL=google/gemma-3-27b-it
UFAL_TP_APIKEY=dummy
LLM_MAX_CONCURRENCY=48
LLM_BATCH_SIZE=15
```

## Stress testing

`stress_test.py` is a full async stress test suite that simulates realistic
concurrent load (multiple users, multiple rules, sentence batching):

```bash
# Requires httpx — install with: pip install httpx pyyaml
python3 stress_test.py --model gemma-27b-cluster --scenario sanity
python3 stress_test.py --model gemma-27b-cluster --scenario moderate
python3 stress_test.py --endpoint http://<gpu-node>:8421/v1/chat/completions --model apertus-8b --scenario sanity
```

Models and scenarios are defined in `config.yaml`. Results saved to `results/`.

See `docs/stress_test_results.md` for historical benchmark data (Apertus 8B,
EuroLLM 22B, GPT-4o comparison from March 2026).
