"""
LLM Stress Test for SPRINT App — Friday Workshop Simulation

Simulates realistic concurrent load against UFAL-hosted models.
Uses the same async pattern (httpx + asyncio) as the real backend.

Usage:
    poetry run python stress_test.py --config config.yaml --scenario moderate
    poetry run python stress_test.py --model eurollm-22b --users 15 --evals 2
    poetry run python stress_test.py --model eurollm-22b --scenario sanity
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

# ── Setup ────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _load_api_key(env_var: str = "UFAL_TP_APIKEY") -> str:
    """Load API key from env var, falling back to project .env file."""
    key = os.getenv(env_var, "").strip()
    if key:
        return key
    # Try loading from project .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{env_var}=") and not line.startswith("#"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    return ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stress_test")


# ── Prompt Builder ───────────────────────────────────────────

def load_prompts() -> dict:
    with open(BASE_DIR / "sample_prompts.json", encoding="utf-8") as f:
        return json.load(f)


def build_messages(prompts: dict, rule: dict, sentence_indices: list[tuple[int, str]]) -> list[dict]:
    """Build chat messages matching the real app's prompt structure."""
    # Format examples
    examples_lines = []
    example_output = []
    idx = 1
    for s in rule.get("violation_examples", []):
        sid = f"ex{idx}"
        examples_lines.append(f'sent_id: "{sid}" | Sentence: "{s}"')
        example_output.append({"sent_id": sid, "violation": True, "reason": "...", "suggestion": "..."})
        idx += 1
    for s in rule.get("compliant_examples", []):
        sid = f"ex{idx}"
        examples_lines.append(f'sent_id: "{sid}" | Sentence: "{s}"')
        example_output.append({"sent_id": sid, "violation": False, "reason": "...", "suggestion": None})
        idx += 1

    # Format sentences to evaluate
    sent_lines = []
    for i, text in sentence_indices:
        sent_lines.append(f'sent_id: "{i}" | Sentence: "{text}"')

    # Fill payload template
    payload = prompts["payload_template"]
    payload = payload.replace("{{rule_definition}}", rule["definition"])
    payload = payload.replace("{{rule_conditions}}", rule["conditions"])
    payload = payload.replace("{{sentences_examples_input_output}}", "\n".join(examples_lines))
    payload = payload.replace("{{example_output}}", "Example output:\n" + json.dumps(example_output, ensure_ascii=False))
    payload = payload.replace("{{sentences_list}}", "\n".join(sent_lines))

    return [
        {"role": "system", "content": prompts["system_message"]},
        {"role": "user", "content": payload},
    ]


# ── Single LLM Request ──────────────────────────────────────

async def call_llm(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    api_key: str,
    model_id: str,
    messages: list[dict],
    llm_params: dict,
    request_id: str,
) -> dict:
    """
    Make a single LLM API call. Returns a result dict with timing and response info.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": llm_params.get("temperature", 0.0),
        "max_tokens": llm_params.get("max_tokens", 8192),
    }

    result = {
        "request_id": request_id,
        "model": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_chars": sum(len(m["content"]) for m in messages),
    }

    t0 = time.monotonic()
    try:
        async with semaphore:
            wait_time = time.monotonic() - t0
            result["queue_wait_seconds"] = round(wait_time, 3)

            t1 = time.monotonic()
            resp = await client.post(endpoint, json=payload, headers=headers)
            elapsed = time.monotonic() - t1
            result["status_code"] = resp.status_code
            result["latency_seconds"] = round(elapsed, 3)

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                result["output_chars"] = len(content)
                result["prompt_tokens"] = usage.get("prompt_tokens", 0)
                result["completion_tokens"] = usage.get("completion_tokens", 0)
                result["total_tokens"] = usage.get("total_tokens", 0)
                result["raw_response"] = content
                result["raw_prompt"] = json.dumps(messages, ensure_ascii=False)

                # Validate JSON output
                quality = validate_response(content, messages, result.get("_ground_truth"))
                result["quality"] = quality
                result["error"] = None
            else:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:500]}"
                result["quality"] = {"valid_json": False, "has_all_fields": False, "correct_count": False}

    except httpx.TimeoutException:
        elapsed = time.monotonic() - t0
        result["latency_seconds"] = round(elapsed, 3)
        result["error"] = "TIMEOUT"
        result["quality"] = {"valid_json": False, "has_all_fields": False, "correct_count": False}
    except Exception as e:
        elapsed = time.monotonic() - t0
        result["latency_seconds"] = round(elapsed, 3)
        result["error"] = f"{type(e).__name__}: {e}"
        result["quality"] = {"valid_json": False, "has_all_fields": False, "correct_count": False}

    return result


def _extract_json(raw: str) -> str:
    """Try to extract a JSON array from LLM output that may contain extra text."""
    raw = raw.strip()
    # Strip markdown code fences
    if "```" in raw:
        m = re.search(r'```(?:json)?\s*\n?(.*?)```', raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    # If still not starting with [, try to find the first [ ... last ]
    if not raw.startswith("["):
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]
    return raw


def validate_response(content: str, messages: list[dict], ground_truth: dict | None = None) -> dict:
    """Check if LLM output is valid JSON with expected structure and violation correctness."""
    content = _extract_json(content)

    quality = {
        "valid_json": False, "has_all_fields": False, "correct_count": False,
        "violation_accuracy": None, "violation_tp": 0, "violation_fp": 0,
        "violation_tn": 0, "violation_fn": 0, "violation_total": 0,
    }

    try:
        parsed = json.loads(content)
        quality["valid_json"] = True
    except json.JSONDecodeError:
        return quality

    if not isinstance(parsed, list):
        return quality

    # Check structure of each element
    required_keys = {"sent_id", "violation", "reason"}
    all_ok = all(
        isinstance(item, dict) and required_keys.issubset(item.keys())
        for item in parsed
    )
    quality["has_all_fields"] = all_ok

    # Count sent_ids in the user message to check we got the right number back
    user_msg = messages[-1]["content"] if messages else ""
    expected_ids = re.findall(r'sent_id: "(\d+)"', user_msg)
    quality["correct_count"] = len(parsed) == len(expected_ids)
    quality["expected_count"] = len(expected_ids)
    quality["actual_count"] = len(parsed)

    # Violation correctness check against ground truth
    if ground_truth and all_ok:
        tp = fp = tn = fn = 0
        for item in parsed:
            sid = str(item.get("sent_id", ""))
            expected_violation = ground_truth.get(sid)
            if expected_violation is None:
                continue  # no ground truth for this sent_id
            raw_v = item.get("violation", False)
            if isinstance(raw_v, str):
                actual = raw_v.lower().strip() in ("true", "yes", "1")
            else:
                actual = bool(raw_v)
            if expected_violation and actual:
                tp += 1
            elif expected_violation and not actual:
                fn += 1
            elif not expected_violation and actual:
                fp += 1
            else:
                tn += 1
        total = tp + fp + tn + fn
        quality["violation_tp"] = tp
        quality["violation_fp"] = fp
        quality["violation_tn"] = tn
        quality["violation_fn"] = fn
        quality["violation_total"] = total
        quality["violation_accuracy"] = round((tp + tn) / max(total, 1) * 100, 1)

    return quality


# ── User Simulation ──────────────────────────────────────────

async def simulate_user_evaluation(
    user_id: int,
    eval_id: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    api_key: str,
    model_id: str,
    llm_params: dict,
    prompts: dict,
    num_rules: int,
    num_sentences: int,
    batch_size: int = 0,
) -> dict:
    """
    Simulate one evaluation by one user: fire N rule calls concurrently.
    If batch_size > 0, splits sentences into chunks (mirrors app's LLM_BATCH_SIZE).
    """
    # Pick sentences from annotated set (with ground truth)
    annotated = prompts["sample_sentences_annotated"]
    selected = random.sample(annotated, min(num_sentences, len(annotated)))
    sentence_indices = [(i, s["text"]) for i, s in enumerate(selected)]

    # Pick rules (use first N from the 6 seed rules)
    rules = prompts["rules"][:num_rules]

    # Split sentences into chunks if batch_size is set
    if batch_size > 0 and len(sentence_indices) > batch_size:
        chunks = [sentence_indices[i:i + batch_size] for i in range(0, len(sentence_indices), batch_size)]
    else:
        chunks = [sentence_indices]

    num_calls = len(rules) * len(chunks)
    eval_start = time.monotonic()
    logger.info("User %d, eval %d: starting (%d rules × %d sentences, %d chunks → %d calls)",
                user_id, eval_id, len(rules), len(selected), len(chunks), num_calls)

    # Fire all rule×chunk calls concurrently (same as real app with splitting)
    tasks = []
    for rule_idx, rule in enumerate(rules):
        # Build full ground truth map for this rule
        ground_truth = {}
        for i, s in enumerate(selected):
            expected = s["violations"].get(rule["name"])
            if expected is not None:
                ground_truth[str(i)] = expected

        for chunk_idx, chunk in enumerate(chunks):
            request_id = f"u{user_id}_e{eval_id}_r{rule_idx}_c{chunk_idx}"
            messages = build_messages(prompts, rule, chunk)

            # Ground truth subset for this chunk
            chunk_truth = {str(sid): ground_truth[str(sid)] for sid, _ in chunk if str(sid) in ground_truth}

            task = _call_llm_with_truth(
                client, semaphore, endpoint, api_key, model_id,
                messages, llm_params, request_id, chunk_truth,
            )
            tasks.append(task)

    results = await asyncio.gather(*tasks)
    eval_elapsed = time.monotonic() - eval_start

    return {
        "user_id": user_id,
        "eval_id": eval_id,
        "eval_latency_seconds": round(eval_elapsed, 3),
        "num_rules": len(rules),
        "num_sentences": len(selected),
        "batch_size": batch_size if batch_size > 0 else len(sentence_indices),
        "num_chunks": len(chunks),
        "requests": results,
    }


async def _call_llm_with_truth(
    client, semaphore, endpoint, api_key, model_id,
    messages, llm_params, request_id, ground_truth,
):
    """Wrapper that calls LLM and then applies ground truth validation."""
    result = await call_llm(
        client, semaphore, endpoint, api_key, model_id,
        messages, llm_params, request_id,
    )
    # Re-validate with ground truth if we got a successful response
    if result.get("error") is None and result.get("raw_response"):
        result["quality"] = validate_response(result["raw_response"], messages, ground_truth)
    return result


async def simulate_user(
    user_id: int,
    num_evals: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    api_key: str,
    model_id: str,
    llm_params: dict,
    prompts: dict,
    num_rules: int,
    num_sentences: int,
    batch_size: int = 0,
) -> list[dict]:
    """Simulate a single user doing multiple evaluations sequentially."""
    evaluations = []
    for eval_id in range(num_evals):
        if eval_id > 0:
            # Realistic gap between evaluations (user edits, then re-evaluates)
            gap = random.uniform(5.0, 15.0)
            logger.info("User %d: waiting %.1fs before next evaluation", user_id, gap)
            await asyncio.sleep(gap)

        result = await simulate_user_evaluation(
            user_id, eval_id, client, semaphore, endpoint, api_key,
            model_id, llm_params, prompts, num_rules, num_sentences,
            batch_size=batch_size,
        )
        evaluations.append(result)
        logger.info(
            "User %d, eval %d: done in %.1fs (%d/%d requests OK)",
            user_id, eval_id, result["eval_latency_seconds"],
            sum(1 for r in result["requests"] if r["error"] is None),
            len(result["requests"]),
        )
    return evaluations


# ── Main Test Runner ─────────────────────────────────────────

async def run_test(
    model_key: str,
    model_id: str,
    model_label: str,
    config: dict,
    scenario: dict,
    prompts: dict,
) -> dict:
    """Run a complete stress test for one model + one scenario."""
    api_conf = dict(config["api"])  # copy to allow per-model overrides
    # Per-model endpoint/api_key_env override
    model_conf = config["models"].get(model_key, {})
    if model_conf.get("endpoint"):
        api_conf["endpoint"] = model_conf["endpoint"]
    api_key_env = model_conf.get("api_key_env", "UFAL_TP_APIKEY")
    api_conf["api_key"] = _load_api_key(api_key_env)
    if not api_conf["api_key"]:
        logger.error("No API key found. Set %s env var or add it to project .env", api_key_env)
        return {"summary": {"error": "No API key"}, "evaluations": []}
    llm_params = config["llm_params"]

    num_users = scenario["users"]
    num_evals = scenario["evals_per_user"]
    num_rules = scenario["rules_per_eval"]
    num_sentences = scenario["sentences_per_eval"]
    stagger = scenario["stagger_seconds"]
    max_concurrency = scenario["max_concurrency"]
    batch_size = scenario.get("batch_size", 0)

    logger.info("=" * 70)
    logger.info("MODEL: %s (%s)", model_label, model_id)
    batch_info = f", batch_size={batch_size}" if batch_size > 0 else ""
    logger.info("SCENARIO: %d users × %d evals, %d rules × %d sentences, stagger=%.1fs, concurrency=%d%s",
                num_users, num_evals, num_rules, num_sentences, stagger, max_concurrency, batch_info)
    logger.info("=" * 70)

    semaphore = asyncio.Semaphore(max_concurrency)

    async with httpx.AsyncClient(timeout=api_conf["timeout_seconds"]) as client:
        test_start = time.monotonic()

        # Launch users with staggered arrival
        user_tasks = []
        for uid in range(num_users):
            if uid > 0 and stagger > 0:
                delay = random.uniform(0, stagger)
                await asyncio.sleep(delay)

            task = asyncio.create_task(
                simulate_user(
                    uid, num_evals, client, semaphore,
                    api_conf["endpoint"], api_conf["api_key"], model_id,
                    llm_params, prompts, num_rules, num_sentences,
                    batch_size=batch_size,
                )
            )
            user_tasks.append(task)

        all_user_results = await asyncio.gather(*user_tasks)
        test_elapsed = time.monotonic() - test_start

    # Flatten results
    all_evaluations = []
    all_requests = []
    for user_evals in all_user_results:
        for ev in user_evals:
            all_evaluations.append(ev)
            all_requests.extend(ev["requests"])

    # Compute summary
    summary = compute_summary(model_key, model_id, model_label, scenario, all_evaluations, all_requests, test_elapsed)

    return {
        "summary": summary,
        "evaluations": all_evaluations,
    }


def compute_summary(
    model_key: str,
    model_id: str,
    model_label: str,
    scenario: dict,
    evaluations: list[dict],
    requests: list[dict],
    total_elapsed: float,
) -> dict:
    """Compute aggregate statistics from test results."""
    successful = [r for r in requests if r["error"] is None]
    failed = [r for r in requests if r["error"] is not None]

    latencies = [r["latency_seconds"] for r in successful]
    queue_waits = [r.get("queue_wait_seconds", 0) for r in successful]
    eval_latencies = [e["eval_latency_seconds"] for e in evaluations]

    prompt_tokens = sum(r.get("prompt_tokens", 0) for r in successful)
    completion_tokens = sum(r.get("completion_tokens", 0) for r in successful)

    valid_json_count = sum(1 for r in successful if r.get("quality", {}).get("valid_json", False))
    all_fields_count = sum(1 for r in successful if r.get("quality", {}).get("has_all_fields", False))
    correct_count = sum(1 for r in successful if r.get("quality", {}).get("correct_count", False))

    # Aggregate violation correctness across all requests
    total_tp = sum(r.get("quality", {}).get("violation_tp", 0) for r in successful)
    total_fp = sum(r.get("quality", {}).get("violation_fp", 0) for r in successful)
    total_tn = sum(r.get("quality", {}).get("violation_tn", 0) for r in successful)
    total_fn = sum(r.get("quality", {}).get("violation_fn", 0) for r in successful)
    total_judged = total_tp + total_fp + total_tn + total_fn
    violation_accuracy = round((total_tp + total_tn) / max(total_judged, 1) * 100, 1)
    precision = round(total_tp / max(total_tp + total_fp, 1) * 100, 1)
    recall = round(total_tp / max(total_tp + total_fn, 1) * 100, 1)
    f1 = round(2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1) * 100, 1)

    def safe_stats(values: list[float]) -> dict:
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "stdev": 0}
        sorted_v = sorted(values)
        p90_idx = int(len(sorted_v) * 0.9)
        p95_idx = int(len(sorted_v) * 0.95)
        return {
            "min": round(sorted_v[0], 3),
            "max": round(sorted_v[-1], 3),
            "mean": round(statistics.mean(sorted_v), 3),
            "median": round(statistics.median(sorted_v), 3),
            "p90": round(sorted_v[min(p90_idx, len(sorted_v) - 1)], 3),
            "p95": round(sorted_v[min(p95_idx, len(sorted_v) - 1)], 3),
            "stdev": round(statistics.stdev(sorted_v), 3) if len(sorted_v) > 1 else 0,
        }

    summary = {
        "model_key": model_key,
        "model_id": model_id,
        "model_label": model_label,
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "total_requests": len(requests),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "error_rate_pct": round(len(failed) / max(len(requests), 1) * 100, 1),
        "total_evaluations": len(evaluations),
        "request_latency": safe_stats(latencies),
        "queue_wait": safe_stats(queue_waits),
        "evaluation_latency": safe_stats(eval_latencies),
        "throughput_requests_per_second": round(len(successful) / max(total_elapsed, 0.001), 2),
        "tokens": {
            "total_prompt": prompt_tokens,
            "total_completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "quality": {
            "valid_json": valid_json_count,
            "valid_json_pct": round(valid_json_count / max(len(successful), 1) * 100, 1),
            "has_all_fields": all_fields_count,
            "has_all_fields_pct": round(all_fields_count / max(len(successful), 1) * 100, 1),
            "correct_count": correct_count,
            "correct_count_pct": round(correct_count / max(len(successful), 1) * 100, 1),
            "violation_accuracy_pct": violation_accuracy,
            "violation_precision_pct": precision,
            "violation_recall_pct": recall,
            "violation_f1_pct": f1,
            "violation_tp": total_tp,
            "violation_fp": total_fp,
            "violation_tn": total_tn,
            "violation_fn": total_fn,
            "violation_total_judged": total_judged,
        },
        "errors": {},
    }

    # Group errors by type
    for r in failed:
        err = r.get("error", "unknown")
        # Truncate long errors for grouping
        err_key = err[:100]
        summary["errors"][err_key] = summary["errors"].get(err_key, 0) + 1

    return summary


# ── Report Generation ────────────────────────────────────────

def print_summary(summary: dict):
    """Print a human-readable summary to console."""
    print("\n" + "=" * 70)
    print(f"  MODEL: {summary['model_label']}")
    print(f"  ({summary['model_id']})")
    print("=" * 70)

    sc = summary["scenario"]
    print(f"\n  Scenario: {sc['users']} users × {sc['evals_per_user']} evals, "
          f"{sc['rules_per_eval']} rules × {sc['sentences_per_eval']} sentences")
    print(f"  Total time: {summary['total_elapsed_seconds']:.1f}s")
    print(f"\n  Requests: {summary['successful_requests']}/{summary['total_requests']} OK "
          f"({summary['error_rate_pct']}% errors)")
    print(f"  Throughput: {summary['throughput_requests_per_second']} req/s")

    rl = summary["request_latency"]
    print(f"\n  Request latency (seconds):")
    print(f"    min={rl['min']}  median={rl['median']}  mean={rl['mean']}  "
          f"p90={rl['p90']}  p95={rl['p95']}  max={rl['max']}")

    el = summary["evaluation_latency"]
    print(f"\n  Evaluation latency (seconds) — what the user waits:")
    print(f"    min={el['min']}  median={el['median']}  mean={el['mean']}  "
          f"p90={el['p90']}  p95={el['p95']}  max={el['max']}")

    qw = summary["queue_wait"]
    print(f"\n  Queue wait (seconds) — time waiting for semaphore:")
    print(f"    min={qw['min']}  median={qw['median']}  mean={qw['mean']}  "
          f"p90={qw['p90']}  max={qw['max']}")

    tk = summary["tokens"]
    print(f"\n  Tokens: {tk['total_prompt']} prompt + {tk['total_completion']} completion = {tk['total']} total")

    q = summary["quality"]
    print(f"\n  Output quality (of {summary['successful_requests']} successful requests):")
    print(f"    Valid JSON:     {q['valid_json']}/{summary['successful_requests']} ({q['valid_json_pct']}%)")
    print(f"    All fields OK:  {q['has_all_fields']}/{summary['successful_requests']} ({q['has_all_fields_pct']}%)")
    print(f"    Correct count:  {q['correct_count']}/{summary['successful_requests']} ({q['correct_count_pct']}%)")

    if q.get("violation_total_judged", 0) > 0:
        print(f"\n  Violation correctness (vs ground truth, {q['violation_total_judged']} judgments):")
        print(f"    Accuracy:   {q['violation_accuracy_pct']}%")
        print(f"    Precision:  {q['violation_precision_pct']}%  (TP={q['violation_tp']}, FP={q['violation_fp']})")
        print(f"    Recall:     {q['violation_recall_pct']}%  (TP={q['violation_tp']}, FN={q['violation_fn']})")
        print(f"    F1:         {q['violation_f1_pct']}%")
        print(f"    TN={q['violation_tn']}  (correct non-violations)")

    if summary["errors"]:
        print(f"\n  Errors:")
        for err, count in summary["errors"].items():
            print(f"    [{count}×] {err}")

    print("\n" + "=" * 70)


def save_results(test_result: dict, model_key: str, scenario_name: str):
    """Save full results + summary + raw prompts/responses to files."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{ts}_{model_key}_{scenario_name}"

    # Extract raw prompts/responses into a separate file
    raw_exchanges = []
    for ev in test_result.get("evaluations", []):
        for req in ev.get("requests", []):
            exchange = {
                "request_id": req.get("request_id"),
                "model": req.get("model"),
                "timestamp": req.get("timestamp"),
                "latency_seconds": req.get("latency_seconds"),
                "error": req.get("error"),
                "prompt": req.get("raw_prompt"),
                "response": req.get("raw_response"),
                "quality": req.get("quality"),
            }
            raw_exchanges.append(exchange)

    raw_path = RESULTS_DIR / f"{prefix}_raw_exchanges.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for ex in raw_exchanges:
            f.write(json.dumps(ex, ensure_ascii=False, default=str) + "\n")

    # Full results (strip raw_prompt/raw_response to keep size manageable)
    slim_result = json.loads(json.dumps(test_result, default=str))
    for ev in slim_result.get("evaluations", []):
        for req in ev.get("requests", []):
            req.pop("raw_prompt", None)
            if "raw_response" in req:
                req["raw_response"] = req["raw_response"][:500] + "..." if len(req.get("raw_response", "")) > 500 else req.get("raw_response", "")

    full_path = RESULTS_DIR / f"{prefix}_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(slim_result, f, ensure_ascii=False, indent=2, default=str)

    # Summary only
    summary_path = RESULTS_DIR / f"{prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(test_result["summary"], f, ensure_ascii=False, indent=2)

    # Append to comparison log (one line per test run)
    comparison_path = RESULTS_DIR / "comparison_log.jsonl"
    with open(comparison_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(test_result["summary"], ensure_ascii=False) + "\n")

    logger.info("Results saved: %s", full_path)
    logger.info("Summary saved: %s", summary_path)
    logger.info("Raw exchanges: %s", raw_path)

    return full_path, summary_path


def print_comparison_table():
    """Print a comparison table from the comparison log."""
    comparison_path = RESULTS_DIR / "comparison_log.jsonl"
    if not comparison_path.exists():
        print("No comparison data yet.")
        return

    entries = []
    with open(comparison_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("No comparison data yet.")
        return

    print("\n" + "=" * 120)
    print("  MODEL COMPARISON")
    print("=" * 120)
    header = f"  {'Model':<30} {'Scenario':<12} {'Sents':<6} {'Reqs':<6} {'OK%':<6} {'Median':<8} {'P90':<8} {'JSON%':<7} {'Count%':<7} {'Acc%':<6} {'F1%':<6} {'Thr/s':<6}"
    print(header)
    print("-" * 140)

    for e in entries:
        sc = e.get("scenario", {})
        sc_label = f"{sc.get('users', '?')}u×{sc.get('evals_per_user', '?')}e"
        rl = e.get("request_latency", {})
        q = e.get("quality", {})
        ok_pct = round(100 - e.get("error_rate_pct", 0), 1)
        row = (
            f"  {e.get('model_label', '?'):<30} "
            f"{sc_label:<12} "
            f"{sc.get('sentences_per_eval', '?'):<6} "
            f"{e.get('total_requests', 0):<6} "
            f"{ok_pct:<6} "
            f"{rl.get('median', 0):<8} "
            f"{rl.get('p90', 0):<8} "
            f"{q.get('valid_json_pct', 0):<7} "
            f"{q.get('correct_count_pct', 0):<7} "
            f"{q.get('violation_accuracy_pct', '-'):<6} "
            f"{q.get('violation_f1_pct', '-'):<6} "
            f"{e.get('throughput_requests_per_second', 0):<6}"
        )
        print(row)

    print("=" * 140)
    print()


# ── CLI ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Stress Test for SPRINT App")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--model", help="Model key from config (e.g. eurollm-22b), or 'all' to test all")
    parser.add_argument("--scenario", help="Scenario name from config (e.g. sanity, moderate, full, peak)")
    parser.add_argument("--users", type=int, help="Override: number of users")
    parser.add_argument("--evals", type=int, help="Override: evaluations per user")
    parser.add_argument("--rules", type=int, help="Override: rules per evaluation")
    parser.add_argument("--sentences", type=int, help="Override: sentences per evaluation")
    parser.add_argument("--concurrency", type=int, help="Override: max concurrent requests")
    parser.add_argument("--stagger", type=float, help="Override: max stagger seconds between users")
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Override: max sentences per LLM call (0=no splitting)")
    parser.add_argument("--endpoint", help="Override API endpoint URL (e.g. http://localhost:8000/v1/chat/completions for self-hosted vLLM)")
    parser.add_argument("--api-key", dest="api_key", help="Override API key (use 'dummy' for self-hosted vLLM)")
    parser.add_argument("--compare", action="store_true", help="Print comparison table from previous runs")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.compare:
        print_comparison_table()
        return

    prompts = load_prompts()

    # Determine scenario
    scenario_name = args.scenario or "sanity"
    if scenario_name not in config["scenarios"]:
        logger.error("Unknown scenario: %s. Available: %s", scenario_name, list(config["scenarios"].keys()))
        sys.exit(1)

    scenario = dict(config["scenarios"][scenario_name])  # copy so we can override

    # Apply CLI overrides
    if args.users is not None:
        scenario["users"] = args.users
    if args.evals is not None:
        scenario["evals_per_user"] = args.evals
    if args.rules is not None:
        scenario["rules_per_eval"] = args.rules
    if args.sentences is not None:
        scenario["sentences_per_eval"] = args.sentences
    if args.concurrency is not None:
        scenario["max_concurrency"] = args.concurrency
    if args.stagger is not None:
        scenario["stagger_seconds"] = args.stagger
    if args.batch_size is not None:
        scenario["batch_size"] = args.batch_size

    # Determine models to test
    model_keys = []
    if args.model == "all":
        model_keys = list(config["models"].keys())
    elif args.model:
        if args.model not in config["models"]:
            logger.error("Unknown model: %s. Available: %s", args.model, list(config["models"].keys()))
            sys.exit(1)
        model_keys = [args.model]
    else:
        # Default: first model in config
        model_keys = [list(config["models"].keys())[0]]

    # Apply endpoint/api-key overrides
    if args.endpoint:
        config["api"]["endpoint"] = args.endpoint
        logger.info("Endpoint override: %s", args.endpoint)
    if args.api_key:
        os.environ["UFAL_TP_APIKEY"] = args.api_key
        logger.info("API key override from --api-key")

    # Run tests
    for model_key in model_keys:
        model_conf = config["models"][model_key]
        model_id = model_conf["id"]
        model_label = model_conf.get("label", model_key)

        test_result = asyncio.run(
            run_test(model_key, model_id, model_label, config, scenario, prompts)
        )

        print_summary(test_result["summary"])
        save_results(test_result, model_key, scenario_name)

    # Print comparison if we tested multiple models
    if len(model_keys) > 1:
        print_comparison_table()


if __name__ == "__main__":
    main()
