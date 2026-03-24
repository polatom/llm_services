#!/usr/bin/env python3
"""
test_endpoint.py — Test the vLLM endpoint with SPRINT-like and PONK-like prompts.

Sends Czech legal text evaluation requests to a vLLM server, in the same
format that the SPRINT and PONK apps use in production.

No external dependencies — uses only the Python standard library.

Usage:
    python3 test_endpoint.py --url http://localhost:8421 --mode health
    python3 test_endpoint.py --url http://localhost:8421 --mode single
    python3 test_endpoint.py --url http://localhost:8421 --mode concurrent --requests 20
    python3 test_endpoint.py --url http://localhost:8421 --mode sprint
    python3 test_endpoint.py --url http://localhost:8421 --mode ponk
    python3 test_endpoint.py --url http://localhost:8421 --mode ponk --doc-chars 40000 --chunk-chars 5000
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = SCRIPT_DIR.parent / "sample_prompts.json"

DEFAULT_URL = "http://localhost:8421"
DEFAULT_MODEL = "google/gemma-3-27b-it"


# ── HTTP helpers (stdlib only, no pip dependencies) ───────────

def http_get(url: str, timeout: int = 10) -> dict:
    """GET request → parsed JSON."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_post(url: str, payload: dict, api_key: str = "dummy",
              timeout: int = 600) -> tuple:
    """POST JSON → (parsed_response, latency_seconds)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body, time.monotonic() - t0


# ── Prompt building (mirrors SPRINT backend logic) ────────────

def load_prompts() -> dict:
    """Load sample_prompts.json (shared with stress_test.py)."""
    if not PROMPTS_FILE.exists():
        print(f"ERROR: {PROMPTS_FILE} not found.")
        print(f"       Expected at: {PROMPTS_FILE}")
        print(f"       Make sure you're running from the repo.")
        sys.exit(1)
    return json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))


def build_messages(prompts: dict, rule: dict,
                   sentences: list) -> list:
    """
    Build the chat messages for one rule + sentence batch.

    This is the same prompt format the SPRINT/PONK backend sends to the LLM:
    - system message: expert persona
    - user message: rule definition + conditions + examples + sentences
    The LLM should return a JSON array of {sent_id, violation, reason, suggestion}.
    """
    sentences_text = "\n".join(
        f'- sent_id: "{sid}"\n  text: "{text}"'
        for sid, text in sentences
    )

    violation_ex = "\n".join(
        f'  - "{ex}"' for ex in rule.get("violation_examples", [])
    )
    compliant_ex = "\n".join(
        f'  - "{ex}"' for ex in rule.get("compliant_examples", [])
    )
    examples = (
        f"Violation examples:\n{violation_ex}\n"
        f"Compliant examples:\n{compliant_ex}"
    )

    example_output = (
        "Example output (for reference only, do not copy):\n"
        '[{"sent_id": "0", "violation": false, '
        '"reason": "Věta je v souladu s pravidlem.", "suggestion": null}]'
    )

    payload = prompts["payload_template"]
    payload = payload.replace("{{rule_definition}}", rule["definition"])
    payload = payload.replace("{{rule_conditions}}", rule["conditions"])
    payload = payload.replace("{{sentences_examples_input_output}}", examples)
    payload = payload.replace("{{example_output}}", example_output)
    payload = payload.replace("{{sentences_list}}", sentences_text)

    return [
        {"role": "system", "content": prompts["system_message"]},
        {"role": "user", "content": payload},
    ]


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) if present."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        s = "\n".join(lines[1:end])
    return s.strip()


# ── Test modes ────────────────────────────────────────────────

def test_health(base_url: str, model: str) -> bool:
    """GET /v1/models — is the server alive and serving our model?"""
    print(f"  Checking {base_url}/v1/models ...")
    try:
        data = http_get(f"{base_url}/v1/models")
        models = [m["id"] for m in data.get("data", [])]
        print(f"  Server is up. Available models:")
        for m in models:
            tag = " <-- target" if m == model else ""
            print(f"    - {m}{tag}")
        if model not in models:
            print(f"\n  WARNING: '{model}' not in model list.")
            print(f"           Use --model to set the correct name.")
            return False
        return True
    except urllib.error.URLError as e:
        print(f"  FAILED: Cannot connect — {e.reason}")
        print(f"          Is vLLM running? Is the URL correct?")
        return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_single(base_url: str, model: str, api_key: str) -> bool:
    """Send one evaluation request (1 rule, 5 sentences), show full response."""
    prompts = load_prompts()
    rule = prompts["rules"][0]  # Double Comparison

    # Pick 5 sentences: first 3 are violations for this rule, next 2 are clean
    annotated = prompts["sample_sentences_annotated"]
    sentences = [(i, s["text"]) for i, s in enumerate(annotated[:5])]
    expected = {}
    for i, s in enumerate(annotated[:5]):
        expected[str(i)] = s["violations"].get(rule["name"], False)

    messages = build_messages(prompts, rule, sentences)

    print(f"  Rule:      {rule['name']}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Sending request ...")
    print()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 8192,
    }

    try:
        resp, latency = http_post(
            f"{base_url}/v1/chat/completions", payload, api_key,
        )
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})

        print(f"  Response in {latency:.1f}s")
        print(f"  Tokens: {usage.get('prompt_tokens', '?')} prompt"
              f" + {usage.get('completion_tokens', '?')} completion")
        print()

        clean = strip_code_fences(content)
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as e:
            print(f"  FAIL: Response is not valid JSON")
            print(f"        {e}")
            print(f"  Raw (first 500 chars): {content[:500]}")
            return False

        print("  Valid JSON response:")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print()

        # Validate structure
        if not isinstance(parsed, list):
            print(f"  WARNING: Expected JSON array, got {type(parsed).__name__}")
            return False

        has_fields = all(
            isinstance(it, dict) and "sent_id" in it and "violation" in it
            for it in parsed
        )
        if not has_fields:
            print("  WARNING: Some items missing required fields (sent_id, violation)")
            return False

        print(f"  All {len(parsed)} items have required fields (sent_id, violation)")

        # Check against ground truth
        correct = 0
        total = 0
        for item in parsed:
            sid = str(item["sent_id"])
            if sid in expected:
                total += 1
                got = bool(item["violation"])
                exp = expected[sid]
                mark = "OK" if got == exp else "WRONG"
                print(f"    sent_id={sid}: violation={got} (expected {exp}) {mark}")
                if got == exp:
                    correct += 1

        if total:
            print(f"\n  Accuracy: {correct}/{total}"
                  f" ({100*correct/total:.0f}%)")

        return True

    except Exception as e:
        print(f"  Request FAILED: {e}")
        return False


def test_concurrent(base_url: str, model: str, api_key: str,
                    num_requests: int) -> bool:
    """Send N requests in parallel, report latency stats."""
    prompts = load_prompts()
    rules = prompts["rules"]
    annotated = prompts["sample_sentences_annotated"]

    print(f"  Sending {num_requests} concurrent requests ...")
    print()

    def send_one(rid: int) -> dict:
        rule = rules[rid % len(rules)]
        start = (rid * 5) % max(len(annotated) - 5, 1)
        sents = [(i, s["text"])
                 for i, s in enumerate(annotated[start:start + 5])]
        if not sents:
            sents = [(i, s["text"]) for i, s in enumerate(annotated[:5])]

        messages = build_messages(prompts, rule, sents)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 8192,
        }
        try:
            resp, latency = http_post(
                f"{base_url}/v1/chat/completions", payload, api_key,
            )
            content = resp["choices"][0]["message"]["content"]
            clean = strip_code_fences(content)
            try:
                json.loads(clean)
                valid = True
            except json.JSONDecodeError:
                valid = False
            return {"id": rid, "rule": rule["name"],
                    "latency": latency, "valid_json": valid, "error": None}
        except Exception as e:
            return {"id": rid, "rule": rule["name"],
                    "latency": None, "valid_json": False, "error": str(e)}

    t0 = time.monotonic()
    results = []
    workers = min(num_requests, 48)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(send_one, i): i for i in range(num_requests)}
        for fut in as_completed(futures):
            r = fut.result()
            lat = f"{r['latency']:.1f}s" if r["latency"] else "N/A"
            ok = "OK" if r["valid_json"] else (r["error"] or "bad JSON")
            print(f"    [{r['id']:3d}] {r['rule']:35s} {lat:>8s}  {ok}")
            results.append(r)

    wall = time.monotonic() - t0
    _print_summary(results, num_requests, wall)
    return all(r["error"] is None for r in results)


def test_sprint(base_url: str, model: str, api_key: str,
                batch_size: int) -> bool:
    """Simulate a full SPRINT evaluation: all 6 rules × sentence batches."""
    prompts = load_prompts()
    rules = prompts["rules"]
    annotated = prompts["sample_sentences_annotated"]
    all_sents = [(i, s["text"]) for i, s in enumerate(annotated)]

    if batch_size > 0 and len(all_sents) > batch_size:
        batches = [all_sents[i:i + batch_size]
                   for i in range(0, len(all_sents), batch_size)]
    else:
        batches = [all_sents]

    total = len(rules) * len(batches)

    print(f"  Simulating full SPRINT evaluation")
    print(f"  Rules:      {len(rules)}")
    print(f"  Sentences:  {len(all_sents)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches:    {len(batches)}")
    print(f"  Total requests: {total}"
          f" ({len(rules)} rules x {len(batches)} batches)")
    print()

    def send_one(rule_idx: int, batch_idx: int) -> dict:
        rule = rules[rule_idx]
        batch = batches[batch_idx]
        messages = build_messages(prompts, rule, batch)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 8192,
        }
        try:
            resp, latency = http_post(
                f"{base_url}/v1/chat/completions", payload, api_key,
            )
            content = resp["choices"][0]["message"]["content"]
            clean = strip_code_fences(content)
            try:
                json.loads(clean)
                valid = True
            except json.JSONDecodeError:
                valid = False
            return {"rule": rule["name"], "batch": batch_idx,
                    "latency": latency, "valid_json": valid, "error": None}
        except Exception as e:
            return {"rule": rule["name"], "batch": batch_idx,
                    "latency": None, "valid_json": False, "error": str(e)}

    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = []
        for ri in range(len(rules)):
            for bi in range(len(batches)):
                futures.append(pool.submit(send_one, ri, bi))
        for fut in as_completed(futures):
            r = fut.result()
            lat = f"{r['latency']:.1f}s" if r["latency"] else "N/A"
            ok = "OK" if r["valid_json"] else (r["error"] or "bad JSON")
            print(f"    {r['rule']:35s} batch {r['batch']}  {lat:>8s}  {ok}")
            results.append(r)

    wall = time.monotonic() - t0
    _print_summary(results, total, wall)
    return all(r["error"] is None for r in results)


# ── PONK mode: large-document throughput test ─────────────────

# PONK module 3 prompt template (speech act annotation of Czech legal text).
# Mirrors ponk-app3/llm_client.py — the actual prompt the production app sends.
PONK_SYSTEM_MESSAGE = (
    "You are an expert legal document annotator. Your task is to analyze "
    "Czech legal documents and provide structured JSON annotations. "
    "Always respond with valid JSON only, no additional text."
)

PONK_USER_TEMPLATE = """# Task: Annotate Legal Document with Speech Acts

You are an expert annotator tasked with identifying and labeling text spans in a legal advice document according to a predefined set of speech act categories. Your goal is to segment the document into meaningful spans and assign each span the most appropriate speech act label.

## Speech Acts Definitions

### 01_Situace (Situation)
Snippets of text indicating what situation (and goal) the advice applies to.

### 02_Kontext (Context)
Snippets of text giving the broader picture, for instance precedent cases or typical procedures and their outcomes.

### 03_Postup (Procedure)
Snippets of text describing what the recipient is advised to do.

### 04_Proces (Process)
Snippets of text describing the expected responses of authorities or other parties to steps taken by the recipient.

### 05_Podminky (Conditions, options)
Snippets of text specifying circumstances under which an action can or cannot be taken.

### 06_Doporuceni (Recommendations)
Snippets of text that recommend additional actions or compare the individual options with respect to their desired impact.

### 07_Odkazy (Links)
Explicit textual links to other documents in Frank Bold's knowledge base of legal advice.

### 08_Prameny (References)
References to external documents, particularly laws and regulations.

### 09_Nezaraditelne (Not classified)
Any other text.

## Instructions

1. Read through the entire document carefully
2. Identify meaningful text spans that correspond to one of the speech act categories above
3. Spans can be of any length (words, sentences, paragraphs) - prefer smaller, more granular spans
4. Each category can and should be used multiple times throughout the document
5. Spans must NOT overlap - each character position belongs to at most one annotation
6. Every part of the document should ideally be covered by at least one annotation

## Output Format

Return a JSON object with this structure:
{{
  "annotations": [
    {{
      "start": 0,
      "end": 100,
      "label": "01_Situace"
    }},
    ...
  ]
}}

Where:
- "start": character offset where the span begins (inclusive, 0-indexed, relative to the chunk below)
- "end": character offset where the span ends (exclusive)
- "label": one of the speech act labels (01_Situace, 02_Kontext, etc.)

IMPORTANT: Spans must NOT overlap.

## Document to Annotate

{text}
"""

# Sample Czech legal paragraphs used to build a synthetic large document.
_LEGAL_PARAGRAPHS = [
    "Nejvyšší správní soud rozhodl v senátě složeném z předsedy senátu "
    "JUDr. Petra Nováka a soudců JUDr. Marie Horákové a Mgr. Jana Krátkého "
    "v právní věci žalobce: ABC s.r.o., se sídlem Praha 1, Vodičkova 123/4, "
    "IČO: 12345678, zastoupeného advokátem JUDr. Tomášem Dvořákem, se sídlem "
    "Praha 2, Vinohradská 567/8, proti žalovanému: Ministerstvo financí, "
    "se sídlem Praha 1, Letenská 15, o žalobě proti rozhodnutí žalovaného "
    "ze dne 15. 3. 2024, č. j. MF-12345/2024/6789-2,",

    "Žalobce se domáhá zrušení rozhodnutí žalovaného ze dne 15. března 2024, "
    "kterým bylo zamítnuto jeho odvolání proti platebnímu výměru Finančního "
    "úřadu pro hlavní město Prahu ze dne 10. ledna 2024 na daň z příjmů "
    "právnických osob za zdaňovací období roku 2022 ve výši 2 450 000 Kč. "
    "Žalobce namítá, že žalovaný nesprávně posoudil právní otázku uznatelnosti "
    "nákladů vynaložených na výzkum a vývoj podle § 34 odst. 4 zákona "
    "č. 586/1992 Sb., o daních z příjmů.",

    "Podle § 65 odst. 1 zákona č. 150/2002 Sb., soudní řád správní, "
    "kdo tvrdí, že byl na svých právech zkrácen přímo nebo v důsledku "
    "porušení svých práv v předcházejícím řízení úkonem správního orgánu, "
    "jímž se zakládají, mění, ruší nebo závazně určují jeho práva nebo "
    "povinnosti, může se žalobou domáhat zrušení takového rozhodnutí, "
    "popřípadě vyslovení jeho nicotnosti.",

    "Soud přezkoumal napadené rozhodnutí v mezích žalobních bodů podle "
    "§ 75 odst. 2 s.ř.s. a dospěl k závěru, že žaloba je důvodná. "
    "Ze správního spisu vyplývá, že žalobce v rozhodném období realizoval "
    "projekt výzkumu a vývoje zaměřený na optimalizaci výrobních procesů "
    "s využitím strojového učení. Náklady na tento projekt byly řádně "
    "evidovány v oddělené analytické evidenci v souladu s § 34b odst. 1 "
    "zákona o daních z příjmů.",

    "Žalovaný ve svém rozhodnutí konstatoval, že výdaje na pořízení "
    "softwarových licencí a cloudových služeb v celkové výši 890 000 Kč "
    "nelze považovat za výdaje na výzkum a vývoj, neboť se jedná o běžné "
    "provozní náklady. S tímto závěrem se soud neztotožňuje. Z předložených "
    "důkazů je zřejmé, že uvedené softwarové nástroje byly pořízeny "
    "výhradně pro účely výzkumného projektu a bez nich by realizace "
    "projektu nebyla možná.",

    "Doporučuje se, aby žalobce v dalším řízení předložil podrobný rozpis "
    "nákladů s uvedením přímé vazby každé položky na konkrétní etapu "
    "výzkumného projektu. Dále je vhodné zajistit nezávislý znalecký "
    "posudek potvrzující inovativní povahu projektu ve smyslu Frascati "
    "manuálu OECD. Správní orgán je povinen se s těmito důkazy řádně "
    "vypořádat v odůvodnění svého rozhodnutí.",

    "Na základě výše uvedeného soud napadené rozhodnutí žalovaného podle "
    "§ 78 odst. 1 s.ř.s. zrušil a věc vrátil žalovanému k dalšímu řízení. "
    "V dalším řízení je žalovaný vázán právním názorem soudu vysloveným "
    "v tomto rozsudku (§ 78 odst. 5 s.ř.s.). O náhradě nákladů řízení "
    "soud rozhodl podle § 60 odst. 1 s.ř.s. tak, že žalovaný je povinen "
    "zaplatit žalobci náhradu nákladů řízení ve výši 15 342 Kč.",

    "Podle § 250 odst. 1 daňového řádu lze podat žalobu ve správním "
    "soudnictví ve lhůtě dvou měsíců ode dne doručení rozhodnutí, proti "
    "němuž žaloba směřuje. Tato lhůta je zachována, je-li žaloba podána "
    "u příslušného soudu nebo u soudu, který je povinen ji postoupit "
    "příslušnému soudu. Zmeškání lhůty nelze prominout. Stavební úřad "
    "je povinen posoudit žádost o stavební povolení v souladu s územním "
    "plánem a technickými normami platnými ke dni podání žádosti.",
]


def _generate_legal_text(target_chars: int) -> str:
    """Generate a synthetic Czech legal document of approximately target_chars."""
    paragraphs = []
    total = 0
    i = 0
    while total < target_chars:
        p = _LEGAL_PARAGRAPHS[i % len(_LEGAL_PARAGRAPHS)]
        paragraphs.append(p)
        total += len(p) + 2  # +2 for "\n\n" separator
        i += 1
    text = "\n\n".join(paragraphs)
    return text[:target_chars]


def _build_ponk_messages(text_chunk: str) -> list:
    """Build PONK-style chat messages for a text chunk."""
    return [
        {"role": "system", "content": PONK_SYSTEM_MESSAGE},
        {"role": "user", "content": PONK_USER_TEMPLATE.format(text=text_chunk)},
    ]


def test_ponk(base_urls, model: str, api_key: str,
              doc_chars: int, chunk_chars: int, target_seconds: float) -> bool:
    """
    Simulate PONK module 3: large document → split into chunks → send concurrently.

    The PONK app sends an entire document (up to 40k chars) to the LLM for
    speech-act annotation. A single 40k request is too slow, so the plan is to
    split into ~5k-char chunks and send them in parallel. This test verifies
    that the model + vLLM deployment can handle this within the target time.
    """
    doc = _generate_legal_text(doc_chars)

    # Split into chunks at paragraph boundaries (\n\n)
    chunks = []
    start = 0
    while start < len(doc):
        end = min(start + chunk_chars, len(doc))
        # Try to split at a paragraph boundary
        if end < len(doc):
            boundary = doc.rfind("\n\n", start, end)
            if boundary > start:
                end = boundary + 2  # include the \n\n
        chunks.append((start, doc[start:end]))
        start = end

    print(f"  Simulating PONK module 3 (speech act annotation)")
    print(f"  Document:   {len(doc):,} chars ({len(doc)/1000:.0f}k)")
    print(f"  Chunk size: ~{chunk_chars:,} chars")
    print(f"  Chunks:     {len(chunks)}")
    print(f"  Target:     {target_seconds:.0f}s wall clock")
    print()
    for i, (off, c) in enumerate(chunks):
        print(f"    chunk {i}: offset {off:>6,} — {off+len(c):>6,}  "
              f"({len(c):,} chars)")
    print()

    # Support both single URL (str) and list of URLs
    if isinstance(base_urls, str):
        base_urls = [base_urls]
    num_servers = len(base_urls)
    if num_servers > 1:
        print(f"  Servers:    {num_servers} (round-robin distribution)")
        print()

    def send_chunk(chunk_idx: int) -> dict:
        offset, text = chunks[chunk_idx]
        # Round-robin: distribute chunks across servers
        url = base_urls[chunk_idx % num_servers]
        messages = _build_ponk_messages(text)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 8192,
        }
        try:
            resp, latency = http_post(
                f"{url}/v1/chat/completions", payload, api_key,
                timeout=600,
            )
            content = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
            clean = strip_code_fences(content)
            try:
                parsed = json.loads(clean)
                # Accept both {"annotations": [...]} and [...]
                if isinstance(parsed, dict):
                    annots = parsed.get("annotations", [])
                elif isinstance(parsed, list):
                    annots = parsed
                else:
                    annots = []
                valid = len(annots) > 0
            except json.JSONDecodeError:
                annots = []
                valid = False
            return {
                "chunk": chunk_idx, "offset": offset,
                "chars": len(text), "latency": latency,
                "valid_json": valid, "annotations": len(annots),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "error": None,
            }
        except Exception as e:
            return {
                "chunk": chunk_idx, "offset": offset,
                "chars": len(text), "latency": None,
                "valid_json": False, "annotations": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "error": str(e),
            }

    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = {pool.submit(send_chunk, i): i for i in range(len(chunks))}
        for fut in as_completed(futures):
            r = fut.result()
            lat = f"{r['latency']:.1f}s" if r["latency"] else "N/A"
            ok = f"{r['annotations']} annots" if r["valid_json"] else (
                r["error"] or "bad JSON")
            tok = (f"{r['prompt_tokens']}+{r['completion_tokens']}tok"
                   if r["prompt_tokens"] else "")
            print(f"    chunk {r['chunk']:2d}  {r['chars']:>6,} chars  "
                  f"{lat:>8s}  {tok:>20s}  {ok}")
            results.append(r)

    wall = time.monotonic() - t0

    # Summary
    latencies = sorted(r["latency"] for r in results if r["latency"])
    valid = sum(1 for r in results if r["valid_json"])
    errors = sum(1 for r in results if r["error"])
    total_annots = sum(r["annotations"] for r in results)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)

    met_target = wall <= target_seconds

    print()
    print("=" * 60)
    print(f"  PONK Throughput Test")
    print(f"  Document:    {len(doc):,} chars")
    print(f"  Chunks:      {len(chunks)} ({chunk_chars:,} chars each)")
    print(f"  Valid JSON:  {valid}/{len(chunks)}")
    print(f"  Annotations: {total_annots} total")
    print(f"  Tokens:      {total_prompt:,} prompt + {total_completion:,} completion")
    print(f"  Errors:      {errors}/{len(chunks)}")
    if latencies:
        print(f"  Latency min: {latencies[0]:.1f}s")
        print(f"  Latency med: {latencies[len(latencies) // 2]:.1f}s")
        print(f"  Latency max: {latencies[-1]:.1f}s")
    print(f"  Wall time:   {wall:.1f}s")
    print(f"  Target:      {target_seconds:.0f}s")
    print(f"  Result:      {'PASS' if met_target else 'FAIL'}")
    if not met_target:
        print(f"  Suggestion:  Try smaller chunks (--chunk-chars {chunk_chars // 2})"
              f" or fewer chunks to reduce output tokens.")
    print("=" * 60)

    return met_target and errors == 0


# ── Summary printer ───────────────────────────────────────────

def _print_summary(results: list, total: int, wall: float):
    latencies = sorted(r["latency"] for r in results if r["latency"])
    valid = sum(1 for r in results if r["valid_json"])
    errors = sum(1 for r in results if r["error"])

    print()
    print("=" * 60)
    print(f"  Requests:    {total}")
    print(f"  Valid JSON:  {valid}/{total}")
    print(f"  Errors:      {errors}/{total}")
    print(f"  Wall time:   {wall:.1f}s")
    if latencies:
        print(f"  Latency min: {latencies[0]:.1f}s")
        print(f"  Latency med: {latencies[len(latencies) // 2]:.1f}s")
        p90 = latencies[min(int(len(latencies) * 0.9), len(latencies) - 1)]
        print(f"  Latency p90: {p90:.1f}s")
        print(f"  Latency max: {latencies[-1]:.1f}s")
        print(f"  Throughput:  {total / wall:.2f} req/s")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test the vLLM endpoint with SPRINT-like Czech legal text prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_endpoint.py --url http://localhost:8421 --mode health
  python3 test_endpoint.py --url http://tdll-8gpu5:8421 --mode single
  python3 test_endpoint.py --mode concurrent --requests 20
  python3 test_endpoint.py --mode sprint
  python3 test_endpoint.py --mode ponk
  python3 test_endpoint.py --mode ponk --doc-chars 40000 --chunk-chars 5000 --target-seconds 30
        """,
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"vLLM base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--urls", nargs="+", default=None,
        help="Multiple vLLM base URLs for multi-server mode (e.g. "
             "--urls http://host:8421 http://host:8422). "
             "Chunks are distributed round-robin across servers. "
             "If not set, uses --url for all requests.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name as shown by /v1/models (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key", default="dummy",
        help="API key (default: dummy — self-hosted vLLM needs no auth)",
    )
    parser.add_argument(
        "--mode", choices=["health", "single", "concurrent", "sprint", "ponk"],
        default="single",
        help="Test mode (default: single)",
    )
    parser.add_argument(
        "--requests", type=int, default=10,
        help="Number of requests for --mode concurrent (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=15,
        help="Sentences per request for --mode sprint (default: 15)",
    )
    parser.add_argument(
        "--doc-chars", type=int, default=40000,
        help="Document size in chars for --mode ponk (default: 40000)",
    )
    parser.add_argument(
        "--chunk-chars", type=int, default=5000,
        help="Chunk size in chars for --mode ponk (default: 5000)",
    )
    parser.add_argument(
        "--target-seconds", type=float, default=30.0,
        help="Target wall time for --mode ponk (default: 30)",
    )

    args = parser.parse_args()

    # Build the list of server URLs
    all_urls = args.urls if args.urls else [args.url]

    print()
    print("=" * 60)
    print("  vLLM Endpoint Test")
    if len(all_urls) == 1:
        print(f"  URL:   {all_urls[0]}")
    else:
        print(f"  URLs:  {len(all_urls)} servers")
        for u in all_urls:
            print(f"         {u}")
    print(f"  Model: {args.model}")
    print(f"  Mode:  {args.mode}")
    print("=" * 60)
    print()

    # Health check — verify all servers
    healthy = True
    for u in all_urls:
        if not test_health(u, args.model):
            healthy = False

    if args.mode == "health":
        sys.exit(0 if healthy else 1)

    if not healthy:
        print("\n  Server is not reachable or model not found. Fix first.")
        sys.exit(1)

    print()

    ok = False
    if args.mode == "single":
        ok = test_single(all_urls[0], args.model, args.api_key)
    elif args.mode == "concurrent":
        ok = test_concurrent(all_urls[0], args.model, args.api_key, args.requests)
    elif args.mode == "sprint":
        ok = test_sprint(all_urls[0], args.model, args.api_key, args.batch_size)
    elif args.mode == "ponk":
        ok = test_ponk(all_urls, args.model, args.api_key,
                       args.doc_chars, args.chunk_chars, args.target_seconds)

    print()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
