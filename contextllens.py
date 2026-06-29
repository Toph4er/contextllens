#!/usr/bin/env python3
"""
contextllens.py — LLM long-context & concurrency benchmark.

Version 0.4.0

Features:
  - Multi-Needle Retrieval: Tests 3 needles at 10%, 50%, and 90% context depth.
  - Coherency Suite: Detects looping, length failures, and structural collapse.
  - Enhanced Metrics: All coherency checks are exported to JSON/CSV.

Modes:
  single  (default) — One-shot benchmark with multi-needle prompt
  warm    — Cold + warm run comparison (KV cache effect)
  ramp    — Growing context benchmark (powers of 2 from 1K to target)
  concurrency=N — N concurrent requests at once (default: concurrency=2)
  concurrency-ramp=N — Scaling concurrency: 1→2→4→...→N

Configuration:
  Copy config.yaml.example to config.yaml and edit with your endpoints.
  Or specify a custom config with --config.

Results:
  Results are saved by default to ./results/. Use --results-path to override.
  Each run creates a timestamped subfolder with results.txt, results.csv,
  results.json, and an Output/ folder with full LLM responses.
"""

__version__ = "0.4.0"

import argparse
import concurrent.futures
import csv
import io
import json
import os
import re
import sys
import time
from datetime import datetime
import requests


# ============================================================

# ============================================================
# HAYSTACK DATA — loaded from haystack_data.json
# ============================================================

def _load_haystack_data():
    """Load haystack paragraphs, needles, query template, and constraints.

    Reads from haystack_data.json in the script directory. Falls back to
    inline defaults if the file is missing or malformed."""
    import json as _json

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haystack_data.json')
    try:
        with open(data_path, encoding="utf-8") as f:
            return _json.load(f)
    except (FileNotFoundError, _json.JSONDecodeError) as e:
        print(f"Warning: could not load haystack_data.json ({e}), using inline defaults.")
        return None


# Lazy-loaded globals — populated on first access
_HAYSTACK_DATA = None


def _get_haystack_data():
    """Return the loaded haystack data."""
    global _HAYSTACK_DATA
    if _HAYSTACK_DATA is None:
        _HAYSTACK_DATA = _load_haystack_data()
    return _HAYSTACK_DATA



# ============================================================
# RUN STATE — per-prompt randomized needles
# ============================================================

_run_rng = None  # seeded RNG for reproducible randomization


def _init_run_state(seed: int):
    """Initialize the run-level RNG from a seed.

    Each call to build_haystack_prompt should call this first with
    its own per-step seed so needles are deterministic
    but vary across ramp steps."""
    global _run_rng
    import random as _random
    _run_rng = _random.Random(seed)


def get_needle_set() -> dict:
    """Return a randomly selected needle set (start/mid/end) from the pool.

    Uses _run_rng so results are deterministic for a given seed."""
    data = _get_haystack_data()
    global _run_rng
    if data and "needle_sets" in data:
        sets = data["needle_sets"]
        return _run_rng.choice(sets)
    # Inline fallback (original single set)
    return {
        "start": "CRITICAL REFERENCE START: The authorization code is VERTEX-1111-A",
        "mid":   "CRITICAL REFERENCE MID: The authorization code is VERTEX-2222-B",
        "end":   "CRITICAL REFERENCE END: The authorization code is VERTEX-3333-C"
    }


def get_decoy_codes(selected_needles: dict) -> list[str]:
    """Return decoy paragraphs from needle sets NOT selected for this run.

    This ensures that some codes in the haystack are real (from other valid sets)
    and others are fake, preventing models from learning static code patterns."""
    data = _get_haystack_data()
    if not data or "needle_sets" not in data:
        return []

    # Build a signature of the selected set so we can exclude it
    selected_codes = set()
    for text in selected_needles.values():
        # Extract the code part (e.g., "VERTEX-1111-A")
        if " is " in text:
            selected_codes.add(text.split(" is ")[-1])
        elif ":" in text:
            parts = text.split()
            for p in parts:
                if "-" in p and len(p) > 5:
                    selected_codes.add(p)

    # Collect all codes from non-selected sets as decoy paragraphs
    decoys = []
    for ns in data["needle_sets"]:
        # Check if this set matches the selected one (any code overlap)
        ns_codes = set()
        for text in ns.values():
            if " is " in text:
                ns_codes.add(text.split(" is ")[-1])
            elif ":" in text:
                parts = text.split()
                for p in parts:
                    if "-" in p and len(p) > 5:
                        ns_codes.add(p)

        # If no overlap with selected, all entries are decoys
        if not ns_codes.intersection(selected_codes):
            decoys.extend(ns.values())

    return decoys




def get_haystack_paragraphs():
    """Return flat list of all haystack paragraphs from all categories."""
    data = _get_haystack_data()
    if data:
        paras = []
        for category, items in sorted(data.get("paragraphs", {}).items()):
            paras.extend(items)
        return paras
    return []



def get_query():
    """Return the query string from the configured template."""
    data = _get_haystack_data()
    if data:
        return data.get("query_template", "")
    # Inline fallback
    return (
        "What are the three production authorization codes (Alpha, Bravo, and Charlie) mentioned in the document above? "
        "After providing the codes, organize the information from the document into a structured report. "
        "Group the content into meaningful categories and provide a detailed summary of each category. "
        "The report MUST be approximately 1000 words."
    )



class TeeWriter:
    """Writes to both stdout and a StringIO buffer."""
    def __init__(self, original_stdout):
        self.original = original_stdout
        self.buffer = io.StringIO()

    def write(self, text):
        self.original.write(text)
        self.original.flush()
        self.buffer.write(text)

    def flush(self):
        self.original.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def load_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required. Install with: pip install pyyaml")
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Copy config.yaml.example to config.yaml and edit with your endpoints.")
        sys.exit(1)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    models = data.get("models", {})
    if not models:
        print(f"Error: No models found in {config_path}")
        sys.exit(1)

    return models


def find_config() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ["config.yaml", "config.yaml.example"]:
        path = os.path.join(script_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(script_dir, "config.yaml")


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~1.3 chars/token for English text."""
    return max(1, int(len(text) / 1.3))


# ============================================================
# NEEDLE & COHERENCY CHECKS
# ============================================================


def _normalize_unicode(text: str) -> str:
    """Normalize common Unicode lookalikes to ASCII for needle matching."""
    return (
        text
        .replace("\u2011", "-")  # non-breaking hyphen
        .replace("\u2013", "-")  # en-dash
        .replace("\u2014", "-")  # em-dash
        .replace("\u00ad", "")   # soft hyphen (zero-width)
    )



# ============================================================
# NEEDLE & COHERENCY CHECKS
# ============================================================

def _normalize_unicode(text: str) -> str:
    """Normalize common Unicode lookalikes to ASCII for needle matching."""
    return (
        text
        .replace("\u2011", "-")  # non-breaking hyphen
        .replace("\u2013", "-")  # en-dash
        .replace("\u2014", "-")  # em-dash
        .replace("\u00ad", "")   # soft hyphen (zero-width)
    )


def check_needles(text: str, needles: dict) -> tuple[int, list[str]]:
    """Check which needles were retrieved. Returns (count, list of found labels).

    Uses Unicode normalization so codes with en-dashes/other lookalikes still match.
    needles: dict of {position: full_needle_string} for this run."""
    normalized = _normalize_unicode(text)
    found = []
    for key, needle_text in needles.items():
        code = needle_text.split("is ")[1].split()[0] if " is " in needle_text else needle_text.split()[-1]
        if code in normalized:
            found.append(key)
    return len(found), found

def _needles_badge(needles_found: list[str]) -> str:
    """Return a compact per-needle emoji badge in start/mid/end order.

    Examples: '✅ ✅ ✅' (all found), '✅ ❌ ✅' (middle missing)."""
    parts = []
    for key in ["start", "mid", "end"]:
        parts.append("✅" if key in needles_found else "❌")
    return " ".join(parts)

def validate_coherency(text: str):
    """Performs automated sanity checks on the model output."""
    words = text.split()
    word_count = len(words)

    # 1. Repetition Check (4-gram uniqueness)
    grams = [" ".join(words[i:i+4]) for i in range(len(words)-3)]
    unique_ratio = len(set(grams)) / len(grams) if grams else 1.0
    is_looping = unique_ratio < 0.5

    # 2. Length Constraint (Target ~1000 words, window 400-1500)
    length_ok = 400 <= word_count <= 1500

    return {
        "is_looping": is_looping,
        "length_ok": length_ok,
        "word_count": word_count,
    }


# ============================================================
# PROMPT BUILDER
# ============================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a technical analyst tasked with extracting information from "
    "documents and writing formal reports."
)


def build_haystack_prompt(target_tokens: int, seed: int = None,
                          use_system_prompt: bool = True) -> tuple[str, str | None, dict]:
    """Build a varied haystack with needles embedded at 10%, 50%, 90%.

    Returns (user_prompt, system_prompt_or_none, metadata_dict).
    metadata_dict contains {"needles": {...}}.

    use_system_prompt: if True, returns a system prompt; if False, returns None
    so the caller can skip the system role in the API request.

    The target_tokens refers to the total prompt size including needles, query,
    and system prompt. The haystack paragraph content is sized to fill the
    remaining budget after accounting for fixed overhead."""

    # Initialize per-step RNG — each call gets its own seed derived from
    # the global seed + target_tokens, so ramp steps get different needles.
    import random as _random
    if seed is not None:
        step_seed = (seed * 31 + target_tokens) % (2**31)
    else:
        step_seed = _random.randint(0, 2**31 - 1)
    _init_run_state(step_seed)

    # Select needles for this run
    needles = get_needle_set()

    # Calculate fixed overhead: needles + query + optional system prompt
    # This is subtracted from target so the paragraph loop fills only the
    # remaining budget, keeping the total close to target_tokens.
    needle_tokens = sum(estimate_tokens(v) for v in needles.values())
    query_text = get_query()
    query_tokens = estimate_tokens(query_text)
    sys_tokens = estimate_tokens(DEFAULT_SYSTEM_PROMPT) if use_system_prompt else 0
    fixed_overhead = needle_tokens + query_tokens + sys_tokens

    # Paragraph content budget: target minus fixed overhead, with a floor
    # of 500 tokens so we always generate meaningful content.
    paragraph_budget = max(500, target_tokens - fixed_overhead)

    paragraphs_pool = list(get_haystack_paragraphs())
    if not paragraphs_pool:
        print('Error: no haystack paragraphs available. Check haystack_data.json.')
        sys.exit(1)

    # Add decoy codes from non-selected needle sets to the paragraph pool
    decoys = get_decoy_codes(needles)
    if decoys:
        paragraphs_pool.extend(decoys)

    # Shuffle paragraph order (deterministic for same seed)
    step_rng = _random.Random(step_seed + 999)
    paragraphs_pool_shuffled = list(paragraphs_pool)
    step_rng.shuffle(paragraphs_pool_shuffled)

    # Insertion points are relative to the total target (including overhead),
    # not just the paragraph budget.
    insertion_points = {
        "start": int(target_tokens * 0.10),
        "mid":   int(target_tokens * 0.50),
        "end":   int(target_tokens * 0.90),
    }

    paragraphs = []
    current_tokens = 0
    inserted = set()
    idx = 0

    while current_tokens < paragraph_budget:
        # Check if we hit an insertion point
        for key, pos in insertion_points.items():
            if current_tokens >= pos and key not in inserted:
                paragraphs.append(needles[key])
                current_tokens += estimate_tokens(needles[key])
                inserted.add(key)

        para = paragraphs_pool_shuffled[idx % len(paragraphs_pool_shuffled)]
        paragraphs.append(para)
        current_tokens += estimate_tokens(para)
        idx += 1

    # Ensure all needles were inserted if the loop ended early
    for key, needle in needles.items():
        if key not in inserted:
            paragraphs.append(needle)

    user_prompt = "\n\n".join(paragraphs) + "\n\n" + query_text
    metadata = {"needles": needles}

    system_prompt = DEFAULT_SYSTEM_PROMPT if use_system_prompt else None
    return user_prompt, system_prompt, metadata

class ResultsSaver:
    def __init__(self, results_path: str, model_key: str, mode: str,
                 context_tokens: int, cfg: dict, notes: str = ""):
        self.results_path = results_path
        self.model_key = model_key
        self.mode = mode
        self.context_tokens = context_tokens
        self.cfg = cfg
        self.timestamp = datetime.now()
        self.model_label = cfg.get("label", model_key)
        self.notes = notes
        self.runs = []

        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', model_key).strip('_')
        time_str = self.timestamp.strftime("%Y-%m-%d_%H-%M")
        folder_name = f"{time_str}_{safe_key}_{mode.upper()}_{context_tokens}"
        self.run_dir = os.path.join(results_path, folder_name)
        self.output_dir = os.path.join(self.run_dir, "Output")
        os.makedirs(self.output_dir, exist_ok=True)

    def save_run(self, step_label: str, metrics: dict, collected_text: str):
        self.runs.append((step_label, metrics, collected_text))
        if collected_text:
            safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', step_label).strip('_')
            safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', self.model_key).strip('_')
            filename = f"{safe_key}_{safe_label}.txt"
            with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as f:
                f.write(collected_text)

    def finalize(self, terminal_output: str):
        # results.txt — raw terminal output
        txt_path = os.path.join(self.run_dir, "results.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(terminal_output)

        # notes.txt — user-supplied run notes
        if self.notes:
            notes_path = os.path.join(self.run_dir, "notes.txt")
            with open(notes_path, "w", encoding="utf-8") as f:
                f.write(self.notes)

        # results.json — structured metrics
        json_data = {
            "run_info": {
                "timestamp": self.timestamp.isoformat(),
                "model_key": self.model_key,
                "model_label": self.model_label,
                "endpoint": self.cfg.get("endpoint", ""),
                "mode": self.mode,
                "target_context_tokens": self.context_tokens,
                "notes": self.notes,
            },
            "runs": []
        }
        for step_label, metrics, _ in self.runs:
            entry = {"step": step_label}
            entry.update(metrics)
            entry.pop("collected_text", None)
            entry.pop("metadata", None)
            json_data["runs"].append(entry)

        json_path = os.path.join(self.run_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        # results.csv — tabular metrics
        csv_path = os.path.join(self.run_dir, "results.csv")
        if self.runs:
            fieldnames = ["step", "prompt_tokens", "completion_tokens", "total_tokens",
                          "ttft", "prefill_speed", "decode_duration", "gen_speed", "tpot",
                          "wall_clock", "needle_count", "needles_found",
                          "is_looping", "length_ok",
                          "word_count", "reasoning_tokens", "reasoning_is_looping"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for step_label, metrics, _ in self.runs:
                    row = {"step": step_label}
                    row.update(metrics)
                    writer.writerow(row)

        return self.run_dir


# ============================================================
# DISPLAY HELPERS (from v1, adapted for multi-needle + coherency)
# ============================================================

def _format_result_line(idx: int, r: dict) -> str:
    """Format a single request result as a display line."""
    badge = _needles_badge(r.get('needles_found', []))
    coherency_parts = []
    if r.get("is_looping"):
        coherency_parts.append("LOOP")
    if not r.get("length_ok", True):
        coherency_parts.append(f"LEN={r['word_count']}w")
    coh_tag = " | ".join(coherency_parts) if coherency_parts else ""

    # Token breakdown
    reasoning_tok = r.get("reasoning_tokens", 0)
    total_tok = r.get("completion_tokens", 0)
    output_tok = total_tok - reasoning_tok
    tok_breakdown = f" ({output_tok}o/{reasoning_tok}r)" if reasoning_tok > 0 else ""

    line = (
        f"  Request {idx:>2}: {badge}  "
        f"TTFT={r['ttft']:.2f}s  "
        f"Prefill={r['prefill_speed']:,.0f}tok/s  "
        f"GenSpeed={r['gen_speed']:.1f}tok/s  "
        f"TPOT={r['tpot']:.2f}ms  "
        f"Wall={r['wall_clock']:.1f}s"
    )
    if tok_breakdown:
        line += tok_breakdown
    if coh_tag:
        line += f"  ⚠️ {coh_tag}"
    return line


def print_results(results: dict, label: str = "", show_preview: bool = True):
    """Print formatted benchmark results in compact style."""
    sep = "─" * 60
    if label:
        print(sep)
        print(f"  {label}")
        print(sep)

    print(_format_result_line(1, results))

    # Coherency summary line
    r = results
    coh_items = []
    coh_items.append(f"Looping={'YES' if r.get('is_looping') else 'no'}")
    coh_items.append(f"Length={r['word_count']}w {'(OK)' if r.get('length_ok') else '(OUT OF RANGE)'}")
    print(f"\n  Coherency: {' | '.join(coh_items)}")

    # Token breakdown
    reasoning_tok = r.get("reasoning_tokens", 0)
    total_tok = r.get("completion_tokens", 0)
    output_tok = total_tok - reasoning_tok
    if reasoning_tok > 0:
        print(f"  Tokens: {output_tok} output + {reasoning_tok} reasoning = {total_tok} total")

    # Reasoning diagnostics
    if reasoning_tok > 0:
        reason_items = []
        reason_items.append(f"ReasoningLoop={'YES' if r.get('reasoning_is_looping') else 'no'}")
        print(f"  Reasoning: {' | '.join(reason_items)}")

    if show_preview and results["collected_text"]:
        snippet = results["collected_text"][:300]
        print(f"\n  Output preview: {snippet!r}")
        if len(results["collected_text"]) > 300:
            print(f"  ... ({len(results['collected_text'])} chars total)")
    print()


# ============================================================
# BENCHMARK CORE — streaming, with real TTFT / gen speed / TPOT
# ============================================================

def run_single_benchmark(cfg: dict, user_prompt: str, max_tokens: int, timeout: int,
                         metadata: dict, bearer_token: str = None,
                         system_prompt: str = None):
    """Execute a single request with streaming and measure all performance and coherency metrics.

    metadata: dict from build_haystack_prompt containing needles for validation.
    system_prompt: optional system role message (None = no system role)."""
    endpoint = cfg["endpoint"]
    model = cfg.get("model", "")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.1,
        "stream":      True,
    }

    headers = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    start_wall = time.time()

    try:
        response = requests.post(
            f"{endpoint}/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=timeout,
        )
        if response.status_code != 200:
            print(f"  Server Error ({response.status_code}): {response.text[:500]}")
            return None
    except requests.exceptions.ConnectionError as e:
        print(f"  Connection failed: {e}")
        return None
    except requests.exceptions.Timeout:
        print(f"  Request timed out after {timeout}s.")
        return None

    # ---- streaming metrics ----
    ttft            = None
    first_token_ts  = None
    last_token_ts   = None
    delta_token_cnt = 0
    usage           = None

    # Separate buffers for thinking vs output
    reasoning_text  = ""   # reasoning_content / reasoning
    output_text     = ""   # content
    collected_text  = ""   # combined (for backward compat)

    for raw_line in response.iter_lines():
        if not raw_line:
            continue

        line = raw_line.decode("utf-8").strip()
        if line == "data: [DONE]":
            break

        if line.startswith("data: "):
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            if "usage" in chunk:
                usage = chunk["usage"]

            choices = chunk.get("choices")
            if not choices:
                continue

            delta = choices[0].get("delta", {})

            # Track which section the token belongs to
            reasoning_token = delta.get("reasoning_content") or delta.get("reasoning")
            output_token    = delta.get("content")

            if reasoning_token:
                reasoning_text += reasoning_token
                collected_text += reasoning_token
                delta_token_cnt += 1
                if first_token_ts is None:
                    first_token_ts = time.time()
                    ttft = first_token_ts - start_wall
                last_token_ts = time.time()

            if output_token:
                output_text += output_token
                collected_text += output_token
                delta_token_cnt += 1
                if first_token_ts is None:
                    first_token_ts = time.time()
                    ttft = first_token_ts - start_wall
                last_token_ts = time.time()

    end_time = time.time()

    # ---- resolve token counts ----
    est_prompt = estimate_tokens(user_prompt)
    if usage:
        prompt_tokens     = usage.get("prompt_tokens", est_prompt)
        completion_tokens = usage.get("completion_tokens", delta_token_cnt)
        total_tokens      = usage.get("total_tokens", prompt_tokens + completion_tokens)
    else:
        prompt_tokens     = est_prompt
        completion_tokens = delta_token_cnt
        total_tokens      = prompt_tokens + completion_tokens

    # ---- timing ----
    if first_token_ts and last_token_ts:
        decode_duration = last_token_ts - first_token_ts
    else:
        decode_duration = end_time - start_wall

    gen_speed     = (completion_tokens / decode_duration) if (completion_tokens > 1 and decode_duration > 0) else 0.0
    prefill_speed = (prompt_tokens / ttft) if (ttft and prompt_tokens > 0) else 0.0
    tpot          = (decode_duration / completion_tokens * 1000) if (completion_tokens > 0) else 0.0

    # ---- coherency and retrieval validation ----
    # Full text = reasoning + output (for thinking models, reasoning IS the output)
    full_text = reasoning_text + output_text

    # Coherency: checked against full text (what the user actually sees)
    coherency = validate_coherency(full_text)

    # Needle retrieval: checked against full response
    needle_count, needles_found = check_needles(full_text, metadata["needles"])

    metrics = {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
        "ttft":              ttft,
        "prefill_speed":     prefill_speed,
        "decode_duration":   decode_duration,
        "gen_speed":         gen_speed,
        "tpot":              tpot,
        "wall_clock":        end_time - start_wall,
        "needle_count":      needle_count,
        "needles_found":     needles_found,
        "metadata":          metadata,  # for display (will be stripped before save)
        **coherency
    }

    metrics["collected_text"] = collected_text
    metrics["reasoning_text"] = reasoning_text
    metrics["output_text"] = output_text

    # Reasoning diagnostics
    reasoning_words = reasoning_text.split() if reasoning_text else []
    metrics["reasoning_tokens"] = len(reasoning_words)
    if reasoning_text:
        reasoning_grams = [" ".join(reasoning_words[i:i+4]) for i in range(len(reasoning_words)-3)]
        reasoning_unique = len(set(reasoning_grams)) / len(reasoning_grams) if reasoning_grams else 1.0
        metrics["reasoning_is_looping"] = reasoning_unique < 0.5
    else:
        metrics["reasoning_is_looping"] = False

    return metrics


# ============================================================
# BENCHMARK MODES
# ============================================================

def run_single(cfg: dict, context_tokens: int, max_tokens: int, timeout: int,
               saver: ResultsSaver | None, seed: int, bearer_token: str = None,
               use_system_prompt: bool = True):
    """One-shot benchmark with multi-needle prompt."""
    user_prompt, system_prompt, metadata = build_haystack_prompt(
        context_tokens, seed=seed, use_system_prompt=use_system_prompt)
    est = estimate_tokens(user_prompt)
    print(f"Sending request ({est:,} est. prompt tokens)...")
    results = run_single_benchmark(cfg, user_prompt, max_tokens, timeout, metadata,
                                   bearer_token, system_prompt=system_prompt)
    if results:
        print_results(results)
        if saver:
            saver.save_run("single", results, results.get("collected_text", ""))


def run_warm(cfg: dict, context_tokens: int, max_tokens: int, timeout: int,
             saver: ResultsSaver | None, seed: int, bearer_token: str = None,
             use_system_prompt: bool = True):
    """Cold + warm run comparison to measure KV cache effect."""
    user_prompt, system_prompt, metadata = build_haystack_prompt(
        context_tokens, seed=seed, use_system_prompt=use_system_prompt)
    est = estimate_tokens(user_prompt)

    print(f"Running COLD benchmark ({est:,} est. prompt tokens)...")
    cold = run_single_benchmark(cfg, user_prompt, max_tokens, timeout, metadata,
                                bearer_token, system_prompt=system_prompt)
    if not cold:
        return

    print(f"Running WARM benchmark (same prompt, KV cache should be warm)...")
    warm = run_single_benchmark(cfg, user_prompt, max_tokens, timeout, metadata,
                                bearer_token, system_prompt=system_prompt)
    if not warm:
        return

    print_results(cold, label="Cold Run", show_preview=False)
    print_results(warm, label="Warm Run", show_preview=False)

    # Comparison
    print("  Comparison:")
    if cold["ttft"] and warm["ttft"]:
        speedup = cold['ttft'] / warm['ttft']
        print(f"    TTFT Speedup:           {speedup:.1f}x  ({cold['ttft']:.2f}s → {warm['ttft']:.2f}s)")
    if cold["gen_speed"] and warm["gen_speed"]:
        ratio = cold['gen_speed'] / warm['gen_speed']
        print(f"    GenSpeed Ratio:         {ratio:.2f}x  ({cold['gen_speed']:.1f} → {warm['gen_speed']:.1f} tok/s)")
    if cold["prefill_speed"] and warm["prefill_speed"]:
        speedup = warm['prefill_speed'] / cold['prefill_speed']
        print(f"    Prefill Speedup:        {speedup:.1f}x  ({cold['prefill_speed']:.0f} → {warm['prefill_speed']:.0f} tok/s)")
    if cold["wall_clock"] and warm["wall_clock"]:
        speedup = cold['wall_clock'] / warm['wall_clock']
        print(f"    Wall Clock Speedup:     {speedup:.1f}x  ({cold['wall_clock']:.1f}s → {warm['wall_clock']:.1f}s)")

    # Needle comparison
    c_badge = _needles_badge(cold.get("needles_found", []))
    w_badge = _needles_badge(warm.get("needles_found", []))
    print(f"    Cold Needle:            {c_badge}")
    print(f"    Warm Needle:            {w_badge}")

    # Coherency comparison
    cold_issues = []
    if cold.get("is_looping"):
        cold_issues.append("LOOPING")
    if not cold.get("length_ok", True):
        cold_issues.append(f"LEN={cold['word_count']}w")
    warm_issues = []
    if warm.get("is_looping"):
        warm_issues.append("LOOPING")
    if not warm.get("length_ok", True):
        warm_issues.append(f"LEN={warm['word_count']}w")

    print(f"    Cold Coherency:         {'clean' if not cold_issues else ', '.join(cold_issues)}")
    print(f"    Warm Coherency:         {'clean' if not warm_issues else ', '.join(warm_issues)}")

    # Reasoning comparison
    c_reasoning = cold.get("reasoning_tokens", 0)
    w_reasoning = warm.get("reasoning_tokens", 0)
    if c_reasoning > 0 or w_reasoning > 0:
        print(f"    Cold Reasoning Tokens:  {c_reasoning}")
        print(f"    Warm Reasoning Tokens:  {w_reasoning}")
    print()

    if saver:
        saver.save_run("cold", cold, cold.get("collected_text", ""))
        saver.save_run("warm", warm, warm.get("collected_text", ""))


def run_ramp(cfg: dict, max_context_tokens: int, max_tokens: int, timeout: int,
             saver: ResultsSaver | None, seed: int, bearer_token: str = None,
             use_system_prompt: bool = True):
    """
    Growing context benchmark: powers of 2 from 1K to target.
    Each step is an independent request (no KV cache reuse between steps).
    """
    steps = []
    current = 1000
    while current <= max_context_tokens:
        steps.append(current)
        current *= 2
    if not steps:
        steps = [max_context_tokens]

    print(f"Ramp steps: {', '.join(f'{s:,}' for s in steps)}\n")

    results = []
    for step in steps:
        print(f"  [{step:>7,} tokens] ", end="", flush=True)
        user_prompt, system_prompt, metadata = build_haystack_prompt(
            step, seed=seed, use_system_prompt=use_system_prompt)
        result = run_single_benchmark(cfg, user_prompt, max_tokens, timeout, metadata,
                                      bearer_token, system_prompt=system_prompt)
        if result:
            results.append((step, result))
            badge = _needles_badge(result.get('needles_found', []))
            coh_tag = ""
            if result.get("is_looping"):
                coh_tag += " LOOP"
            if not result.get("length_ok", True):
                coh_tag += f" LEN={result['word_count']}w"

            print(f"TTFT={result['ttft']:.2f}s  Prefill={result['prefill_speed']:,.0f}tok/s  "
                  f"GenSpeed={result['gen_speed']:.1f}tok/s  TPOT={result['tpot']:.2f}ms  "
                  f"Wall={result['wall_clock']:.1f}s  {badge}{coh_tag}")
            if saver:
                step_label = f"{step:,}"
                saver.save_run(step_label, result, result.get("collected_text", ""))
        else:
            print("FAILED")
            break

    if not results:
        return

    # Summary table
    print()
    print("=== Ramp Results ===")

    # Dynamic column widths
    w_ctx = max(10, max(len(f"{s:,}") for s, _ in results))
    w_ttft = max(8, max(len(f"{r['ttft']:.2f}") for _, r in results if r["ttft"]))
    w_prefill = max(10, max(len(f"{r['prefill_speed']:,.0f}") for _, r in results if r["prefill_speed"]))
    w_decode = max(8, max(len(f"{r['decode_duration']:.2f}") for _, r in results))
    w_tpot = max(8, max(len(f"{r['tpot']:.2f}") for _, r in results))
    w_gen = max(10, max(len(f"{r['gen_speed']:.2f}") for _, r in results))
    w_wall = max(8, max(len(f"{r['wall_clock']:.2f}") for _, r in results))
    w_needle = 9
    w_scale = 8

    hdr = (
        f"  {'Context':>{w_ctx}s}  "
        f"{'TTFT':>{w_ttft}s}  "
        f"{'Prefill':>{w_prefill}s}  "
        f"{'Decode':>{w_decode}s}  "
        f"{'TPOT':>{w_tpot}s}  "
        f"{'Gen Speed':>{w_gen}s}  "
        f"{'Wall':>{w_wall}s}  "
        f"{'Needle':>{w_needle}s}  "
        f"{'Scale':>{w_scale}s}"
    )
    sep_row = (
        f"  {'-'*w_ctx}  "
        f"{'-'*w_ttft}  "
        f"{'-'*w_prefill}  "
        f"{'-'*w_decode}  "
        f"{'-'*w_tpot}  "
        f"{'-'*w_gen}  "
        f"{'-'*w_wall}  "
        f"{'-'*w_needle}  "
        f"{'-'*w_scale}"
    )
    units = (
        f"  {'(tokens)':>{w_ctx}s}  "
        f"{'(s)':>{w_ttft}s}  "
        f"{'(tok/s)':>{w_prefill}s}  "
        f"{'(s)':>{w_decode}s}  "
        f"{'(ms)':>{w_tpot}s}  "
        f"{'(tok/s)':>{w_gen}s}  "
        f"{'(s)':>{w_wall}s}  "
        f"{'':>{w_needle}s}  "
        f"{'factor':>{w_scale}s}"
    )

    print(hdr)
    print(sep_row)
    print(units)

    for i, (step, r) in enumerate(results):
        ttft_s     = f"{r['ttft']:.2f}" if r["ttft"] else "N/A"
        prefill_s  = f"{r['prefill_speed']:,.0f}" if r["prefill_speed"] else "N/A"

        badge = _needles_badge(r.get('needles_found', []))
        needle_s   = f"{badge}"

        if i > 0 and r["ttft"] and results[i-1][1]["ttft"]:
            ctx_ratio = step / results[i-1][0]
            ttft_ratio = r["ttft"] / results[i-1][1]["ttft"]
            scale = (ttft_ratio / ctx_ratio) if ctx_ratio > 0 else 0.0
            scale_s = f"{scale:.2f}"
        else:
            scale_s = "—"

        print(f"  {step:>{w_ctx},}  "
              f"{ttft_s:>{w_ttft}s}  "
              f"{prefill_s:>{w_prefill}s}  "
              f"{r['decode_duration']:>{w_decode}.2f}  "
              f"{r['tpot']:>{w_tpot}.2f}  "
              f"{r['gen_speed']:>{w_gen}.2f}  "
              f"{r['wall_clock']:>{w_wall}.2f}  "
              f"{needle_s:>{w_needle}s}  "
              f"{scale_s:>{w_scale}s}")

    print()
    print(f"  Needle badge: each emoji = one needle position (start, mid, end)")
    print("  Scaling factor: 1.0 = linear (ideal), >1.5 = quadratic-ish (degrading)")


# ============================================================
# CONCURRENCY MODES
# ============================================================

def run_concurrency(cfg: dict, context_tokens: int, max_tokens: int, timeout: int,
                    concurrency: int, saver: ResultsSaver | None, seed: int,
                    bearer_token: str = None, use_system_prompt: bool = True):
    """Run N concurrent requests with the same prompt, measure throughput."""
    user_prompt, system_prompt, metadata = build_haystack_prompt(
        context_tokens, seed=seed, use_system_prompt=use_system_prompt)
    est = estimate_tokens(user_prompt)

    print(f"Running {concurrency} concurrent requests ({est:,} est. prompt tokens each)...\n")

    results = []
    start_time = time.time()

    def _run(idx: int):
        print(f"  [{idx+1:>2}/{concurrency}] Starting... ", end="", flush=True)
        result = run_single_benchmark(cfg, user_prompt, max_tokens, timeout, metadata,
                                      bearer_token, system_prompt=system_prompt)
        if result:
            badge = _needles_badge(result.get('needles_found', []))
            print(f"TTFT={result['ttft']:.2f}s  GenSpeed={result['gen_speed']:.1f}tok/s  "
                  f"{badge}")
        else:
            print("FAILED")
        return idx, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_run, i) for i in range(concurrency)]
        for future in concurrent.futures.as_completed(futures):
            idx, result = future.result()
            results.append((idx, result))

    wall_clock = time.time() - start_time
    results.sort(key=lambda x: x[0])

    # Per-request detail
    print()
    successful = [r for _, r in results if r is not None]
    failed = [r for _, r in results if r is None]

    for idx, r in results:
        if r:
            print(_format_result_line(idx + 1, r))
        else:
            print(f"  Request {idx+1:>2}: ❌ FAILED")

    # Summary
    print()
    if successful:
        avg_ttft = sum(r['ttft'] for r in successful) / len(successful)
        avg_gen = sum(r['gen_speed'] for r in successful) / len(successful)
        total_gen = sum(r['completion_tokens'] for r in successful)
        throughput = total_gen / wall_clock if wall_clock > 0 else 0

        # Aggregate needle results
        total_needles_found = sum(r['needle_count'] for r in successful)
        max_possible = len(successful) * 3
        perfect_runs = sum(1 for r in successful if r['needle_count'] == 3)

        print("  Step Summary:")
        print(f"    Wall clock: {wall_clock:.1f}s  |  Throughput: {throughput:.1f} tok/s")
        print(f"    Needle: {total_needles_found}/{max_possible} total  ({perfect_runs}/{len(successful)} perfect)")
        print(f"    Avg TTFT: {avg_ttft:.2f}s  |  Avg GenSpeed: {avg_gen:.1f} tok/s")

        # Aggregate coherency issues
        loop_count = sum(1 for r in successful if r.get("is_looping"))
        len_issues = sum(1 for r in successful if not r.get("length_ok", True))
        coh_problems = []
        if loop_count:
            coh_problems.append(f"{loop_count} looping")
        if len_issues:
            coh_problems.append(f"{len_issues} length")
        print(f"    Coherency: {'clean' if not coh_problems else ', '.join(coh_problems)}")

    if failed:
        print(f"  Failed: {len(failed)}/{len(results)}")

    if saver:
        for idx, result in results:
            if result:
                saver.save_run(f"concurrent-{idx+1}", result, result.get("collected_text", ""))

    print()


def run_concurrency_ramp(cfg: dict, context_tokens: int, max_tokens: int, timeout: int,
                         max_concurrency: int, saver: ResultsSaver | None, seed: int,
                         bearer_token: str = None, use_system_prompt: bool = True):
    """
    Ramp concurrency at a fixed context size.
    Powers of 2 from 1 to max_concurrency, each step runs N concurrent requests.
    """
    # Generate concurrency ramp: 1, 2, 4, 8, ... up to max_concurrency
    concurrency_levels = []
    current = 1
    while current <= max_concurrency:
        concurrency_levels.append(current)
        current *= 2
    if not concurrency_levels:
        concurrency_levels = [max_concurrency]

    # Build the prompt once (fixed context)
    user_prompt, system_prompt, metadata = build_haystack_prompt(
        context_tokens, seed=seed, use_system_prompt=use_system_prompt)
    est = estimate_tokens(user_prompt)

    sep = "─" * 60

    print(f"Concurrency Ramp: {' → '.join(str(c) for c in concurrency_levels)}")
    print(f"Context: {context_tokens:,} tokens ({est:,} est. prompt tokens)\n")

    # Warmup: one throwaway C=1 request to build the KV cache
    print("  Warming up KV cache...")
    warmup = run_single_benchmark(cfg, user_prompt, max_tokens, timeout, metadata,
                                  bearer_token, system_prompt=system_prompt)
    if warmup:
        badge = _needles_badge(warmup.get('needles_found', []))
        print(f"  Warmup: TTFT={warmup['ttft']:.2f}s  GenSpeed={warmup['gen_speed']:.1f}tok/s  {badge}")
    else:
        print("  Warmup: FAILED (continuing anyway)")
    print()

    all_results = []

    for workers in concurrency_levels:
        print(sep)
        print(f"  Concurrency: {workers}")
        print(sep)

        step_results = []
        start_time = time.time()

        def _run(idx: int):
            print(f"  [{idx+1:>2}/{workers}] Starting... ", end="", flush=True)
            result = run_single_benchmark(cfg, user_prompt, max_tokens, timeout, metadata,
                                          bearer_token, system_prompt=system_prompt)
            if result:
                badge = _needles_badge(result.get('needles_found', []))
                print(f"TTFT={result['ttft']:.2f}s  GenSpeed={result['gen_speed']:.1f}tok/s  {badge}")
            else:
                print("FAILED")
            return idx, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run, i) for i in range(workers)]
            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                step_results.append((idx, result))

        wall_clock = time.time() - start_time
        step_results.sort(key=lambda x: x[0])
        successful = [r for _, r in step_results if r is not None]
        failed = [r for _, r in step_results if r is None]

        # Per-request detail
        for idx, r in step_results:
            if r:
                print(_format_result_line(idx + 1, r))
            else:
                print(f"  Request {idx+1:>2}: ❌ FAILED")

        # Step summary
        print()
        if successful:
            avg_ttft = sum(r['ttft'] for r in successful) / len(successful)
            avg_gen = sum(r['gen_speed'] for r in successful) / len(successful)
            total_gen = sum(r['completion_tokens'] for r in successful)
            throughput = total_gen / wall_clock if wall_clock > 0 else 0

            # Needle aggregation
            total_needles_found = sum(r['needle_count'] for r in successful)
            max_possible = len(successful) * 3
            perfect_runs = sum(1 for r in successful if r['needle_count'] == 3)

            print("  Step Summary:")
            print(f"    Wall clock: {wall_clock:.1f}s  |  Throughput: {throughput:.1f} tok/s")
            print(f"    Needle: {total_needles_found}/{max_possible} total  ({perfect_runs}/{len(successful)} perfect)")
            print(f"    Avg TTFT: {avg_ttft:.2f}s  |  Avg GenSpeed: {avg_gen:.1f} tok/s")

            # Coherency aggregation
            loop_count = sum(1 for r in successful if r.get("is_looping"))
            len_issues = sum(1 for r in successful if not r.get("length_ok", True))
            coh_problems = []
            if loop_count:
                coh_problems.append(f"{loop_count} looping")
            if len_issues:
                coh_problems.append(f"{len_issues} length")
            print(f"    Coherency: {'clean' if not coh_problems else ', '.join(coh_problems)}")

            all_results.append((workers, step_results, wall_clock, throughput))

            if saver:
                for idx, result in step_results:
                    if result:
                        saver.save_run(f"ramp-w{workers}-r{idx+1}", result,
                                       result.get("collected_text", ""))

            # Degradation vs single-request baseline
            if len(all_results) > 1:
                baseline_gen = all_results[0][3]  # first level's throughput
                degradation = ((baseline_gen - avg_gen) / baseline_gen * 100) if baseline_gen > 0 else 0
                print(f"    Degradation: +{degradation:.1f}% vs single-request baseline")

        if failed:
            print(f"  Failed: {len(failed)}/{len(step_results)}")

        print()

    # Final summary table
    if all_results:
        print(sep)
        print("  CONCURRENCY RAMP SUMMARY")
        print(sep)

        # Compute aggregates per level
        rows = []
        for workers, step_results, wall_clock, throughput in all_results:
            successful = [r for _, r in step_results if r is not None]
            avg_ttft = sum(r['ttft'] for r in successful) / len(successful)
            avg_gen = sum(r['gen_speed'] for r in successful) / len(successful)
            total_gen = sum(r['completion_tokens'] for r in successful)
            total_needles_found = sum(r['needle_count'] for r in successful)
            perfect_runs = sum(1 for r in successful if r['needle_count'] == 3)

            # Coherency aggregation
            loop_count = sum(1 for r in successful if r.get("is_looping"))
            len_issues = sum(1 for r in successful if not r.get("length_ok", True))

            rows.append({
                'workers': workers,
                'avg_ttft': avg_ttft,
                'avg_gen': avg_gen,
                'total_gen': total_gen,
                'throughput': throughput,
                'needle_total': total_needles_found,
                'perfect_runs': perfect_runs,
                'total': len(successful),
                'coh_issues': loop_count + len_issues,
            })

        # Degradation vs single-request baseline
        baseline_gen = rows[0]['avg_gen'] if rows else 0

        # Column widths
        w_workers = max(10, max(len(str(r['workers'])) for r in rows))
        w_ttft = max(10, max(len(f"{r['avg_ttft']:.2f}") for r in rows))
        w_gen = max(12, max(len(f"{r['avg_gen']:.1f}") for r in rows))
        w_throughput = max(16, max(len(f"{r['throughput']:.1f}") for r in rows))
        w_degradation = max(14, 14)
        w_needle = max(8, max(len(f"{r['perfect_runs']}/{r['total']}") for r in rows))

        # Header
        hdr = (
            f"  {'Concurrency':>{w_workers}s}  "
            f"{'Avg TTFT':>{w_ttft}s}  "
            f"{'Avg GenSpeed':>{w_gen}s}  "
            f"{'Total Throughput':>{w_throughput}s}  "
            f"{'Degradation':>{w_degradation}s}  "
            f"{'Perfect':>{w_needle}s}"
        )
        sep_row = (
            f"  {'-'*w_workers}  "
            f"{'-'*w_ttft}  "
            f"{'-'*w_gen}  "
            f"{'-'*w_throughput}  "
            f"{'-'*w_degradation}  "
            f"{'-'*w_needle}"
        )
        units = (
            f"  {'':>{w_workers}s}  "
            f"{'(s)':>{w_ttft}s}  "
            f"{'(tok/s)':>{w_gen}s}  "
            f"{'(tok/s)':>{w_throughput}s}  "
            f"{'':>{w_degradation}s}  "
            f"{'of 3/3':>{w_needle}s}"
        )

        print(hdr)
        print(sep_row)
        print(units)

        for r in rows:
            degradation = ((baseline_gen - r['avg_gen']) / baseline_gen * 100) if baseline_gen > 0 else 0
            deg_str = f"+{degradation:.1f}%"
            needle_s = f"{r['perfect_runs']}/{r['total']}"
            row = (
                f"  {r['workers']:>{w_workers}d}  "
                f"{r['avg_ttft']:>{w_ttft}.2f}s  "
                f"{r['avg_gen']:>{w_gen}.1f}  "
                f"{r['throughput']:>{w_throughput}.1f}  "
                f"{deg_str:>{w_degradation}s}  "
                f"{needle_s:>{w_needle}s}"
            )
            print(row)

        # Insight lines
        print()
        total_perfect = sum(r['perfect_runs'] for r in rows)
        total_reqs = sum(r['total'] for r in rows)
        pass_rate = f"{total_perfect}/{total_reqs} ({total_perfect/total_reqs*100:.0f}%)" if total_reqs else "N/A"

        best_idx = max(range(len(rows)), key=lambda i: rows[i]['throughput'])
        best_row = rows[best_idx]
        best_tp = f"{best_row['throughput']:.1f} tok/s (at concurrency {best_row['workers']})"

        max_deg = ((baseline_gen - rows[-1]['avg_gen']) / baseline_gen * 100) if baseline_gen > 0 else 0
        max_deg_str = f"{max_deg:.1f}% ({baseline_gen:.1f} → {rows[-1]['avg_gen']:.1f} tok/s)"

        # Recommended concurrency: best throughput-to-degradation tradeoff
        recommended = rows[-1]['workers']  # default: max
        for r in rows:
            deg = ((baseline_gen - r['avg_gen']) / baseline_gen * 100) if baseline_gen > 0 else 0
            if deg <= 30:
                recommended = r['workers']
            else:
                break

        # Coherency summary across all levels
        total_coh_issues = sum(r['coh_issues'] for r in rows)

        print(f"  Needle Pass Rate:               {pass_rate}")
        print(f"  Best Total Throughput:          {best_tp}")
        print(f"  Degradation at max:             {max_deg_str}")
        print(f"  Recommended concurrency:        {recommended} (best throughput-to-degradation tradeoff)")
        if total_coh_issues > 0:
            print(f"  Coherency issues across all:    {total_coh_issues} requests had problems")
        else:
            print(f"  Coherency:                      clean across all levels")
        print(sep)
        print()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="contextllens-v2: Enhanced LLM long-context & concurrency benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  single  — One-shot benchmark (default)\n"
            "  warm    — Cold + warm run comparison (KV cache effect)\n"
            "  ramp    — Growing context benchmark (powers of 2 from 1K to target)\n"
            "  concurrency=N — N concurrent requests at once\n"
            "  concurrency-ramp=N — Scaling concurrency: 1→2→4→...→N\n"
            "\n"
            "Examples:\n"
            "  python3 contextllens-v2.py --model qwen/qwen3.6-35b-a3b-mlx\n"
            "  python3 contextllens-v2.py --model qwen/qwen3.6-35b-a3b-mlx --mode warm\n"
            "  python3 contextllens-v2.py --model qwen/qwen3.6-35b-a3b-mlx --mode ramp --context-tokens 32000\n"
            "  python3 contextllens-v2.py --model qwen/qwen3.6-35b-a3b-mlx --mode concurrency=4\n"
            "  python3 contextllens-v2.py --model qwen/qwen3.6-35b-a3b-mlx --mode concurrency-ramp=16\n"
        ),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model identifier (see --list-models).",
    )
    parser.add_argument(
        "--mode", type=str, default="single",
        help="Benchmark mode. Use 'single', 'warm', 'ramp', 'concurrency=N', or 'concurrency-ramp=N' (default: single)",
    )
    parser.add_argument(
        "--context-tokens", type=int, default=32000,
        help="Target prompt tokens (default: 32000). In ramp mode, this is the max.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1500,
        help="Max tokens for the model to generate (default: 1500).",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Request timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: config.yaml in script directory).",
    )
    parser.add_argument(
        "--results-path", type=str, default="./results",
        help="Directory to save results (default: ./results).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Disable saving results to disk.",
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Free-form notes for this run (saved to notes.txt in results folder).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible needle selection (default: random).",
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Skip the system prompt role. By default, a system prompt is sent as the first message.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available model identifiers and exit.",
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config or find_config()
    MODEL_REGISTRY = load_config(config_path)

    if args.list_models:
        print(f"Config: {config_path}")
        print("Available models:")
        for key, cfg in MODEL_REGISTRY.items():
            print(f"  {key:<30s}  {cfg['label']}  ({cfg['context']:,} tok context)")
        sys.exit(0)

    if not args.model:
        parser.error("--model is required (or use --list-models).")

    cfg = MODEL_REGISTRY.get(args.model)
    if not cfg:
        print(f"Unknown model: {args.model}")
        print("Use --list-models to see available models.")
        sys.exit(1)

    # Extract bearer token from config if present
    bearer_token = cfg.get("bearer_token")

    # Parse mode with proper validation
    concurrency = None
    concurrency_ramp_max = None
    mode = args.mode

    if mode.startswith("concurrency-ramp="):
        try:
            concurrency_ramp_max = int(mode.split("=", 1)[1])
        except ValueError:
            print(f"Error: concurrency-ramp requires a number, got: {mode}")
            sys.exit(1)
        if concurrency_ramp_max < 1:
            print("Error: concurrency-ramp must be >= 1")
            sys.exit(1)
    elif mode.startswith("concurrency="):
        try:
            concurrency = int(mode.split("=", 1)[1])
        except ValueError:
            print(f"Error: concurrency requires a number, got: {mode}")
            sys.exit(1)
        if concurrency < 1:
            print("Error: concurrency must be >= 1")
            sys.exit(1)
    elif mode not in ("single", "warm", "ramp"):
        print(f"Error: unknown mode '{mode}'. Use 'single', 'warm', 'ramp', 'concurrency=N', or 'concurrency-ramp=N'.")
        sys.exit(1)

    # Setup results saver
    saver = None
    if not args.no_save:
        saver = ResultsSaver(
            results_path=args.results_path,
            model_key=args.model,
            mode=mode,
            context_tokens=args.context_tokens,
            cfg=cfg,
            notes=args.notes,
        )

    # Capture all terminal output using TeeWriter
    tee = TeeWriter(sys.stdout)
    original_stdout = sys.stdout
    sys.stdout = tee

    # Resolve system prompt flag early so it's available for header and all modes
    use_system_prompt = not args.no_system_prompt

    try:
        # Header
        print(f"{'='*60}")
        print(f"  Model:              {cfg['label']}")
        print(f"  Endpoint:           {cfg['endpoint']}")
        print(f"  Mode:               {mode}")
        print(f"  Target Context:     {args.context_tokens:,} tokens")
        print(f"  Max Gen Tokens:     {args.max_tokens:,}")
        print(f"  System Prompt:      {'enabled' if use_system_prompt else 'disabled (--no-system-prompt)'}")
        if args.seed is not None:
            print(f"  Random Seed:          {args.seed}")
        if saver:
            print(f"  Results Path:       {saver.run_dir}")
        if args.notes:
            print(f"  Notes:              {args.notes}")
        print(f"{'='*60}")
        if mode == "single":
            run_single(cfg, args.context_tokens, args.max_tokens, args.timeout,
                       saver, args.seed, bearer_token,
                       use_system_prompt=use_system_prompt)
        elif mode == "warm":
            run_warm(cfg, args.context_tokens, args.max_tokens, args.timeout,
                     saver, args.seed, bearer_token,
                     use_system_prompt=use_system_prompt)
        elif mode == "ramp":
            run_ramp(cfg, args.context_tokens, args.max_tokens, args.timeout,
                     saver, args.seed, bearer_token,
                     use_system_prompt=use_system_prompt)
        elif concurrency is not None:
            run_concurrency(cfg, args.context_tokens, args.max_tokens, args.timeout,
                            concurrency, saver, args.seed, bearer_token,
                            use_system_prompt=use_system_prompt)
        elif concurrency_ramp_max is not None:
            run_concurrency_ramp(cfg, args.context_tokens, args.max_tokens, args.timeout,
                                 concurrency_ramp_max, saver, args.seed, bearer_token,
                                 use_system_prompt=use_system_prompt)

    finally:
        sys.stdout = original_stdout

    # Finalize results
    if saver:
        terminal_text = tee.getvalue()
        run_dir = saver.finalize(terminal_text)
        print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
