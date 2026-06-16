#!/usr/bin/env python3
"""
contextllens.py — Unified LLM benchmark script.

Modes:
  single  (default) — One-shot benchmark with needle-in-haystack prompt
  warm    — Cold + warm run comparison (KV cache effect)
  ramp    — Growing context benchmark (powers of 2 from 1K to target)
  concurrency=N — N concurrent requests at once (default: concurrency=2)
  concurrency-ramp=N — Scaling concurrency: 1→2→4→...→N

The prompt uses a needle-in-haystack approach: varied technical content
with a hidden fact embedded at ~25% depth, followed by a query asking
the model to retrieve it. This tests both context handling and retrieval.

Configuration:
  Copy config.yaml.example to config.yaml and edit with your endpoints.
  Or specify a custom config with --config.

Results:
  Results are saved by default to ./results/. Use --results-path to override.
  Each run creates a timestamped subfolder with results.txt, results.csv,
  results.json, and an Output/ folder with full LLM responses.

Usage:
    python3 contextllens.py --model qwen/qwen3.6-27b
    python3 contextllens.py --model qwen/qwen3.6-27b-mlx --mode warm
    python3 contextllens.py --model qwen/qwen3.6-27b-mlx --mode ramp --context-tokens 32000
    python3 contextllens.py --model qwen/qwen3.6-27b --mode concurrency=4
    python3 contextllens.py --model qwen/qwen3.6-27b --mode concurrency-ramp=16
    python3 contextllens.py --list-models
"""

__version__ = "0.3.0"

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

# ============================================================
# CONFIG LOADING
# ============================================================
def load_config(config_path: str) -> dict:
    """Load model registry from YAML config file."""
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
    """Find config file: --config > config.yaml > config.yaml.example"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for name in ["config.yaml", "config.yaml.example"]:
        path = os.path.join(script_dir, name)
        if os.path.exists(path):
            return path

    return os.path.join(script_dir, "config.yaml")


# ============================================================
# RESULTS SAVING
# ============================================================
class ResultsSaver:
    """Captures terminal output and saves benchmark results to disk."""

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
        self.runs = []  # list of (step_label, metrics_dict, collected_text)

        # Sanitize model key for filenames
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', model_key)
        safe_key = safe_key.strip('_')

        # Create run folder: YYYY-MM-DD_HH-MM_<Model-Name>_<Mode>_<Context-Tokens>
        time_str = self.timestamp.strftime("%Y-%m-%d_%H-%M")
        mode_label = mode.upper()
        folder_name = f"{time_str}_{safe_key}_{mode_label}_{context_tokens}"
        self.run_dir = os.path.join(results_path, folder_name)
        self.output_dir = os.path.join(self.run_dir, "Output")

        os.makedirs(self.output_dir, exist_ok=True)

    def save_run(self, step_label: str, metrics: dict, collected_text: str):
        """Record a single benchmark run."""
        self.runs.append((step_label, metrics, collected_text))

        # Save individual output file: <model-name>_<step-label>.txt
        if collected_text:
            safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', step_label)
            safe_label = safe_label.strip('_')
            safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', self.model_key)
            safe_key = safe_key.strip('_')
            filename = f"{safe_key}_{safe_label}.txt"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(collected_text)

    def finalize(self, terminal_output: str):
        """Write results.txt, results.csv, and results.json."""
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
            # Remove collected_text from JSON (too large)
            entry.pop("collected_text", None)
            json_data["runs"].append(entry)

        json_path = os.path.join(self.run_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        # results.csv — tabular metrics
        csv_path = os.path.join(self.run_dir, "results.csv")
        if self.runs:
            fieldnames = ["step", "prompt_tokens", "completion_tokens",
                          "total_tokens", "ttft", "prefill_speed",
                          "decode_duration", "gen_speed", "tpot",
                          "wall_clock", "needle_found"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for step_label, metrics, _ in self.runs:
                    row = {"step": step_label}
                    row.update(metrics)
                    writer.writerow(row)

        return self.run_dir


# ============================================================
# NEEDLE-IN-HAYSTACK PROMPT GENERATION
# ============================================================
HAYSTACK_PARAGRAPHS = [
    "Meeting notes from the infrastructure team standup: discussed migration plans for the Kubernetes cluster, reviewed incident response procedures, and assigned action items for the Q3 reliability sprint. The team agreed to prioritize database sharding before the holiday season to handle projected 3x traffic growth.",

    "Code review feedback on PR #4821: The new caching layer implementation looks solid. Consider adding circuit breaker patterns for the external API calls to improve resilience under partial failure conditions. Also, the retry logic should use exponential backoff with jitter to avoid thundering herd problems during recovery.",

    "Architecture decision record ADR-047: We will adopt event sourcing for the order management system. This provides an immutable audit trail and simplifies debugging of complex state transitions. The CQRS pattern will separate read and write concerns, allowing independent scaling of query and command handlers.",

    "Incident report INC-2024-0892: Database connection pool exhaustion caused a 15-minute outage during peak traffic on Black Friday. Root cause: unbounded retry loop in the payment service that amplified load during the PostgreSQL failover. Post-incident fix implemented adaptive retry with circuit breaker and connection pool limits.",

    "Performance analysis of the search pipeline: Elasticsearch query latency increased from 45ms to 230ms after the schema migration to nested objects. The new structure requires additional join operations for each query. Profiling shows the bottleneck is in the fielddata cache, which is consuming 12GB of heap memory per node.",

    "Security audit findings for the authentication service: JWT token validation is missing issuer verification, allowing tokens from any provider to be accepted. Recommend adding audience claims and implementing token rotation with a 15-minute grace period. Additionally, the password hashing algorithm should be upgraded from bcrypt to argon2id.",

    "Deployment notes for release v3.14.2: Rolling update completed successfully across all 12 availability zones with zero downtime. The new blue-green deployment strategy reduced rollout time from 45 minutes to 8 minutes. Canary analysis showed error rates below 0.1% threshold, triggering automatic full promotion.",

    "Database migration plan for the user profiles table: Adding composite indexes on (tenant_id, created_at) will improve query performance by approximately 40% based on staging benchmarks. Estimated migration time is 45 minutes during the maintenance window. Online DDL operations will be used to avoid table locks.",

    "API design review for the new GraphQL endpoint: The subscription schema looks good for real-time notifications. Consider adding rate limiting per client ID to prevent abuse of real-time data streams. The N+1 query problem in the resolver chain should be addressed with DataLoader batching before the production launch.",

    "Monitoring alert configuration update: Added new Prometheus rules for detecting memory leaks in the sidecar containers. Alert threshold set at 85% memory utilization sustained for 5 consecutive minutes. PagerDuty integration configured with auto-resolution after 30 minutes of normal readings to reduce alert fatigue.",

    "Load testing results for the message queue: RabbitMQ cluster handled 50,000 messages per second with 99.9th percentile latency of 12ms. Consumer group rebalancing took 3.2 seconds during scale-up events. The new dead letter queue configuration successfully captured 99.7% of malformed messages for manual inspection.",

    "Container orchestration update: Migrated from Docker Swarm to Kubernetes for better resource management and self-healing capabilities. The new Helm charts include automated canary deployments with Istio traffic splitting. Resource quotas are configured per namespace to prevent noisy neighbor problems in the shared cluster.",

    "Data pipeline optimization: Replaced the hourly batch ETL jobs with real-time Kafka streams for the analytics platform. This reduced data freshness from 1 hour to under 30 seconds for the executive dashboard. The Flink job handles 2 million events per minute with exactly-once semantics using checkpointing.",

    "Network architecture review: The service mesh implementation using Linkerd reduced inter-service latency by 15% through optimized routing. mTLS encryption is now enabled for all east-west traffic within the cluster. The new DNS-based service discovery eliminated hardcoded endpoint configurations across 47 microservices.",

    "Backup and disaster recovery plan: Implemented cross-region replication for the PostgreSQL primary database using logical decoding. RTO target is 15 minutes, RPO target is 5 minutes based on WAL shipping interval. Quarterly failover drills confirm the runbook is effective, with the last drill completing in 12 minutes.",

    "The Battle of Hastings was fought on October 14, 1066, between the Norman-French army of William, the Duke of Normandy, and the English army of the Anglo-Saxon King Harold Godwinson. Beginning at 9 AM, the battle lasted until late afternoon, with Harold ultimately killed and the English army defeated. This decisive Norman victory had profound consequences for English history, leading to significant cultural, linguistic, and political changes.",

    "A classic recipe for sourdough bread begins with a naturally leavened starter culture that has been maintained for months or even years. The process involves mixing flour and water, allowing wild yeast and lactobacilli to ferment the dough over 12-18 hours, then folding and shaping before a final proof. Baking in a preheated Dutch oven at 475°F (246°C) creates the characteristic crust and crumb structure that distinguishes artisan sourdough from commercial yeast breads.",

    "The Fibonacci sequence is defined by the recurrence relation F(n) = F(n-1) + F(n-2), with base cases F(0) = 0 and F(1) = 1. The first twenty terms are: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181. The ratio of consecutive Fibonacci numbers converges to the golden ratio φ = (1 + √5) / 2 ≈ 1.618033988749895, a relationship discovered by Kepler and widely observed in phyllotaxis and spiral patterns.",

    "Proof that the square root of 2 is irrational: Assume √2 = a/b where a and b are coprime integers. Then 2 = a²/b², so a² = 2b². This means a² is even, so a must be even (since the square of an odd number is odd). Let a = 2k. Then 4k² = 2b², so b² = 2k², meaning b² is even and b is even. But this contradicts the assumption that a and b are coprime. Therefore √2 is irrational.",

    "Alice: 'Do you know why a raven is like a writing-desk?'\n    Hummingbird: 'No.'\n    Alice: 'Neither do I.'\n    Hummingbird: 'I haven't the faintest idea.'\n    Alice: 'Same here.'\n    Hummingbird: 'That's because there is no answer. It was a joke, you see.'\n    Alice: 'I thought as much.'\n    Hummingbird: 'Would you like some tea?'\n    Alice: 'I should love to.'\n    Hummingbird: 'There's hardly any sort of tea besides nettle-tea and dandelion-tea. But we've got mint-tea and chamomile-tea.'",

    "The periodic table of elements organizes all known chemical substances by increasing atomic number. As of 2024, there are 118 confirmed elements, with the most recent additions being nihonium (113), moscovium (115), tennessine (117), and oganesson (118), all synthesized in particle accelerators. The table's structure reflects electron configurations, with periods corresponding to electron shell fills and groups sharing valence electron patterns that determine chemical reactivity.",

    "Apollo 11 was the spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle at Tranquility Base on July 20, 1969, at 20:17 UTC. Armstrong became the first person to step onto the lunar surface 6 hours later at 02:56 UTC on July 21, while Aldrin joined him 19 minutes later. They spent about two and a quarter hours together outside the spacecraft, collecting 47.5 pounds (21.5 kg) of lunar material for return to Earth.",

    "The Japanese tea ceremony, known as chanoyu or chadō, is a traditional ritual centered around the preparation and serving of matcha, a finely ground powdered green tea. Rooted in Zen Buddhism, the ceremony emphasizes harmony (和), respect (敬), purity (清), and tranquility (寂). Each movement is deliberate and choreographed, from the cleansing of utensils to the whisking of the tea, embodying the principle of ichigo ichie — one time, one meeting — reminding participants to treasure each encounter as unique and unrepeatable.",

    "The Great Barrier Reef is the world's largest coral reef system, stretching over 2,300 kilometers along the northeast coast of Australia. Comprising over 2,900 individual reefs and 900 islands, it is visible from space and home to an extraordinary diversity of life including 1,500 species of fish, 400 types of coral, and numerous species of mollusk, sponge, and sea star. Despite its ecological significance, the reef faces severe threats from climate change, ocean acidification, and pollution.",

    "The Fibonacci sequence appears throughout nature in unexpected ways. The arrangement of leaves on a stem (phyllotaxis) often follows Fibonacci numbers — sunflower seed heads display spirals in both directions that are consecutive Fibonacci numbers (typically 34 and 55, or 55 and 89). Pinecones, pineapples, and artichokes exhibit similar patterns. The nautilus shell grows in a logarithmic spiral that approximates the golden spiral derived from the Fibonacci sequence, creating a self-similar growth pattern found across scales in biology.",

    "VERTEX-7143-ALPHA: Previous staging code for the legacy authentication gateway. Decommissioned after migration to the unified identity platform. All dependent services were updated to use the new endpoint. No active deployments reference this code.",

    "VERTEX-7739-ECHO: Emergency rollback procedure for the payment processing pipeline. Temporary measure activated during an outage incident. Superseded by the permanent fix. This code path is no longer used in any environment.",

    "VERTEX-5512-BRAVO: Development environment configuration override for local testing. Used only in developer workstations and CI sandbox environments. Not applicable to any production or staging deployment. Has no effect on live systems.",

    "VERTEX-6621-GAMMA: Original reference implementation for the metrics collection agent. Superseded by the v2 agent. The legacy agent is still installed on three legacy servers but receives no updates and is scheduled for removal.",

    "VERTEX-4498-CHARLIE: Staging environment service mesh configuration. Mirrors production topology but with relaxed rate limits and debug logging enabled. Used for pre-deployment validation and performance testing. Not a production artifact.",
]

# The hidden fact embedded in the haystack
NEEDLE = (
    "CRITICAL REFERENCE: Effective immediately, the deployment authorization code for the production "
    "environment is VERTEX-9284-KILO. This code must be included in all "
    "production deployment requests and verified by the on-call engineer."
)

# The query asking the model to find the needle and generate a long response
QUERY = (
    "What is the current production deployment authorization code mentioned in the document above? "
    "After providing the code, compile a formal infrastructure status report "
    "based on the provided meeting notes, incident reports, and performance analyses. "
    "The report should be detailed and approximately 1000 words long."
)


def check_needle(text: str) -> str:
    """Return 'exact', 'substitution', or 'not_found' for needle retrieval."""
    raw = "VERTEX-9284-KILO"
    if raw in text:
        return "exact"
    # Normalize common Unicode lookalikes to ASCII
    normalized = (
        text
        .replace("\u2011", "-")  # non-breaking hyphen
        .replace("\u2013", "-")  # en-dash
        .replace("\u2014", "-")  # em-dash
        .replace("\u00ad", "")   # soft hyphen (zero-width)
    )
    if raw in normalized:
        return "substitution"
    return "not_found"


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~1.3 chars/token for English text."""
    return max(1, int(len(text) / 1.3))


def build_haystack_prompt(target_tokens: int, seed: int = None) -> str:
    """
    Build a varied haystack with the needle embedded at ~25% position.
    Cycles through HAYSTACK_PARAGRAPHS (optionally shuffled by seed) to fill
    the target token count.
    """
    import random
    paragraphs_pool = list(HAYSTACK_PARAGRAPHS)
    if seed is not None:
        random.seed(seed)
        random.shuffle(paragraphs_pool)

    needle_position = max(200, int(target_tokens * 0.25))

    paragraphs = []
    current_tokens = 0
    needle_inserted = False
    idx = 0

    while current_tokens < target_tokens:
        para = paragraphs_pool[idx % len(paragraphs_pool)]
        para_tokens = estimate_tokens(para)

        if not needle_inserted and current_tokens >= needle_position:
            paragraphs.append(NEEDLE)
            current_tokens += estimate_tokens(NEEDLE)
            needle_inserted = True

        paragraphs.append(para)
        current_tokens += para_tokens
        idx += 1

    if not needle_inserted:
        paragraphs.insert(1, NEEDLE)

    prompt = "\n\n".join(paragraphs) + "\n\n" + QUERY
    return prompt


# ============================================================
# BENCHMARK RUNNER
# ============================================================
def run_single_benchmark(cfg: dict, prompt: str, max_tokens: int, timeout: int,
                        bearer_token: str = None) -> dict | None:
    """
    Run a single benchmark request. Returns a metrics dict, or None on failure.
    bearer_token: optional Bearer token for API authentication.
    """
    endpoint = cfg["endpoint"]
    model    = cfg["model"]

    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": 0.0,
        "stream":      True,
    }

    headers = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    start_time = time.time()

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
    collected_text  = ""

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
            token_text = delta.get("content") or delta.get("reasoning_content")

            if token_text:
                collected_text += token_text
                delta_token_cnt += 1

                if first_token_ts is None:
                    first_token_ts = time.time()
                    ttft = first_token_ts - start_time

                last_token_ts = time.time()

    end_time = time.time()

    # ---- resolve token counts ----
    est_prompt = estimate_tokens(prompt)
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
        decode_duration = end_time - start_time

    gen_speed     = (completion_tokens / decode_duration) if (completion_tokens > 1 and decode_duration > 0) else 0.0
    prefill_speed = (prompt_tokens / ttft) if (ttft and prompt_tokens > 0) else 0.0
    tpot          = (decode_duration / completion_tokens * 1000) if (completion_tokens > 0) else 0.0
    needle_result = check_needle(collected_text)
    needle_found  = needle_result != "not_found"

    return {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
        "ttft":              ttft,
        "prefill_speed":     prefill_speed,
        "decode_duration":   decode_duration,
        "gen_speed":         gen_speed,
        "tpot":              tpot,
        "wall_clock":        end_time - start_time,
        "collected_text":    collected_text,
        "needle_found":      needle_found,
        "needle_result":     needle_result,
    }


def _needle_icon(result: str) -> str:
    """Return emoji icon for needle result."""
    if result == "exact":
        return "✅"
    elif result == "substitution":
        return "⚠️"
    return "❌"


def _format_result_line(idx: int, r: dict) -> str:
    """Format a single request result as a display line."""
    icon = _needle_icon(r['needle_result'])
    return (
        f"  Request {idx:>2}: {icon}  "
        f"TTFT={r['ttft']:.2f}s  "
        f"Prefill={r['prefill_speed']:,.0f}tok/s  "
        f"GenSpeed={r['gen_speed']:.1f}tok/s  "
        f"TPOT={r['tpot']:.2f}ms  "
        f"Wall={r['wall_clock']:.1f}s"
    )


def print_results(results: dict, label: str = "", show_preview: bool = True):
    """Print formatted benchmark results in compact style."""
    sep = "─" * 60
    if label:
        print(sep)
        print(f"  {label}")
        print(sep)

    print(_format_result_line(1, results))

    if show_preview and results["collected_text"]:
        snippet = results["collected_text"][:300]
        print(f"\n  Output preview: {snippet!r}")
        if len(results["collected_text"]) > 300:
            print(f"  ... ({len(results['collected_text'])} chars total)")
    print()


# ============================================================
# BENCHMARK MODES
# ============================================================
def run_single(cfg: dict, context_tokens: int, max_tokens: int, timeout: int,
               saver: ResultsSaver | None, bearer_token: str = None):
    """One-shot benchmark with needle-in-haystack prompt."""
    prompt = build_haystack_prompt(context_tokens)
    est = estimate_tokens(prompt)
    print(f"Sending request ({est:,} est. prompt tokens)...")
    results = run_single_benchmark(cfg, prompt, max_tokens, timeout, bearer_token)
    if results:
        print_results(results)
        if saver:
            saver.save_run("single", results, results.get("collected_text", ""))


def run_warm(cfg: dict, context_tokens: int, max_tokens: int, timeout: int,
             saver: ResultsSaver | None, bearer_token: str = None):
    """Cold + warm run comparison to measure KV cache effect."""
    prompt = build_haystack_prompt(context_tokens)
    est = estimate_tokens(prompt)

    print(f"Running COLD benchmark ({est:,} est. prompt tokens)...")
    cold = run_single_benchmark(cfg, prompt, max_tokens, timeout, bearer_token)
    if not cold:
        return

    print(f"Running WARM benchmark (same prompt, KV cache should be warm)...")
    warm = run_single_benchmark(cfg, prompt, max_tokens, timeout, bearer_token)
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

    cold_nr = cold.get("needle_result", "not_found")
    warm_nr = warm.get("needle_result", "not_found")
    print(f"    Cold Needle:            {_needle_icon(cold_nr)}")
    print(f"    Warm Needle:            {_needle_icon(warm_nr)}")
    print()

    if saver:
        saver.save_run("cold", cold, cold.get("collected_text", ""))
        saver.save_run("warm", warm, warm.get("collected_text", ""))


def run_ramp(cfg: dict, max_context_tokens: int, max_tokens: int, timeout: int,
             saver: ResultsSaver | None, bearer_token: str = None):
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
        prompt = build_haystack_prompt(step)
        result = run_single_benchmark(cfg, prompt, max_tokens, timeout, bearer_token)
        if result:
            results.append((step, result))
            print(f"TTFT={result['ttft']:.2f}s  Prefill={result['prefill_speed']:,.0f}tok/s  "
                  f"GenSpeed={result['gen_speed']:.1f}tok/s  TPOT={result['tpot']:.2f}ms  "
                  f"Wall={result['wall_clock']:.1f}s  {_needle_icon(result['needle_result'])}")
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
    w_needle = max(8, max(len(_needle_icon(r['needle_result'])) for _, r in results))
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
        needle_s   = _needle_icon(r['needle_result'])

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
    print("  Needle: ✅ Exact match    ⚠️ Found with char substitution    ❌ Not found")
    print("  Scaling factor: 1.0 = linear (ideal), >1.5 = quadratic-ish (degrading)")


# ============================================================
# CONCURRENCY MODES
# ============================================================
def run_concurrency(cfg: dict, context_tokens: int, max_tokens: int, timeout: int,
                    concurrency: int, saver: ResultsSaver | None,
                    bearer_token: str = None):
    """Run N concurrent requests with the same prompt, measure throughput."""
    prompt = build_haystack_prompt(context_tokens)
    est = estimate_tokens(prompt)

    print(f"Running {concurrency} concurrent requests ({est:,} est. prompt tokens each)...\n")

    results = []
    start_time = time.time()

    def _run(idx: int):
        print(f"  [{idx+1:>2}/{concurrency}] Starting... ", end="", flush=True)
        result = run_single_benchmark(cfg, prompt, max_tokens, timeout, bearer_token)
        if result:
            print(f"TTFT={result['ttft']:.2f}s  GenSpeed={result['gen_speed']:.1f}tok/s  "
                  f"Needle={result['needle_result']}")
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

        print("  Step Summary:")
        print(f"    Wall clock: {wall_clock:.1f}s  |  Throughput: {throughput:.1f} tok/s  |  Needle: {_needle_icon(successful[0]['needle_result'])} {sum(1 for r in successful if r['needle_result']=='exact')}/{len(successful)}")
        print(f"    Avg TTFT: {avg_ttft:.2f}s  |  Avg GenSpeed: {avg_gen:.1f} tok/s")

    if failed:
        print(f"  Failed: {len(failed)}/{len(results)}")

    if saver:
        for idx, result in results:
            if result:
                saver.save_run(f"concurrent-{idx+1}", result, result.get("collected_text", ""))

    print()


def run_concurrency_ramp(cfg: dict, context_tokens: int, max_tokens: int, timeout: int,
                         max_concurrency: int, saver: ResultsSaver | None,
                         bearer_token: str = None):
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
    prompt = build_haystack_prompt(context_tokens)
    est = estimate_tokens(prompt)

    sep = "─" * 60

    print(f"Concurrency Ramp: {' → '.join(str(c) for c in concurrency_levels)}")
    print(f"Context: {context_tokens:,} tokens ({est:,} est. prompt tokens)\n")

    all_results = []

    for workers in concurrency_levels:
        print(sep)
        print(f"  Concurrency: {workers}")
        print(sep)

        step_results = []
        start_time = time.time()

        def _run(idx: int):
            print(f"  [{idx+1:>2}/{workers}] Starting... ", end="", flush=True)
            result = run_single_benchmark(cfg, prompt, max_tokens, timeout, bearer_token)
            if result:
                print(f"TTFT={result['ttft']:.2f}s  GenSpeed={result['gen_speed']:.1f}tok/s  "
                      f"Needle={result['needle_result']}")
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

            needle_pass = sum(1 for r in successful if r['needle_result'] == 'exact')
            needle_sub = sum(1 for r in successful if r['needle_result'] == 'substitution')
            needle_fail = sum(1 for r in successful if r['needle_result'] == 'not_found')

            print("  Step Summary:")
            print(f"    Wall clock: {wall_clock:.1f}s  |  Throughput: {throughput:.1f} tok/s  |  Needle: {_needle_icon(successful[0]['needle_result'])} {needle_pass}/{len(successful)}")
            print(f"    Avg TTFT: {avg_ttft:.2f}s  |  Avg GenSpeed: {avg_gen:.1f} tok/s")

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
            needle_pass = sum(1 for r in successful if r['needle_result'] == 'exact')
            needle_sub = sum(1 for r in successful if r['needle_result'] == 'substitution')
            needle_fail = sum(1 for r in successful if r['needle_result'] == 'not_found')
            rows.append({
                'workers': workers,
                'avg_ttft': avg_ttft,
                'avg_gen': avg_gen,
                'total_gen': total_gen,
                'throughput': throughput,
                'needle_pass': needle_pass,
                'needle_sub': needle_sub,
                'needle_fail': needle_fail,
                'total': len(successful),
            })

        # Degradation vs single-request baseline
        baseline_gen = rows[0]['avg_gen'] if rows else 0

        # Column widths
        w_workers = max(10, max(len(str(r['workers'])) for r in rows))
        w_ttft = max(10, max(len(f"{r['avg_ttft']:.2f}") for r in rows))
        w_gen = max(12, max(len(f"{r['avg_gen']:.1f}") for r in rows))
        w_throughput = max(16, max(len(f"{r['throughput']:.1f}") for r in rows))
        w_degradation = max(14, 14)
        w_needle = max(8, max(len(f"{r['needle_pass']}/{r['total']}") for r in rows))

        # Header
        hdr = (
            f"  {'Concurrency':>{w_workers}s}  "
            f"{'Avg TTFT':>{w_ttft}s}  "
            f"{'Avg GenSpeed':>{w_gen}s}  "
            f"{'Total Throughput':>{w_throughput}s}  "
            f"{'Degradation':>{w_degradation}s}  "
            f"{'Needle':>{w_needle}s}"
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
            f"{'':>{w_needle}s}"
        )

        print(hdr)
        print(sep_row)
        print(units)

        for r in rows:
            degradation = ((baseline_gen - r['avg_gen']) / baseline_gen * 100) if baseline_gen > 0 else 0
            deg_str = f"+{degradation:.1f}%"
            needle_s = f"{r['needle_pass']}/{r['total']}"
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
        total_needle = sum(r['needle_pass'] for r in rows)
        total_reqs = sum(r['total'] for r in rows)
        pass_rate = f"{total_needle}/{total_reqs} ({total_needle/total_reqs*100:.0f}%)" if total_reqs else "N/A"

        best_idx = max(range(len(rows)), key=lambda i: rows[i]['throughput'])
        best_row = rows[best_idx]
        best_tp = f"{best_row['throughput']:.1f} tok/s (at concurrency {best_row['workers']})"

        max_deg = ((baseline_gen - rows[-1]['avg_gen']) / baseline_gen * 100) if baseline_gen > 0 else 0
        max_deg_str = f"{max_deg:.1f}% ({baseline_gen:.1f} → {rows[-1]['avg_gen']:.1f} tok/s)"

        # Recommended concurrency: best throughput-to-degradation tradeoff
        # Pick the point where throughput is still high but degradation isn't too steep
        recommended = rows[-1]['workers']  # default: max
        for r in rows:
            deg = ((baseline_gen - r['avg_gen']) / baseline_gen * 100) if baseline_gen > 0 else 0
            if deg <= 30:
                recommended = r['workers']
            else:
                break

        print(f"  Needle Pass Rate:               {pass_rate}")
        print(f"  Best Total Throughput:          {best_tp}")
        print(f"  Degradation at max:             {max_deg_str}")
        print(f"  Recommended concurrency:        {recommended} (best throughput-to-degradation tradeoff)")
        print(sep)
        print()


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark an LLM endpoint with needle-in-haystack prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  single  — One-shot benchmark (default)\n"
            "  warm    — Cold + warm run comparison (KV cache effect)\n"
            "  ramp    — Growing context benchmark (powers of 2 from 1K to target)\n"
            "  concurrency=N — N concurrent requests at once (default: 2)\n"
            "  concurrency-ramp=N — Scaling concurrency: 1→2→4→...→N (default: 4)\n"
            "\n"
            "Examples:\n"
            "  python3 contextllens.py --model qwen/qwen3.6-27b\n"
            "  python3 contextllens.py --model qwen/qwen3.6-27b-mlx --mode warm\n"
            "  python3 contextllens.py --model qwen/qwen3.6-27b-mlx --mode ramp --context-tokens 32000\n"
            "  python3 contextllens.py --model qwen/qwen3.6-27b --mode concurrency=4\n"
            "  python3 contextllens.py --model qwen/qwen3.6-27b --mode concurrency-ramp=16\n"
            "  python3 contextllens.py --list-models\n"
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

    # Parse mode: concurrency=N or concurrency-ramp=N
    concurrency = None
    concurrency_ramp = None
    mode = args.mode

    if mode.startswith("concurrency-ramp="):
        try:
            concurrency_ramp = int(mode.split("=", 1)[1])
        except ValueError:
            print(f"Error: concurrency-ramp requires a number, got: {mode}")
            sys.exit(1)
        if concurrency_ramp < 1:
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

    try:
        # Header
        print(f"{'='*60}")
        print(f"  Model:              {cfg['label']}")
        print(f"  Endpoint:           {cfg['endpoint']}")
        print(f"  Mode:               {mode}")
        print(f"  Target Context:     {args.context_tokens:,} tokens")
        print(f"  Max Gen Tokens:     {args.max_tokens:,}")
        if saver:
            print(f"  Results Path:       {saver.run_dir}")
        if args.notes:
            print(f"  Notes:              {args.notes}")
        print(f"{'='*60}")

        if mode == "single":
            run_single(cfg, args.context_tokens, args.max_tokens, args.timeout,
                       saver, bearer_token)
        elif mode == "warm":
            run_warm(cfg, args.context_tokens, args.max_tokens, args.timeout,
                     saver, bearer_token)
        elif mode == "ramp":
            run_ramp(cfg, args.context_tokens, args.max_tokens, args.timeout,
                     saver, bearer_token)
        elif concurrency is not None:
            run_concurrency(cfg, args.context_tokens, args.max_tokens, args.timeout,
                            concurrency, saver, bearer_token)
        elif concurrency_ramp is not None:
            run_concurrency_ramp(cfg, args.context_tokens, args.max_tokens, args.timeout,
                                 concurrency_ramp, saver, bearer_token)
    finally:
        sys.stdout = original_stdout

    # Finalize results
    if saver:
        terminal_text = tee.getvalue()
        run_dir = saver.finalize(terminal_text)
        print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
