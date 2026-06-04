#!/usr/bin/env python3
"""
contextllens.py — Unified LLM benchmark script.

Modes:
  single  (default) — One-shot benchmark with needle-in-haystack prompt
  warm    — Cold + warm run comparison (KV cache effect)
  ramp    — Growing context benchmark (powers of 2 from 1K to target)

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
    python3 contextllens.py --list-models
"""

import argparse
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
                 context_tokens: int, cfg: dict):
        self.results_path = results_path
        self.model_key = model_key
        self.mode = mode
        self.context_tokens = context_tokens
        self.cfg = cfg
        self.timestamp = datetime.now()
        self.model_label = cfg.get("label", model_key)
        self.runs = []  # list of (step_label, metrics_dict, collected_text)

        # Sanitize model key for filenames
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', model_key)
        safe_key = safe_key.strip('_')

        # Create run folder: YYYY-MM-DD_HH:MM_<Model-Name>_<Mode>_<Context-Tokens>
        time_str = self.timestamp.strftime("%Y-%m-%d_%H:%M")
        mode_label = mode.upper()
        folder_name = f"{time_str}_{safe_key}_{mode_label}_{context_tokens}"
        self.run_dir = os.path.join(results_path, folder_name)
        self.output_dir = os.path.join(self.run_dir, "Output")

        os.makedirs(self.output_dir, exist_ok=True)

    def save_run(self, step_label: str, metrics: dict, collected_text: str):
        """Record a single benchmark run."""
        self.runs.append((step_label, metrics, collected_text))

        # Save individual output file
        if collected_text:
            safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', step_label)
            safe_label = safe_label.strip('_')
            time_str = datetime.now().strftime("%Y-%m-%d_%H:%M")
            safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', self.model_key)
            safe_key = safe_key.strip('_')
            filename = f"{time_str}_{safe_key}_{metrics.get('prompt_tokens', 'unknown')}.txt"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(collected_text)

    def finalize(self, terminal_output: str):
        """Write results.txt, results.csv, and results.json."""
        # results.txt — raw terminal output
        txt_path = os.path.join(self.run_dir, "results.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(terminal_output)

        # results.json — structured metrics
        json_data = {
            "run_info": {
                "timestamp": self.timestamp.isoformat(),
                "model_key": self.model_key,
                "model_label": self.model_label,
                "endpoint": self.cfg.get("endpoint", ""),
                "mode": self.mode,
                "target_context_tokens": self.context_tokens,
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
]

# The hidden fact embedded in the haystack
NEEDLE = (
    "CRITICAL REFERENCE: The deployment authorization code for the production "
    "environment is VERTEX-9284-KILO. This code must be included in all "
    "production deployment requests and verified by the on-call engineer."
)

# The query asking the model to find the needle and generate a long response
QUERY = (
    "What is the deployment authorization code mentioned in the document above? "
    "After providing the code, compile a formal infrastructure status report "
    "based on the provided meeting notes, incident reports, and performance analyses. "
    "The report should be detailed and approximately 1000 words long."
)


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~1.3 chars/token for English text."""
    return max(1, int(len(text) / 1.3))


def build_haystack_prompt(target_tokens: int) -> str:
    """
    Build a varied haystack with the needle embedded at ~25% position.
    Cycles through HAYSTACK_PARAGRAPHS to fill the target token count.
    """
    needle_position = max(200, int(target_tokens * 0.25))

    paragraphs = []
    current_tokens = 0
    needle_inserted = False
    idx = 0

    while current_tokens < target_tokens:
        para = HAYSTACK_PARAGRAPHS[idx % len(HAYSTACK_PARAGRAPHS)]
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
def run_single_benchmark(cfg: dict, prompt: str, max_tokens: int, timeout: int) -> dict | None:
    """
    Run a single benchmark request. Returns a metrics dict, or None on failure.
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

    start_time = time.time()

    try:
        response = requests.post(
            f"{endpoint}/chat/completions",
            json=payload,
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
    needle_found  = "VERTEX-9284-KILO" in collected_text

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
    }


def print_results(results: dict, label: str = "", show_preview: bool = True):
    """Print formatted benchmark results."""
    if label:
        print(f"\n=== {label} ===")

    print(f"  Prompt Tokens Processed:  {results['prompt_tokens']:>10,}")
    print(f"  Tokens Generated:         {results['completion_tokens']:>10,}")
    print(f"  Total Tokens (round-trip):{results['total_tokens']:>10,}")
    if results["ttft"]:
        print(f"  Time to First Token:      {results['ttft']:>10.4f}s")
        print(f"  Prefill Speed:            {results['prefill_speed']:>10.1f} tok/s")
    print(f"  Decode Duration:          {results['decode_duration']:>10.2f}s")
    print(f"  Generation Speed:         {results['gen_speed']:>10.2f} tok/s")
    print(f"  TPOT (ms/token):          {results['tpot']:>10.2f}")
    print(f"  Total Wall Clock:         {results['wall_clock']:>10.2f}s")
    print(f"  Needle Retrieved:         {'✅ PASS' if results['needle_found'] else '❌ FAIL':>10s}")

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
               saver: ResultsSaver | None):
    """One-shot benchmark with needle-in-haystack prompt."""
    prompt = build_haystack_prompt(context_tokens)
    print(f"Sending request ({estimate_tokens(prompt):,} est. prompt tokens)...")
    results = run_single_benchmark(cfg, prompt, max_tokens, timeout)
    if results:
        print_results(results)
        if saver:
            saver.save_run("single", results, results.get("collected_text", ""))


def run_warm(cfg: dict, context_tokens: int, max_tokens: int, timeout: int,
             saver: ResultsSaver | None):
    """Cold + warm run comparison to measure KV cache effect."""
    prompt = build_haystack_prompt(context_tokens)
    est = estimate_tokens(prompt)

    print(f"Running COLD benchmark ({est:,} est. prompt tokens)...")
    cold = run_single_benchmark(cfg, prompt, max_tokens, timeout)
    if not cold:
        return

    print(f"\nRunning WARM benchmark (same prompt, KV cache should be warm)...")
    warm = run_single_benchmark(cfg, prompt, max_tokens, timeout)
    if not warm:
        return

    print_results(cold, label="Cold Run", show_preview=False)
    print_results(warm, label="Warm Run", show_preview=False)

    # Comparison
    print("=== Cold vs Warm Comparison ===")
    if cold["ttft"] and warm["ttft"]:
        print(f"  TTFT Speedup:           {cold['ttft'] / warm['ttft']:>10.1f}x")
    if cold["gen_speed"] and warm["gen_speed"]:
        print(f"  Decode Speed Ratio:     {cold['gen_speed'] / warm['gen_speed']:>10.2f}x")
    if cold["prefill_speed"] and warm["prefill_speed"]:
        print(f"  Prefill Speedup:        {warm['prefill_speed'] / cold['prefill_speed']:>10.1f}x")
    print(f"  Wall Clock Speedup:       {cold['wall_clock'] / warm['wall_clock']:>10.1f}x")
    print(f"  Cold Needle:              {'✅ PASS' if cold['needle_found'] else '❌ FAIL':>10s}")
    print(f"  Warm Needle:              {'✅ PASS' if warm['needle_found'] else '❌ FAIL':>10s}")

    if warm["collected_text"]:
        snippet = warm["collected_text"][:300]
        print(f"\n  Output preview: {snippet!r}")
        if len(warm["collected_text"]) > 300:
            print(f"  ... ({len(warm['collected_text'])} chars total)")
    print()

    if saver:
        saver.save_run("cold", cold, cold.get("collected_text", ""))
        saver.save_run("warm", warm, warm.get("collected_text", ""))


def run_ramp(cfg: dict, max_context_tokens: int, max_tokens: int, timeout: int,
             saver: ResultsSaver | None):
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
        result = run_single_benchmark(cfg, prompt, max_tokens, timeout)
        if result:
            results.append((step, result))
            print(f"TTFT={result['ttft']:.2f}s  Prefill={result['prefill_speed']:.0f}tok/s  "
                  f"GenSpeed={result['gen_speed']:.1f}tok/s  Wall={result['wall_clock']:.2f}s")
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
    hdr = f"  {'Context':>10s}  {'TTFT':>8s}  {'Prefill':>10s}  {'Decode':>8s}  {'TPOT':>8s}  {'Gen Speed':>10s}  {'Wall':>8s}  {'Needle':>8s}  {'Scale':>8s}"
    sep = f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}"
    fmt = f"  {{:>10,}}  {{:>8s}}  {{:>10s}}  {{:>8.2f}}  {{:>8.2f}}  {{:>10.2f}}  {{:>8.2f}}  {{:>8s}}  {{:>8s}}"

    print(hdr)
    print(f"  {'(tokens)':>10s}  {'(s)':>8s}  {'(tok/s)':>10s}  {'(s)':>8s}  {'(ms)':>8s}  {'(tok/s)':>10s}  {'(s)':>8s}  {'':>8s}  {'factor':>8s}")
    print(sep)

    for i, (step, r) in enumerate(results):
        ttft_s     = f"{r['ttft']:.2f}" if r["ttft"] else "N/A"
        prefill_s  = f"{r['prefill_speed']:.0f}" if r["prefill_speed"] else "N/A"
        needle_s   = "✅" if r["needle_found"] else "❌"

        if i > 0 and r["ttft"] and results[i-1][1]["ttft"]:
            ctx_ratio = step / results[i-1][0]
            ttft_ratio = r["ttft"] / results[i-1][1]["ttft"]
            scale = (ttft_ratio / ctx_ratio) if ctx_ratio > 0 else 0.0
            scale_s = f"{scale:.2f}"
        else:
            scale_s = "—"

        print(fmt.format(step, ttft_s, prefill_s, r["decode_duration"], r["tpot"],
                         r["gen_speed"], r["wall_clock"], needle_s, scale_s))

    print()
    print("  Scaling factor: 1.0 = linear (ideal), >1.5 = quadratic-ish (degrading)")

    if results[-1][1]["collected_text"]:
        snippet = results[-1][1]["collected_text"][:300]
        print(f"\n  Output preview (from {results[-1][0]:,} token run): {snippet!r}")
        if len(results[-1][1]["collected_text"]) > 300:
            print(f"  ... ({len(results[-1][1]['collected_text'])} chars total)")
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
            "\n"
            "Examples:\n"
            "  python3 contextllens.py --model qwen/qwen3.6-27b\n"
            "  python3 contextllens.py --model qwen/qwen3.6-27b-mlx --mode warm\n"
            "  python3 contextllens.py --model qwen/qwen3.6-27b-mlx --mode ramp --context-tokens 32000\n"
            "  python3 contextllens.py --list-models\n"
        ),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model identifier (see --list-models).",
    )
    parser.add_argument(
        "--mode", choices=["single", "warm", "ramp"], default="single",
        help="Benchmark mode (default: single).",
    )
    parser.add_argument(
        "--context-tokens", type=int, default=3300,
        help="Target prompt tokens (default: 3300). In ramp mode, this is the max.",
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

    # Setup results saver
    saver = None
    if not args.no_save:
        saver = ResultsSaver(
            results_path=args.results_path,
            model_key=args.model,
            mode=args.mode,
            context_tokens=args.context_tokens,
            cfg=cfg,
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
        print(f"  Mode:               {args.mode}")
        print(f"  Target Context:     {args.context_tokens:,} tokens")
        print(f"  Max Gen Tokens:     {args.max_tokens:,}")
        if saver:
            print(f"  Results Path:       {saver.run_dir}")
        print(f"{'='*60}")

        if args.mode == "single":
            run_single(cfg, args.context_tokens, args.max_tokens, args.timeout, saver)
        elif args.mode == "warm":
            run_warm(cfg, args.context_tokens, args.max_tokens, args.timeout, saver)
        elif args.mode == "ramp":
            run_ramp(cfg, args.context_tokens, args.max_tokens, args.timeout, saver)
    finally:
        sys.stdout = original_stdout

    # Finalize results
    if saver:
        terminal_text = tee.getvalue()
        run_dir = saver.finalize(terminal_text)
        print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
