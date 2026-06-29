# ContextLLens

Unified LLM benchmark script with needle-in-haystack prompts. Compare prefill speed, decode speed, KV cache behavior, and context scaling across any OpenAI-compatible API endpoint.

**Version:** 0.4.0

## Features

- **Multi-Needle Retrieval** — 3 needles embedded at 10%, 50%, and 90% context depth. Tests whether models can retrieve multiple facts scattered throughout long context, not just a single needle.
- **Coherency Suite** — Automated checks for looping (4-gram repetition collapse), length violations (target ~1000 words, window 400-1500), and reasoning-loop detection for thinking models.
- **Reasoning Diagnostics** — For models with chain-of-thought/reasoning tokens, the benchmark separately tracks reasoning loop detection and token counts.
- **Five benchmark modes:**
  - `single` — One-shot benchmark
  - `warm` — Cold + warm run comparison (measures KV cache effect)
  - `ramp` — Growing context benchmark (powers of 2 from 1K to target)
  - `concurrency=N` — N concurrent requests at once (default: `concurrency=2`)
  - `concurrency-ramp=N` — Scaling concurrency: 1→2→4→...→N (default: `concurrency-ramp=4`)
- **Rich metrics:** TTFT, prefill speed, decode speed, TPOT (ms/token), wall clock, needle retrieval pass/fail, scaling factor, and aggregate throughput for concurrency modes.
- **Multi-model config** — Add endpoints in `config.yaml`, select with `--model`. Optional `bearer_token` for cloud APIs (OpenAI, Anthropic, etc.).
- **Automatic results saving** — Each run saves `results.txt`, `results.csv`, `results.json`, and full model outputs to a timestamped folder in `./results/`. Concurrency runs save individual output files per request.
- **Unicode normalization** — Needle matching normalizes common Unicode lookalikes (en-dashes, em-dashes, soft hyphens) so codes with subtle character substitutions still match.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp config.yaml.example config.yaml
# Edit config.yaml with your endpoints

# Run a benchmark
python3 contextllens.py --model my-model --context-tokens 32000 --max-tokens 1500

# Compare cold vs warm (KV cache effect)
python3 contextllens.py --model my-model --mode warm --context-tokens 32000

# Ramp test (1K → 2K → 4K → ... → target)
python3 contextllens.py --model my-model --mode ramp --context-tokens 131000 --timeout 3600

# Concurrent requests (default: 2)
python3 contextllens.py --model my-model --mode concurrency=4

# Scaling concurrency (default: 1→2→4)
python3 contextllens.py --model my-model --mode concurrency-ramp=16

# Add notes to a run (saved to notes.txt)
python3 contextllens.py --model my-model --notes "Warm cache from earlier run"

# List configured models
python3 contextllens.py --list-models
```

## Configuration

Edit `config.yaml` to add your model endpoints:

```yaml
models:
  my-gpu-model:
    endpoint: "http://localhost:8000/v1"
    model:    "my-model-name"
    label:    "My GPU Model"
    context:  131072

  # Cloud model requiring API key
  my-cloud-model:
    endpoint: "https://api.openai.com/v1"
    model:    "gpt-4o"
    label:    "GPT-4o (OpenAI)"
    context:  128000
    bearer_token: "sk-..."
```

- `endpoint` — Base URL for the OpenAI-compatible API (no trailing slash)
- `model` — Model name sent in the API payload (use LM Studio API Identifier if applicable)
- `label` — Human-readable label shown in reports
- `context` — Max context window in tokens (informational)
- `bearer_token` — Optional API key for cloud endpoints (OpenAI, Anthropic, etc.). Omit for local endpoints that don't require authentication.

## Benchmark Modes

### Single (`--mode single`)
One-shot benchmark with a multi-needle prompt at the specified context size.

### Warm (`--mode warm`)
Runs the same prompt twice and compares cold vs warm metrics. Reveals the KV cache effect — how much faster the second identical request is when the prompt's KV cache is already computed.

### Ramp (`--mode ramp`)
Geometric progression from 1K to target context (powers of 2). Each step is an independent request. The summary table includes a **scaling factor** column:
- `1.0` = linear scaling (ideal)
- `>1.5` = quadratic-ish scaling (degrading)

### Concurrency (`--mode concurrency=N`)
Launches N concurrent requests at once and measures per-request and aggregate metrics. Each request gets a unique prompt (shuffled with a different seed) so KV cache doesn't bleed between requests.

Reports per-request metrics, aggregate throughput, needle pass rate, and speed degradation compared to single-request baseline.

### Concurrency Ramp (`--mode concurrency-ramp=N`)
Sequential scaling: 1→2→4→...→N concurrent requests. Each concurrency level completes fully before moving to the next. Reveals the sweet spot where adding more concurrency stops being worth it.

Reports per-level throughput, degradation curve, and best-concurrency-level.

## Needle-in-Haystack Prompt Design

The prompt uses **multiple needle sets** drawn from a pool of 5 sets, each containing 3 needles (start, mid, end) with unique authorization codes. One set is randomly selected per run, and decoy codes from non-selected sets are injected as distractors. This prevents models from learning static code patterns.

Needles are embedded at **10%, 50%, and 90%** of the total context depth.

Paragraphs are **shuffled with a unique seed per request** (seed = `target_tokens + request_index`), so concurrent runs get different orderings while repeated runs with the same context size get the same ordering.

## Metrics

| Metric | Description |
|---|---|
| **TTFT** | Time to First Token — prefill duration |
| **Prefill Speed** | Prompt tokens processed per second |
| **Decode Duration** | Time from first to last generated token |
| **Gen Speed** | Output tokens per second |
| **TPOT** | Time Per Output Token (ms) — industry standard for decode comparison |
| **Wall Clock** | Total request duration |
| **Needle Retrieved** | ✅/❌ per position (start, mid, end) — whether each hidden code was found |
| **Scaling Factor** | How prefill time scales as context grows (ramp mode only) |
| **Total Throughput** | Sum of all gen speeds across concurrent requests |
| **Degradation** | Per-request speed drop vs single-request baseline (concurrency modes) |
| **Looping** | YES/NO — 4-gram repetition ratio below 0.5 threshold |
| **Length** | Word count with (OK)/(OUT OF RANGE) — target ~1000 words, window 400-1500 |
| **Reasoning Tokens** | Count of reasoning/thinking tokens (for models that produce them) |
| **Reasoning Loop** | YES/NO — loop detection applied to reasoning text separately |

## Results

Results are saved by default to `./results/` in a timestamped subfolder:

```
results/
└── 2026-06-28_23-36_qwen_qwen3_6-35b-a3b-mlx_SINGLE_32000/
    ├── results.txt      # Raw terminal output
    ├── results.csv      # Tabular metrics
    ├── results.json     # Structured metrics with run metadata
    ├── notes.txt        # User-supplied notes (if --notes was used)
    └── Output/          # Full model responses per step/request
        └── qwen_qwen3_6-35b-a3b-mlx_single.txt
```

| Flag | Description |
|---|---|
| `--results-path <dir>` | Override the default `./results/` directory |
| `--no-save` | Disable saving results to disk |
| `--notes "..."` | Add free-form notes (saved to `notes.txt` and `results.json`) |
| `--seed <N>` | Use a fixed random seed for reproducible needle/paragraph selection |
| `--no-system-prompt` | Skip the system prompt role (useful for models that don't support it) |

## Requirements

- Python 3.10+
- `requests` — HTTP client
- `pyyaml` — Config file parsing

```bash
pip install -r requirements.txt
```

## Notes

- **KV Cache** — For fair cold-start comparisons, reload the model between runs or use different prompts.
- **Concurrency** — The number of concurrent requests is limited by your server's resources. Local MLX models on Mac Studio hit a hard limit around concurrency=2 at 16K context due to unified memory constraints. Enterprise GPU clusters handle higher concurrency but may show MoE expert contention effects.
- **Bearer tokens** — Omit `bearer_token` for local endpoints that don't require authentication. For cloud APIs (OpenAI, Anthropic, etc.), include your API key as the `bearer_token` value.
- **Reasoning models** — Models that produce chain-of-thought/reasoning tokens have those tracked separately from output tokens. The coherency checks apply to the combined reasoning + output text.

## License

MIT
