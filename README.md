# ContextLLens

Unified LLM benchmark script with needle-in-haystack prompts. Compare prefill speed, decode speed, KV cache behavior, and context scaling across any OpenAI-compatible API endpoint.

## Features

- **Needle-in-haystack prompts** — Varied technical content with a hidden fact embedded at ~25% depth, plus a retrieval query. No repetitive garbage.
- **Three benchmark modes:**
  - `single` — One-shot benchmark
  - `warm` — Cold + warm run comparison (measures KV cache effect)
  - `ramp` — Growing context benchmark (powers of 2 from 1K to target)
- **Rich metrics:** TTFT, prefill speed, decode speed, TPOT (ms/token), wall clock, needle retrieval pass/fail, and scaling factor.
- **Multi-model config** — Add endpoints in `config.yaml`, select with `--model`.
- **Automatic results saving** — Each run saves `results.txt`, `results.csv`, `results.json`, and full model outputs to a timestamped folder in `./results/`.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp config.yaml.example config.yaml
# Edit config.yaml with your endpoints

# Run a benchmark
python3 contextllens.py --model my-model --context-tokens 32000 --max-tokens 300

# Compare cold vs warm (KV cache effect)
python3 contextllens.py --model my-model --mode warm --context-tokens 32000

# Ramp test (1K → 2K → 4K → ... → target)
python3 contextllens.py --model my-model --mode ramp --context-tokens 131000 --timeout 3600

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
```

- `endpoint` — Base URL for the OpenAI-compatible API (no trailing slash)
- `model` — Model name sent in the API payload (use LM Studio API Identifier if applicable)
- `label` — Human-readable label shown in reports
- `context` — Max context window in tokens (informational)

## Benchmark Modes

### Single (`--mode single`)
One-shot benchmark with a needle-in-haystack prompt at the specified context size.

### Warm (`--mode warm`)
Runs the same prompt twice and compares cold vs warm metrics. Reveals the KV cache effect — how much faster the second identical request is when the prompt's KV cache is already computed.

### Ramp (`--mode ramp`)
Geometric progression from 1K to target context (powers of 2). Each step is an independent request. The summary table includes a **scaling factor** column:
- `1.0` = linear scaling (ideal)
- `>1.5` = quadratic-ish scaling (degrading)

## Metrics

| Metric | Description |
|---|---|
| **TTFT** | Time to First Token — prefill duration |
| **Prefill Speed** | Prompt tokens processed per second |
| **Decode Duration** | Time from first to last generated token |
| **Gen Speed** | Output tokens per second |
| **TPOT** | Time Per Output Token (ms) — industry standard for decode comparison |
| **Wall Clock** | Total request duration |
| **Needle Retrieved** | ✅/❌ — whether the hidden fact was found in the output |
| **Scaling Factor** | How prefill time scales as context grows (ramp mode only) |

## Results

Results are saved by default to `./results/` in a timestamped subfolder:

```
results/
└── 2026-06-04_17-14_my-model_RAMP_262000/
    ├── results.txt      # Raw terminal output
    ├── results.csv      # Tabular metrics
    ├── results.json     # Structured metrics with run metadata
    ├── notes.txt        # User-supplied notes (if --notes was used)
    └── Output/          # Full model responses per step
        ├── my-model_1_000.txt
        ├── my-model_2_000.txt
        └── ...
```

| Flag | Description |
|---|---|
| `--results-path <dir>` | Override the default `./results/` directory |
| `--no-save` | Disable saving results to disk |
| `--notes "..."` | Add free-form notes (saved to `notes.txt` and `results.json`) |

## Requirements

- Python 3.10+
- `requests` — HTTP client
- `pyyaml` — Config file parsing

```bash
pip install -r requirements.txt
```

## Notes

- **KV Cache** — For fair cold-start comparisons, reload the model between runs or use different prompts.

## License

MIT
