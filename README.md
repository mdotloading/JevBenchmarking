# Jev Benchmarking

This repository benchmarks decision-oriented models across the three output primitives used by Jev: categorical choice, binary decision (`noul`), and ordinal score. The goal is to compare task accuracy, response latency, token usage, and estimated cost under one shared evaluation harness while keeping the same structured state and question format for every model.

The benchmark runs each model on the first 200 examples from the test split of each selected dataset configuration. Per-example predictions are written as JSON Lines files, and aggregate metrics are written to `results/summary.csv`.

## What is being evaluated?

| Primitive | Meaning | Dataset configuration | Benchmark task |
|---|---|---|---|
| `choice` | Select exactly one label from a supplied set | [`banking77`](https://huggingface.co/datasets/Praveenrajus/jev-bench/viewer/banking77) | Route a banking support request to the correct intent |
| `noul` | Produce a binary yes/no decision | [`boolq`](https://huggingface.co/datasets/Praveenrajus/jev-bench/viewer/boolq) | Answer a boolean reading-comprehension question |
| `score` | Produce an integer rating from an ordered scale | [`helpsteer2_helpfulness`](https://huggingface.co/datasets/Praveenrajus/jev-bench/viewer/helpsteer2_helpfulness) | Rate the helpfulness of an assistant response |

All three configurations are part of [`Praveenrajus/jev-bench`](https://huggingface.co/datasets/Praveenrajus/jev-bench) on Hugging Face. The benchmark uses each configuration's `test` split. Rows contain a JSON-encoded `state`, a JSON-encoded `question`, the primitive, source metadata, and a reference label. [`src/data/parser.py`](src/data/parser.py) converts these fields into the common Python representation consumed by every model adapter.

## Models

| Benchmark name | Model or service | Execution mode | Adapter |
|---|---|---|---|
| `jev` | `jev-latest` | Hosted API call to TypeSafe (`https://api.typesafe.ai/v1/systemone`) | [`src/models/jev.py`](src/models/jev.py) |
| `laya` | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | Local inference on CUDA when available, otherwise CPU | [`src/models/laya.py`](src/models/laya.py) |
| `jpt` | SystemOne-compatible local service | HTTP call to a locally hosted server at `127.0.0.1:8080` | [`src/models/jpt.py`](src/models/jpt.py) |
| `qwen` | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Local Transformers inference with automatic device placement | [`src/models/qwen_local.py`](src/models/qwen_local.py) |

Jev therefore measures a remote, production API path, while Laya and Qwen measure in-process local inference. JPT uses HTTP but targets a service hosted on the same machine. These deployment differences are important when interpreting latency.

[`src/models/decision_2.0.py`](src/models/decision_2.0.py) is an exploratory script for `vllm-sr/Decision-2.0-Kai-0.6B`; it is not included in the model map executed by the main benchmark.

## Evaluation behavior

Each model receives the same parsed example and returns a normalized record containing its prediction, label, correctness, input/output token counts, cost, source, primitive, and example ID.

- `choice` predictions are compared directly with the reference label.
- `noul` outputs from Jev-compatible models are thresholded at `0.5`; Qwen is instructed to return `0` or `1` directly.
- `score` outputs from Jev-compatible models are rounded to an integer; Qwen is instructed to return an integer directly.
- Accuracy is exact-match accuracy for all three primitives.
- The configured sample size is 200 examples per model-primitive pair, or the full test split when it contains fewer than 200 examples.
- Evaluation is sequential and deterministic decoding is used for Qwen (`do_sample=False`).

## Latency methodology

Latency is measured per example with Python's high-resolution monotonic timer, `time.perf_counter()`. Timing begins immediately before the model adapter is called and ends immediately after it returns.

This means:

- dataset download, row parsing, model loading, and one-time initialization are excluded;
- Jev latency includes request serialization, network transfer, remote processing, and response parsing;
- JPT latency includes the complete HTTP round trip to the local service;
- Laya and Qwen latency covers local prediction/generation;
- CUDA is synchronized immediately before and after Laya and Qwen inference so asynchronous GPU execution is included in the measurement;
- Laya and Qwen receive one untimed warm-up prediction per primitive before measurement begins;
- there is no warm-up request for the API-based Jev and JPT adapters, and no batching, concurrency, or retry logic.

The summary reports arithmetic mean, median, and p95 latency in milliseconds. The p95 value is calculated with `statistics.quantiles(latencies, n=100)[94]`. Because hosted and local models include different infrastructure boundaries, latency values represent end-to-end behavior of each configured deployment rather than isolated model compute time.

## Metrics and output files

Raw predictions are saved to:

```text
results/raw/<model>_<primitive>.jsonl
```

The aggregate file is [`results/summary.csv`](results/summary.csv). It contains one row per model-primitive pair with:

- sample count, correct count, and accuracy;
- mean, median, and p95 latency;
- total input and output tokens;
- total cost and average cost per request.

Jev cost is currently estimated from input tokens at `$0.042` per million input tokens. Local model adapters report zero monetary API cost; that does not include hardware, electricity, or hosting costs.

## Results

The latest aggregate results from [`results/summary.csv`](results/summary.csv) are shown below. Accuracy is exact match, and all latency values are milliseconds. Bold values mark the highest accuracy and lowest median latency observed for each primitive.

| Model | Primitive | Samples | Accuracy | Correct | Mean latency (ms) | Median latency (ms) | p95 latency (ms) | Input tokens | Output tokens | Total cost | Avg. cost/request |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev | `choice` | 200 | **82.0%** | 164 | 470.22 | 461.53 | 563.11 | 337,810 | 165,817 | $0.014188 | $0.000071 |
| Jev | `noul` | 200 | **92.5%** | 185 | 469.56 | 461.84 | 542.80 | 88,804 | 4,200 | $0.003730 | $0.000019 |
| Jev | `score` | 200 | **34.0%** | 68 | 460.32 | 452.57 | 525.44 | 181,219 | 3,600 | $0.007611 | $0.000038 |
| Laya | `choice` | 200 | 37.0% | 74 | 50.18 | **48.20** | 69.51 | 66,419 | 0 | $0.000000 | $0.000000 |
| Laya | `noul` | 200 | 83.5% | 167 | 47.65 | **48.03** | 61.95 | 38,339 | 0 | $0.000000 | $0.000000 |
| Laya | `score` | 200 | 22.5% | 45 | 51.36 | **48.96** | 72.58 | 92,559 | 0 | $0.000000 | $0.000000 |
| JPT | `choice` | 200 | 66.0% | 132 | 205.18 | 189.48 | 267.23 | 191,803 | 200 | $0.000000 | $0.000000 |
| JPT | `noul` | 200 | 78.0% | 156 | 116.00 | 110.21 | 159.95 | 48,393 | 200 | $0.000000 | $0.000000 |
| JPT | `score` | 200 | 29.0% | 58 | 175.18 | 159.11 | 239.19 | 141,469 | 200 | $0.000000 | $0.000000 |
| Qwen | `choice` | 200 | 36.0% | 72 | 490.17 | 475.72 | 743.24 | 92,203 | 1,400 | $0.000000 | $0.000000 |
| Qwen | `noul` | 200 | 44.5% | 89 | 293.30 | 279.35 | 391.28 | 46,026 | 800 | $0.000000 | $0.000000 |
| Qwen | `score` | 200 | 32.5% | 65 | 321.98 | 306.93 | 419.32 | 134,503 | 800 | $0.000000 | $0.000000 |

The generated CSV column order is:

```csv
n_samples,accuracy,correct,mean_latency_ms,median_latency_ms,total_input_tokens,total_output_tokens,total_cost,average_cost,p95_latency_ms,model,primitive
```

## Setup

Python dependencies are pinned in [`requirements.txt`](requirements.txt). A CUDA-capable PyTorch installation is recommended for the locally hosted models but is not required by Laya's device selection logic.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the repository root with the credentials needed by the enabled adapters:

```dotenv
HF_TOKEN=your_hugging_face_token
TYPESAFE_API_KEY=your_typesafe_api_key
```

- `HF_TOKEN` is used to load the JevBench dataset.
- `TYPESAFE_API_KEY` authenticates requests to the hosted Jev API.
- The JPT-compatible local service must be running at `http://127.0.0.1:8080/v1/systemone` before the benchmark starts.
- Model weights for Laya and Qwen are downloaded on first use and cached locally.

The repository also contains an OpenRouter smoke test that expects `OPENROUTER_API_KEY`, but OpenRouter is not used by the main benchmark.

## Running the benchmark

Run from the repository root so imports and relative result paths resolve correctly:

```powershell
python -m src.models.benchmark
```

The benchmark iterates over all four configured models and all three primitives, prints a summary for each combination, writes per-example JSONL files under `results/raw/`, and finally writes `results/summary.csv`.

## Repository layout

```text
src/
|-- data/
|   |-- parser.py          # Converts JevBench rows to shared Python objects
|   `-- check_data.py      # Inspects and validates dataset examples
|-- models/
|   |-- benchmark.py       # Main benchmark runner and aggregation
|   |-- jev.py             # Hosted Jev API adapter
|   |-- jpt.py             # Local SystemOne HTTP adapter
|   |-- laya.py            # Local Laya adapter
|   |-- qwen_local.py      # Local Qwen adapter and prompt formatting
|   `-- decision_2.0.py    # Exploratory Decision 2.0 script
`-- smoke-tests/           # Environment and individual-adapter checks
results/
`-- summary.csv            # Aggregate benchmark results
```
