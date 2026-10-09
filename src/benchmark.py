"""Benchmark matrix: Jev vs JPT vs Decision 2.0 and their fine-tunes on the JevBench test sets.

    python -m src.benchmark                                  # run every (model, dataset) pair not in results/raw yet
    python -m src.benchmark --models jpt decision2 --datasets boolq --rerun
    python -m src.benchmark --n-samples 5 --models decision2_boolq_lora   # quick check

Models: jev, jpt, decision2, plus every checkpoints/<jpt|decision2>-<dataset or modality>[-expert]-<lora|merged>
folder (checkpoints/archive is ignored). A fine-tune runs only on datasets of its own question type.
Outputs results/raw/<model>_<dataset>.jsonl (one record per example) and results/master_benchmark_matrix.csv,
which is rebuilt from the raw files of all current models after every run, so nothing finished is ever dropped.
"""
import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from llm2jev.prompt import options_of

from src.data.jevbench import BENCHMARK_DATASETS, DATASETS_BY_MODALITY, MODALITY_OF, gold_key, load_examples
from src.metrics import brier, expected_calibration_error, predicted_index
from src.models import FAMILIES, load_model

# ============================================================
# SETTINGS
# ============================================================
N_SAMPLES = 500                  # first N rows of each test split
DATASETS = BENCHMARK_DATASETS    # name -> "ID" | "OOD"
CHECKPOINTS = Path("checkpoints")
RAW_DIR = Path("results/raw")
MATRIX_PATH = Path("results/master_benchmark_matrix.csv")
REMOTE_FAMILIES = {"jev"}        # API models: no warmup, no CUDA sync around the timing

MATRIX_COLUMNS = ["model", "checkpoint", "dataset", "modality", "dataset_type", "accuracy", "mean_brier", "ece",
                  "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "mean_input_tokens", "total_cost_usd",
                  "n_samples"]


def discover_models():
    """model name -> {"family", "checkpoint" (folder or None), "modality" (None = every dataset)}."""
    models = {family: {"family": family, "checkpoint": None, "modality": None} for family in FAMILIES}
    for path in sorted(p for p in CHECKPOINTS.iterdir() if p.is_dir()):
        family, _, rest = path.name.partition("-")
        target = rest.rpartition("-")[0].removesuffix("-expert")  # jpt-banking77-merged -> banking77
        modality = target if target in DATASETS_BY_MODALITY else MODALITY_OF.get(target)
        if family in FAMILIES and path.name.endswith(("-lora", "-merged")) and modality:
            models[path.name.replace("-", "_")] = {"family": family, "checkpoint": path, "modality": modality}
    return models


def raw_path(model_name, dataset):
    return RAW_DIR / f"{model_name}_{dataset}.jsonl"


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(example, prediction, latency_ms):
    """One raw record: the model's probabilities in the question's option order, its answer and Brier score."""
    keys, _ = options_of(example["question"])
    by_key = dict(zip(prediction["keys"], prediction["probabilities"]))
    probabilities = torch.tensor([float(by_key.get(key, 0.0)) for key in keys])
    probabilities = probabilities / probabilities.sum()
    gold = keys.index(gold_key(example))
    predicted = int(predicted_index(probabilities, example["primitive"]))
    return {"example_id": example["id"], "prediction": keys[predicted], "pred_idx": predicted,
            "confidence": probabilities[predicted].item(), "probabilities": probabilities.tolist(),
            "correct": int(predicted == gold), "gold_idx": gold,
            "brier": brier(probabilities[None], torch.tensor([gold]))[0].item(), "latency_ms": latency_ms,
            "input_tokens": prediction["input_tokens"], "output_tokens": 0, "cost": prediction["cost"]}


def run(model, examples, remote):
    """Predict every example, timing each call (CUDA-synchronized for local models)."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() and not remote else (lambda: None)
    if not remote:
        model.predict(examples[-1])  # warmup: kernels, caches
    records = []
    for example in examples:
        sync()
        start = time.perf_counter()
        prediction = model.predict(example)
        sync()
        latency_ms = prediction.get("request_ms", (time.perf_counter() - start) * 1000)
        records.append(score(example, prediction, latency_ms))
    return records


def matrix_row(model_name, spec, dataset, records):
    latencies = [r["latency_ms"] for r in records]
    return {"model": model_name, "checkpoint": spec["checkpoint"].name if spec["checkpoint"] else "",
            "dataset": dataset, "modality": MODALITY_OF[dataset], "dataset_type": DATASETS[dataset],
            "accuracy": statistics.fmean(r["correct"] for r in records),
            "mean_brier": statistics.fmean(r["brier"] for r in records),
            "ece": expected_calibration_error([r["confidence"] for r in records], [r["correct"] for r in records]),
            "mean_latency_ms": statistics.fmean(latencies), "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "mean_input_tokens": statistics.fmean(r["input_tokens"] for r in records),
            "total_cost_usd": sum(r["cost"] for r in records), "n_samples": len(records)}


def rebuild_matrix(models):
    """The matrix from every raw file of the current models; printed sorted by question type and dataset."""
    rows = [matrix_row(name, spec, dataset, read_records(raw_path(name, dataset)))
            for name, spec in models.items() for dataset in DATASETS if raw_path(name, dataset).is_file()]
    with MATRIX_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    width = max([len(r["model"]) for r in rows] + [5])
    print(f"\n{'dataset':<24} {'type':<4} {'model':<{width}} {'acc':>6} {'brier':>6} {'ece':>6} {'p50 ms':>8} {'n':>4}")
    for r in sorted(rows, key=lambda r: (r["modality"], r["dataset_type"], r["dataset"], -r["accuracy"])):
        print(f"{r['dataset']:<24} {r['dataset_type']:<4} {r['model']:<{width}} {r['accuracy']:>6.3f} "
              f"{r['mean_brier']:>6.3f} {r['ece']:>6.3f} {r['median_latency_ms']:>8.1f} {r['n_samples']:>4}")
    print(f"\nMatrix ({len(rows)} rows): {MATRIX_PATH}")


def main():
    models = discover_models()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=models, default=list(models))
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--rerun", action="store_true", help="Run again even when a raw file is complete")
    args = p.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Pairs to run: a fine-tune only on its own question type, and only when its raw file is missing, short,
    # or holds other examples than today's test split (the dataset on the Hub can change).
    examples = {dataset: load_examples(dataset, "test", args.n_samples) for dataset in args.datasets}
    pending = {}
    for name in args.models:
        for dataset in args.datasets:
            path = raw_path(name, dataset)
            wanted_ids = [example["id"] for example in examples[dataset]]
            done = path.is_file() and [r["example_id"] for r in read_records(path)][:len(wanted_ids)] == wanted_ids
            if models[name]["modality"] in (None, MODALITY_OF[dataset]) and (args.rerun or not done):
                pending.setdefault(name, []).append(dataset)

    for name, datasets in pending.items():
        spec = models[name]
        print(f"\n=== {name} on {datasets} ===", flush=True)
        model = load_model(spec["family"], spec["checkpoint"])
        for dataset in datasets:
            records = run(model, examples[dataset], remote=spec["family"] in REMOTE_FAMILIES)
            with raw_path(name, dataset).open("w", encoding="utf-8") as f:
                f.writelines(json.dumps(record) + "\n" for record in records)
            row = matrix_row(name, spec, dataset, records)
            print(f"  {dataset}: acc={row['accuracy']:.3f} brier={row['mean_brier']:.3f} ece={row['ece']:.3f} "
                  f"p50={row['median_latency_ms']:.1f}ms", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rebuild_matrix(models)


if __name__ == "__main__":
    main()
