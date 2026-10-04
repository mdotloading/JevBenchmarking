import os
import time
import statistics
import torch
from pathlib import Path
import csv
import json
from dotenv import load_dotenv
from datasets import load_dataset
from src.data.parser import parse_row
from src.models.jev import predict_jev
from src.models.laya import predict_laya
from src.models.jpt import predict_jpt
from src.models.qwen_local import predict_qwen

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

N_SAMPLES = 200

RESULTS_DIR = Path("results")
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
RAW_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)




def save_results(
        results,
        model_name,
        primitive,
):
    """
    Saves every single predictions json output in a jsonl file
    
    """
    path = (
        RAW_RESULTS_DIR / f"{model_name}_{primitive}.jsonl"
    )

    with path.open("w", encoding = "utf-8") as f:
        for result in results: 
            json.dump(result, f)
            f.write("\n")



def save_summaries(summaries):
    path = RESULTS_DIR / "summary.csv"

    if not summaries:
        return
    
    with path.open("w", newline= "",encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summaries[0].keys()
        )
        writer.writeheader()
        writer.writerows(summaries)




def warmup(dataset, predict_fn, sync_cuda=False):
    example = parse_row(dataset[-1])

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    predict_fn(example)

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()



def run_benchmark(dataset, predict_fn, n=N_SAMPLES, sync_cuda=False):
    results = []
    subset = dataset.select(range(min(n, len(dataset))))

    for row in subset:
        example = parse_row(row)
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = predict_fn(example)

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000
        result['latency_ms'] = latency_ms
        results.append(result)
    return results


def summarize_results(results):
    latencies = [result['latency_ms'] for result in results]
    correct_count = sum(result['correct'] for result in results)
    accuracy = correct_count / len(results)
    total_cost = sum(
        result['cost'] for result in results
    )

    return {
        "n_samples": len(results),
        "accuracy": accuracy,
        "correct": correct_count,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "median_latency_ms": statistics.median(latencies),
        "total_input_tokens": sum(
            result["input_tokens"] for result in results
        ),
        "total_output_tokens": sum(
            result["output_tokens"] for result in results
        ),
        "total_cost": total_cost,
        "average_cost": total_cost / len(results),
        "p95_latency_ms": statistics.quantiles(latencies, n=100)[94]
    }




def print_summary(name, summary):
    print(f"\n--- {name.upper()} ---")
    print(f"Samples: {summary['n_samples']}")
    print(
        f"Accuracy: "
        f"{summary['correct']}/{summary['n_samples']} "
        f"({summary['accuracy']:.2%})"
    )
    print(
        f"Mean latency: "
        f"{summary['mean_latency_ms']:.2f} ms"
    )
    print(
        f"Median latency: "
        f"{summary['median_latency_ms']:.2f} ms"
    )
    print(
        f"p95 latency: "
        f"{summary['p95_latency_ms']:.2f} ms"
    )
    print(
        f"Input tokens: "
        f"{summary['total_input_tokens']}"
    )
    print(
        f"Output tokens: "
        f"{summary['total_output_tokens']}"
    )
    print(
        f"Average cost per request: "
        f"${summary['average_cost']:.6f}"
    )





#========================MAIN========================

def main():

    all_summaries = []

    choice_ds = load_dataset(
        "Praveenrajus/jev-bench",
        "banking77",
        split="test",
        token=HF_TOKEN,
    )

    noul_ds = load_dataset(
        "Praveenrajus/jev-bench",
        "boolq",
        split="test",
        token=HF_TOKEN,
    )

    score_ds = load_dataset(
        "Praveenrajus/jev-bench",
        "helpsteer2_helpfulness",
        split="test",
        token=HF_TOKEN,
    )


    # ================= map datasets and models for modulairty ===================
    datasets = {
        "choice": choice_ds,
        "noul": noul_ds,
        "score": score_ds
    }

    models = {
        "jev": {
            "predict_fn": predict_jev,
            "sync_cuda": False,
        },
        "laya": {
            "predict_fn": predict_laya,
            "sync_cuda": True,
        },
        "jpt": {
            "predict_fn": predict_jpt,
            "sync_cuda": False
        },
        "qwen": {
            "predict_fn": predict_qwen,
            "sync_cuda": True,
        }
    }

    # ========================== loop over the hash maps - once for every model, once for every modality =======================
    for model_name, model_config in models.items():

        predict_fn = model_config['predict_fn']
        sync_cuda = model_config['sync_cuda']
        print(f"\n============{model_name.upper()} ==============")


        for primitive, dataset in datasets.items():
            # warm up the forward pass of local inference models
            if sync_cuda:
                warmup(
                    dataset,
                    predict_fn,
                    sync_cuda=True
                )

            # ACTUAL benchmark loop that returns a results list with every example as a json entry... 
            # run benchmark itself loops over the datasets entries so one call per model and dataset
            results = run_benchmark(
                dataset,
                predict_fn,
                sync_cuda = sync_cuda,
                n = N_SAMPLES
            )

            # save jsonl file with every single example (1 file per model and modality)
            save_results(
                results,
                model_name,
                primitive
            )

            summary = summarize_results(results)
            summary['model'] = model_name
            summary['primitive'] = primitive

            all_summaries.append(summary)

            print_summary(
                primitive,
                summary,
            )
    

    # Save a summary csv including just the summaries of all 12 runs
    save_summaries(all_summaries)

            





if __name__ == "__main__":
    main()