# Jev Benchmarking

Jev (hosted API) vs JPT and Decision 2.0 (local, 0.8B) and their LoRA fine-tunes, on the three Jev question
types: `choice` (pick one option), `noul` (yes/no) and `score` (ordinal rating). Data: [`Praveenrajus/jev-bench`](https://huggingface.co/datasets/Praveenrajus/jev-bench).

## Setup
```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt
```
`.env` in the repository root: `HF_TOKEN=...` (dataset) and `TYPESAFE_API_KEY=...` (Jev API).

## Benchmark
```powershell
python -m src.benchmark                         # runs only (model, dataset) pairs not in results/raw yet
python -m src.benchmark --models jpt --rerun    # force a rerun
```
Models: `jev`, `jpt`, `decision2`, plus every folder in `checkpoints/` named
`<jpt|decision2>-<dataset or modality>[-expert]-<lora|merged>` (`checkpoints/archive/` is ignored).
A fine-tune runs only on test sets of its own question type. Test sets: banking77, boolq, helpsteer2_helpfulness (ID)
and clinc150, fever_evidence, yelp5 (OOD, never trained on); first `N_SAMPLES = 500` rows each.

Outputs: `results/raw/<model>_<dataset>.jsonl` (per example) and `results/master_benchmark_matrix.csv`
(accuracy, Brier, ECE, latency, tokens, cost), rebuilt from all raw files after every run.

## Fine-tuning
```powershell
python -m src.training.train_lora --model decision2 --train-on banking77   # specialist on one dataset
python -m src.training.train_lora --model jpt --train-on choice            # expert on a question type
```
Or edit the settings at the top of [src/training/train_lora.py](src/training/train_lora.py): `MODEL`, `TRAIN_ON`,
`PROFILES` (specialist / expert hyperparameters), `BATCH_SIZE`. Output: `checkpoints/<model>-<target>[-expert]-lora`.

## Code
```text
src/data/jevbench.py     dataset registry (types, held-out sets, expert mixes), loading, gold answers
src/metrics.py           prediction rule, Brier, ECE, training loss (shared by training and benchmark)
src/models/__init__.py   load_model(family, checkpoint): "jev" | "jpt" | "decision2"
src/models/base.py       predict() shared by the local models
src/models/jev.py        Jev API client
src/models/jpt.py        JPT: label-token logits of the chat prompt
src/models/decision2.py  Decision 2.0: the model's own encoder + candidate head
src/training/train_lora.py
src/benchmark.py
```
