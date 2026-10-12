# Jev Benchmarking

Jev (hosted API) vs JPT and Decision 2.0 (local, 0.8B) and their LoRA fine-tunes, on the three Jev question
types: `choice` (pick one option), `noul` (yes/no) and `score` (ordinal rating). Data: [`Praveenrajus/jev-bench`](https://huggingface.co/datasets/Praveenrajus/jev-bench).
A separate [tabular comparison](#tabular-comparison-classical-ml-vs-zero-shot-decision-models) pits Jev and
Decision 2.0 (zero-shot) against logistic regression, SVM and XGBoost on public tables.

## Setup
```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt
pip install xgboost   # only for the tabular benchmark
```
`.env` in the repository root: `HF_TOKEN=...` (dataset, private adapters) and `TYPESAFE_API_KEY=...` (Jev API).

`checkpoints/` is not in git. The Decision 2.0 fine-tunes are private Hub repos; fetch them with
```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); from huggingface_hub import snapshot_download; [snapshot_download(f'mdotloading/decision2-{t}-lora', local_dir=f'checkpoints/decision2-{t}-lora') for t in ['choice-expert', 'noul-expert', 'score-expert', 'banking77', 'boolq', 'helpsteer2_helpfulness']]"
```

## Models
| model (benchmark name) | family | kind | training | source |
|---|---|---|---|---|
| `jev` | Jev (`jev-latest`) | base | - | TypeSafe System One API, $0.042 / 1M input tokens |
| `jpt` | JPT ([`kirp/jpt-0.8b`](https://huggingface.co/kirp/jpt-0.8b)) | base | - | Hub |
| `decision2` | Decision 2.0 ([`vllm-sr/Decision-2.0-Eos-0.8B`](https://huggingface.co/vllm-sr/Decision-2.0-Eos-0.8B), pinned to `3594047`) | base | - | Hub |
| `decision2_choice_expert_lora` | Decision 2.0 | expert | LoRA r64, 1 epoch, 2,500 rows each of banking77, massive, ledgar | `mdotloading/decision2-choice-expert-lora` |
| `decision2_noul_expert_lora` | Decision 2.0 | expert | LoRA r64, 1 epoch, 2,500 rows each of boolq, paws, mnli_binary | `mdotloading/decision2-noul-expert-lora` |
| `decision2_score_expert_lora` | Decision 2.0 | expert | LoRA r64, 1 epoch, 2,500 rows each of helpsteer2_helpfulness, sst5 | `mdotloading/decision2-score-expert-lora` |
| `decision2_banking77_lora` | Decision 2.0 | specialist | LoRA r16, 3 epochs (kept epoch 1), 8,000 rows | `mdotloading/decision2-banking77-lora` |
| `decision2_boolq_lora` | Decision 2.0 | specialist | LoRA r16, 3 epochs (kept epoch 2), 8,000 rows | `mdotloading/decision2-boolq-lora` |
| `decision2_helpsteer2_helpfulness_lora` | Decision 2.0 | specialist | LoRA r16, 3 epochs (kept epoch 2), 8,000 rows | `mdotloading/decision2-helpsteer2_helpfulness-lora` |
| `jpt_banking77_merged` | JPT | specialist | LoRA merged into the weights | local `checkpoints/` only |
| `jpt_boolq_merged` | JPT | specialist | LoRA merged into the weights | local `checkpoints/` only |
| `jpt_helpsteer2_helpfulness_merged` | JPT | specialist | LoRA merged into the weights | local `checkpoints/` only |

An **expert** is tuned on several datasets of one question type (`EXPERT_MIX` in `src/data/jevbench.py`), a
**specialist** on one dataset. Every Decision 2.0 fine-tune also retrains the candidate head, and keeps the epoch
with the best validation Brier. `checkpoints/archive/` holds older JPT LoRAs and experts from before the training
refactor (other expert mixes, temperature scaling); the benchmark ignores it.

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

## Results
All 12 models, 36 (model, test set) pairs, 500 rows each (run of 2026-10-12). Accuracy uses the prediction rule of
`src/metrics.py` (score: rounded expected level); Brier is the multi-class sum (0 = certain and right, 2 = certain
and wrong); ECE is top-label over 10 bins. For fine-tunes, the bracket is the accuracy change in points against the
family's base model; **bold** is the best accuracy on that test set. With 500 rows, accuracy differences under
about 4 points are noise.

**choice** (banking77: 77 banking intents; clinc150: 150 intents, held out)

| model | banking77 acc | Brier | ECE | clinc150 acc | Brier | ECE |
|---|---|---|---|---|---|---|
| Jev (base) | 0.802 | 0.326 | 0.104 | **0.892** | 0.169 | 0.038 |
| JPT (base) | 0.646 | 0.540 | 0.155 | 0.644 | 0.535 | 0.103 |
| JPT specialist `jpt_banking77_merged` | 0.852 (+20.6) | 0.228 | 0.049 | 0.718 (+7.4) | 0.441 | 0.049 |
| Decision 2.0 (base) | 0.836 | 0.235 | 0.037 | 0.852 | 0.223 | 0.039 |
| Decision 2.0 expert `decision2_choice_expert_lora` | 0.870 (+3.4) | 0.198 | 0.043 | 0.842 (-1.0) | 0.251 | 0.045 |
| Decision 2.0 specialist `decision2_banking77_lora` | **0.904 (+6.8)** | 0.156 | 0.038 | 0.870 (+1.8) | 0.202 | 0.046 |

**noul** (boolq: yes/no about a passage; fever_evidence: is a claim supported, held out)

| model | boolq acc | Brier | ECE | fever_evidence acc | Brier | ECE |
|---|---|---|---|---|---|---|
| Jev (base) | **0.918** | 0.126 | 0.007 | **0.970** | 0.052 | 0.022 |
| JPT (base) | 0.782 | 0.327 | 0.083 | 0.900 | 0.143 | 0.018 |
| JPT specialist `jpt_boolq_merged` | 0.842 (+6.0) | 0.224 | 0.045 | 0.774 (-12.6) | 0.292 | 0.106 |
| Decision 2.0 (base) | 0.766 | 0.347 | 0.088 | 0.922 | 0.105 | 0.045 |
| Decision 2.0 expert `decision2_noul_expert_lora` | 0.818 (+5.2) | 0.264 | 0.057 | 0.936 (+1.4) | 0.107 | 0.026 |
| Decision 2.0 specialist `decision2_boolq_lora` | 0.832 (+6.6) | 0.254 | 0.093 | 0.924 (+0.2) | 0.117 | 0.039 |

**score** (helpsteer2_helpfulness: 5-level helpfulness of a response; yelp5: 5-star review rating, held out)

| model | helpsteer2_helpfulness acc | Brier | ECE | yelp5 acc | Brier | ECE |
|---|---|---|---|---|---|---|
| Jev (base) | 0.308 | 0.841 | 0.253 | **0.670** | 0.508 | 0.185 |
| JPT (base) | 0.300 | 0.753 | 0.084 | 0.566 | 0.560 | 0.097 |
| JPT specialist `jpt_helpsteer2_helpfulness_merged` | **0.364 (+6.4)** | 0.677 | 0.040 | 0.400 (-16.6) | 0.634 | 0.112 |
| Decision 2.0 (base) | 0.290 | 0.826 | 0.170 | 0.512 | 0.685 | 0.182 |
| Decision 2.0 expert `decision2_score_expert_lora` | 0.280 (-1.0) | 0.706 | 0.022 | 0.420 (-9.2) | 0.621 | 0.125 |
| Decision 2.0 specialist `decision2_helpsteer2_helpfulness_lora` | 0.326 (+3.6) | 0.682 | 0.044 | 0.460 (-5.2) | 0.678 | 0.152 |

**Reading it**
- Jev is the most accurate model on 4 of 6 test sets. The exceptions are banking77, where the Decision 2.0 specialist
  wins (0.904), and helpsteer2, where the JPT specialist wins (0.364). Untuned, Decision 2.0 beats Jev only on banking77
  (0.836 vs 0.802); it trails by 4 to 5 points on clinc150 and fever_evidence and by 16 on yelp5.
- Decision 2.0 fine-tunes gain in-distribution (specialists +6.8 / +6.6 / +3.6, experts +3.4 / +5.2 / -1.0) and keep
  their held-out accuracy on choice and noul (clinc150 -1.0 to +1.8, fever_evidence +0.2 to +1.4).
- JPT specialists gain in-distribution too, but forget more: fever_evidence drops from 0.900 to 0.774 and yelp5 from
  0.566 to 0.400. Only clinc150 improves (0.644 to 0.718), and from a weak base.
- Score is hard for every model (helpsteer2 accuracy 0.28 to 0.36 on 5 levels). Tuning mostly improves calibration
  there (Decision 2.0 helpsteer2 ECE 0.170 to 0.022 / 0.044, Brier 0.826 to 0.706 / 0.682) rather than accuracy, and
  every score fine-tune loses accuracy on yelp5.

**Speed and cost** (base models; local models on an RTX 4060 8 GB, Jev is the API round trip)

| test set | Jev p50 / p95 ms | JPT p50 / p95 ms | Decision 2.0 p50 / p95 ms | input tokens (Jev / JPT / Decision 2.0) | Jev cost |
|---|---|---|---|---|---|
| banking77 | 226 / 274 | 189 / 203 | 324 / 337 | 1689 / 959 / 1788 | $0.035 |
| clinc150 | 236 / 296 | 247 / 259 | 577 / 649 | 2535 / 1351 / 3001 | $0.053 |
| boolq | 218 / 257 | 98 / 118 | 100 / 117 | 437 / 235 / 228 | $0.009 |
| fever_evidence | 223 / 275 | 90 / 100 | 89 / 98 | 381 / 161 / 148 | $0.008 |
| helpsteer2_helpfulness | 227 / 280 | 156 / 242 | 161 / 250 | 937 / 741 / 742 | $0.020 |
| yelp5 | 224 / 271 | 105 / 152 | 107 / 153 | 529 / 328 / 336 | $0.011 |

Jev cost 0.137 USD for all 3,000 requests. Decision 2.0's choice latency grows with the option list
(clinc150: 150 options, about 3,000 tokens). The Decision 2.0 fine-tunes run as unmerged LoRA adapters, which adds
about 30 to 115 ms per request (boolq: 100 to 141 ms); the JPT fine-tunes are merged and run at base speed.

## Fine-tuning
```powershell
python -m src.training.train_lora --model decision2 --train-on banking77   # specialist on one dataset
python -m src.training.train_lora --model jpt --train-on choice            # expert on a question type
```
Or edit the settings at the top of [src/training/train_lora.py](src/training/train_lora.py): `MODEL`, `TRAIN_ON`,
`PROFILES` (specialist / expert hyperparameters), `BATCH_SIZE`. Output: `checkpoints/<model>-<target>[-expert]-lora`.
Held-out sets can never be trained on. The specialist profile now defaults to `epochs=1`; the Decision 2.0
specialists above were trained with `epochs=3`.

## Tabular comparison: classical ML vs zero-shot decision models
```powershell
python -m src.tabular_benchmark                                              # all three tables, all models
python -m src.tabular_benchmark --datasets car --models xgb jev --n-test 50  # quick check
```
Separate from the main benchmark (it writes only to `results/tabular/`). It asks: on ordinary tabular data, how many
labelled rows does a classical model need to match Jev and the untuned Decision 2.0, which see none? Three public
OpenML tables, one per question type: `adult` (noul), `obesity` (choice, 7 options) and `car` (score, 4 ordered
levels). Each row becomes a System One request: its columns are the `state` and its target column is the question.
The same 500 held-out rows score every model; logistic regression, SVM and XGBoost (library defaults, no tuning) are
trained on 16, 64, 256, 1024 and all remaining rows (16 to 1024 averaged over 3 random subsets). Metrics are the main
benchmark's (accuracy, Brier, top-label ECE, score answer = rounded expected level); outputs are
`results/tabular/matrix.csv` (also macro F1, ECE, latency, tokens, cost) and `results/tabular/raw/`.

Each cell is **accuracy / Brier** (higher accuracy, lower Brier is better). With 500 test rows, accuracy differences
under about 4 points are noise.

**adult** (noul: income above $50K (census record); 2 options, 48842 rows, 48342 left for training)

| model | trained on 0 rows | 16 rows | 64 rows | 256 rows | 1024 rows | all rows |
|---|---|---|---|---|---|---|
| class frequencies only | 0.760 / 0.365 | | | | | |
| Jev (zero-shot) | 0.806 / 0.261 | | | | | |
| Decision 2.0 (zero-shot) | 0.782 / 0.405 | | | | | |
| Logistic regression | | 0.802 / 0.282 | 0.820 / 0.246 | 0.825 / 0.239 | 0.855 / 0.206 | 0.844 / 0.196 |
| SVM (last column: 5000-row subsample) | | 0.722 / 0.463 | 0.789 / 0.280 | 0.817 / 0.251 | 0.849 / 0.215 | 0.854 / 0.205 |
| XGBoost | | 0.759 / 0.346 | 0.789 / 0.298 | 0.809 / 0.279 | 0.843 / 0.209 | 0.870 / 0.170 |

**obesity** (choice: weight category (lifestyle and body measurements); 7 options, 2111 rows, 1611 left for training)

| model | trained on 0 rows | 16 rows | 64 rows | 256 rows | 1024 rows | all rows |
|---|---|---|---|---|---|---|
| class frequencies only | 0.166 / 0.856 | | | | | |
| Jev (zero-shot) | 0.778 / 0.331 | | | | | |
| Decision 2.0 (zero-shot) | 0.104 / 1.157 | | | | | |
| Logistic regression | | 0.425 / 0.744 | 0.577 / 0.564 | 0.733 / 0.387 | 0.863 / 0.256 | 0.878 / 0.222 |
| SVM | | 0.150 / 0.886 | 0.539 / 0.603 | 0.737 / 0.357 | 0.897 / 0.144 | 0.918 / 0.115 |
| XGBoost | | 0.410 / 0.741 | 0.661 / 0.463 | 0.843 / 0.238 | 0.939 / 0.091 | 0.958 / 0.069 |

**car** (score: how acceptable a car is (price, upkeep, size, safety); 4 ordered levels, 1728 rows, 1228 left for training)

| model | trained on 0 rows | 16 rows | 64 rows | 256 rows | 1024 rows | all rows |
|---|---|---|---|---|---|---|
| class frequencies only | 0.700 / 0.458 | | | | | |
| Jev (zero-shot) | 0.630 / 0.455 | | | | | |
| Decision 2.0 (zero-shot) | 0.416 / 0.642 | | | | | |
| Logistic regression | | 0.705 / 0.373 | 0.799 / 0.277 | 0.848 / 0.197 | 0.896 / 0.129 | 0.908 / 0.122 |
| SVM | | 0.319 / 0.493 | 0.737 / 0.319 | 0.863 / 0.163 | 0.973 / 0.037 | 0.976 / 0.036 |
| XGBoost | | 0.671 / 0.439 | 0.827 / 0.278 | 0.900 / 0.136 | 0.976 / 0.032 | 0.984 / 0.028 |

Car is ordinal, so MAE (levels off) and quadratic weighted kappa (QWK) say more than accuracy:

| model | MAE | QWK |
|---|---|---|
| class frequencies only | 0.416 | 0.000 |
| Jev | 0.444 | 0.550 |
| Decision 2.0 | 0.788 | 0.353 |
| Logistic regression, all rows | 0.092 | 0.907 |
| SVM, all rows | 0.026 | 0.973 |
| XGBoost, all rows | 0.016 | 0.985 |


**Reading it**
- **Class frequencies only** is the floor: always predict the training class shares.
- **adult:** Jev (0.806 / 0.261) is level with logistic regression on 16 rows and is overtaken at about 64 rows (0.820 / 0.246). XGBoost needs about 1024 rows to pass it, and wins clearly with all rows (0.870 / 0.170). Decision 2.0 is slightly more accurate than the baseline (0.782 vs 0.760) but much worse calibrated (Brier 0.405, above the baseline's 0.365).
- **obesity:** Jev (0.778) beats every classical model up to 64 rows and logistic regression and SVM up to 256. XGBoost passes it at 256 rows (0.843 / 0.238). Decision 2.0 is below the 0.166 baseline: see the notes below.
- **car:** the table is easy for classical models: logistic regression and XGBoost beat Jev with 16 rows, SVM with 64. Jev's accuracy (0.630) is below the class-frequency baseline (0.700) and its Brier (0.455) is equal to it, though its QWK (0.550) shows it gets the order of levels right.
- With all rows, XGBoost is best or close to it on every table. This is the expected result for tabular data, so the useful number is the crossover row count, not the final score.
- Jev and Decision 2.0 are also calibrated differently: Jev's ECE is 0.05 to 0.10, Decision 2.0's up to 0.44.

**Where the options come from.** Only the target values come from the table: the `adult` income labels, the 7
`obesity` categories and the 4 `car` classes. Everything the model reads about them is written by hand in the
`DATASETS` settings of [src/tabular_benchmark.py](src/tabular_benchmark.py): the question text, the option
descriptions, and for `car` the ordering (`unacc < acc < good < vgood`; the table stores ordered levels but does not
name the direction). The `obesity` descriptions use the standard BMI ranges, except that the two overweight levels
("lower" and "upper" part of 25 to 29.9) are a guess, because the dataset does not define them. Results depend on
these wordings.

**Why Decision 2.0 is poor here.** On obesity it puts 331 of 500 rows into `Overweight_Level_II` and 142 into
`Obesity_Type_III`, so it scores below chance, while Jev spreads its answers over all seven. On car it chooses the best
level ("Very good") for 181 of 500 cars. The model is small (0.8B), and the obesity task needs BMI from height and
weight, which is arithmetic on the numbers. It has not been checked whether option order or wording contributes.
It is a zero-shot result: tuning Decision 2.0 on a few hundred rows of each table would be the fair next comparison.

## Repository layout
```text
src/data/jevbench.py       dataset registry (types, held-out sets, expert mixes), loading, gold answers
src/metrics.py             prediction rule, Brier, ECE, training loss (shared by training and benchmark)
src/models/__init__.py     load_model(family, checkpoint): "jev" | "jpt" | "decision2"
src/models/base.py         predict() shared by the local models
src/models/jev.py          Jev API client
src/models/jpt.py          JPT: label-token logits of the chat prompt
src/models/decision2.py    Decision 2.0: the model's own encoder + candidate head
src/training/train_lora.py LoRA fine-tuning (specialists and experts)
src/benchmark.py           main benchmark
src/tabular_benchmark.py   classical ML vs Jev / Decision 2.0 on public tables (self-contained)

checkpoints/               fine-tunes found by the benchmark (not in git); archive/ is ignored
results/raw/               per-example records of the main benchmark (not in git)
results/master_benchmark_matrix.csv
results/tabular/           matrix.csv and raw/ of the tabular benchmark
```
