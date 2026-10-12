"""Tabular benchmark: classical ML (logistic regression, SVM, XGBoost) vs the zero-shot decision models (Jev,
Decision 2.0 release) on public tables, one per question type.

    python -m src.tabular_benchmark                                              # every dataset and model below
    python -m src.tabular_benchmark --datasets car --models xgb jev --n-test 50  # quick check

Each table row is parsed into a System One request: state = the row's fields, question = the target column as a
noul / choice / score question. Classical models train on N_TRAIN rows (a learning curve) of the rows outside the
test set; the decision models see no training rows. Every model is scored on the same test rows with the main
benchmark's rules (score answer = rounded expected level; Brier; top-label ECE), plus macro F1 and, for score, MAE
and quadratic weighted kappa.

Self-contained (no src imports except a Windows workaround for Decision 2.0): scikit-learn, xgboost, pandas,
requests, transformers; TYPESAFE_API_KEY in the environment or .env for Jev.
Outputs results/tabular/raw/<model>_<dataset>.jsonl (decision model requests + answers, reused while the requests
are unchanged) and results/tabular/matrix.csv.
"""
import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # e.g. Colab: set os.environ["TYPESAFE_API_KEY"] instead
    pass

# ============================================================
# SETTINGS
# ============================================================
N_TEST = 500                         # test rows per dataset, the same rows for every model
N_TRAIN = (16, 64, 256, 1024, None)  # classical training rows (learning curve); None = every non-test row
REPEATS = 3                          # random training subsets per size below the full pool (metrics averaged)
SVM_MAX_TRAIN = 5000                 # SVC is O(n^2): larger training sets are subsampled to this
SEED = 42
OUT = Path("results/tabular")

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_USD_PER_MILLION_INPUT_TOKENS = 0.042
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
DECISION2_ID = "vllm-sr/Decision-2.0-Eos-0.8B"
DECISION2_REVISION = "3594047d69f476f1d01cf84c593e213fc3a4dfe0"  # the pin in src/models/decision2.py

# One OpenML table per question type. rename/values only make the state readable; every model gets the same columns.
DATASETS = {
    "adult": dict(  # noul: 48,842 people, 24% positive
        openml_id=1590, modality="noul", positive=">50K", drop=["fnlwgt"],  # fnlwgt: census sampling weight
        rename={"education-num": "years_of_education", "marital-status": "marital_status",
                "capital-gain": "capital_gain_usd", "capital-loss": "capital_loss_usd",
                "hours-per-week": "hours_worked_per_week", "native-country": "native_country"},
        instructions="Based on this 1994 US census record, does the person earn more than $50,000 a year?",
        criteria={"true": "Income above $50K a year", "false": "Income of $50K a year or less"}),
    "obesity": dict(  # choice: 2,111 people, 7 classes (77% of rows are SMOTE-synthetic, per the dataset's paper)
        openml_id=46597, modality="choice",
        rename={"Height": "height_m", "Weight": "weight_kg",
                "family_history_with_overweight": "family_history_of_overweight",
                "FAVC": "eats_high_calorie_food_often", "FCVC": "vegetables_with_meals_1_never_to_3_always",
                "NCP": "main_meals_per_day", "CAEC": "eats_between_meals", "SMOKE": "smokes",
                "CH2O": "water_per_day_1_low_to_3_high", "SCC": "monitors_calories",
                "FAF": "physical_activity_0_none_to_3_most_days", "TUE": "screen_time_0_low_to_2_high",
                "CALC": "drinks_alcohol", "MTRANS": "usual_transport"},
        instructions="Which weight category does this person fall into?",
        criteria={"Insufficient_Weight": "Underweight (BMI below 18.5)",
                  "Normal_Weight": "Normal weight (BMI 18.5 to 24.9)",
                  "Overweight_Level_I": "Overweight, lower part of BMI 25 to 29.9",
                  "Overweight_Level_II": "Overweight, upper part of BMI 25 to 29.9",
                  "Obesity_Type_I": "Obesity class I (BMI 30 to 34.9)",
                  "Obesity_Type_II": "Obesity class II (BMI 35 to 39.9)",
                  "Obesity_Type_III": "Obesity class III (BMI 40 or more)"}),
    "car": dict(  # score: 1,728 cars, 4 ordered levels (70% unacceptable)
        openml_id=40975, modality="score", levels={"unacc": 0, "acc": 1, "good": 2, "vgood": 3},
        rename={"buying": "buying_price", "maint": "maintenance_cost", "persons": "seats",
                "lug_boot": "luggage_space", "safety": "safety_rating"},
        values={"vhigh": "very high", "med": "medium", "5more": "5 or more", "more": "more than 4"},
        instructions="How acceptable is this car to buy, judging by its price, running costs, size and safety?",
        criteria=["Unacceptable", "Acceptable", "Good", "Very good"]),
}
COLUMNS = ["dataset", "modality", "model", "n_train", "accuracy", "accuracy_std", "macro_f1", "brier", "ece", "mae",
           "qwk", "fit_s", "median_latency_ms", "mean_input_tokens", "total_cost_usd", "n_test"]


# ============================================================
# Tables -> System One requests
# ============================================================
def load_table(name):
    """-> (features, class index per row, class keys in option order). Column names and categorical values are
    made readable here, so the decision models and the classical models see exactly the same table."""
    spec = DATASETS[name]
    bunch = fetch_openml(data_id=spec["openml_id"], as_frame=True)
    known = bunch.target.notna().to_numpy()
    X = bunch.data[known].drop(columns=spec.get("drop", [])).rename(columns=spec.get("rename", {}))
    X = X.reset_index(drop=True)
    values = spec.get("values", {})
    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].astype(object).map(lambda v: values.get(v, v))
    target = bunch.target[known].astype(str).str.strip().reset_index(drop=True)
    if spec["modality"] == "noul":
        if spec["positive"] not in set(target):
            raise SystemExit(f"{name}: no target value {spec['positive']!r} in {sorted(set(target))}")
        keys, index = ["false", "true"], (target == spec["positive"]).astype(int)
    elif spec["modality"] == "score":
        keys, index = [str(level) for level in range(len(spec["criteria"]))], target.map(spec["levels"])
    else:
        keys = [k for k in spec["criteria"] if k in set(target)] + sorted(set(target) - set(spec["criteria"]))
        index = target.map({key: i for i, key in enumerate(keys)})
    if index.isna().any():
        raise SystemExit(f"{name}: target values without a level: {sorted(set(target[index.isna()]))}")
    return X, index.to_numpy().astype(int), keys


def question_of(spec, keys):
    """The target column as a System One question."""
    criteria = spec["criteria"]
    if spec["modality"] == "choice":  # every observed class, with its description when there is one
        criteria = {key: criteria.get(key, key.replace("_", " ")) for key in keys}
    return {"type": spec["modality"], "instructions": spec["instructions"], "criteria": criteria}


def row_to_state(row):
    """One table row -> a System One state: its fields as JSON values, missing values left out, floats rounded."""
    state = {}
    for column, value in row.items():
        if pd.isna(value):
            continue
        if isinstance(value, (float, np.floating)):
            value = int(value) if float(value).is_integer() else round(float(value), 2)
        elif isinstance(value, np.integer):
            value = int(value)
        state[column] = value if isinstance(value, (int, float)) else str(value)
    return state


# ============================================================
# Classical models
# ============================================================
CLASSICAL = {  # library defaults, no tuning
    "logreg": lambda: LogisticRegression(max_iter=5000),
    "svm": lambda: SVC(probability=True, random_state=SEED),  # RBF kernel, Platt-scaled probabilities
    "xgb": lambda: XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, tree_method="hist",
                                 random_state=SEED),
}


def preprocessor(X):
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
         categorical)])


def fit_proba(make_model, X_train, y_train, X_test, n_classes):
    """Train on the rows given -> test probabilities over all n_classes (0 for classes missing from training;
    a small subsample can miss a rare class, and XGBoost needs consecutive labels)."""
    present = np.unique(y_train)
    proba = np.zeros((len(X_test), n_classes))
    if len(present) == 1:
        proba[:, present[0]] = 1.0
        return proba
    model = make_pipeline(preprocessor(X_train), make_model())
    model.fit(X_train, np.searchsorted(present, y_train))
    proba[:, present] = model.predict_proba(X_test)
    return proba


def subsample(rows, y, n, seed):
    """n of `rows`, stratified by class when possible."""
    if n >= len(rows):
        return rows
    try:
        return train_test_split(rows, train_size=n, stratify=y[rows], random_state=seed)[0]
    except ValueError:  # a class too rare to stratify
        return np.random.default_rng(seed).choice(rows, n, replace=False)


# ============================================================
# Decision models (zero-shot)
# ============================================================
class Jev:
    """Jev over the System One API, retrying transient failures with backoff (as src/models/jev.py does)."""

    def __init__(self):
        key = os.getenv("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("TYPESAFE_API_KEY not set")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    def system_one(self, state, question):
        """-> (answer, input tokens, request ms of the successful attempt)."""
        payload = {"model": JEV_MODEL, "state": state, "questions": {"q": question}}
        for attempt in range(8):
            start = time.perf_counter()
            try:
                response = self.session.post(JEV_URL, json=payload, timeout=(10, 120))
            except (requests.ConnectionError, requests.Timeout):
                response = None
            latency_ms = (time.perf_counter() - start) * 1000
            if response is not None and response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                result = response.json()
                return result["answers"]["q"], result["usage"]["input_tokens"], latency_ms
            time.sleep(min(2 * 2 ** attempt, 60) * random.uniform(0.75, 1.25))
        raise RuntimeError("Jev API unreachable after 8 attempts")


class Decision2:
    """The Decision 2.0 release (no adapter) through its own System One runtime."""

    def __init__(self):
        if os.name == "nt":  # the release's file-hash check builds Windows paths; the repo's loader patches that
            import src.models.decision2  # noqa: F401
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(DECISION2_ID, revision=DECISION2_REVISION,
                                               code_revision=DECISION2_REVISION, trust_remote_code=True)

    def system_one(self, state, question):
        start = time.perf_counter()
        result = self.model.system_one(state=state, questions={"q": question})
        latency_ms = (time.perf_counter() - start) * 1000
        return result["answers"]["q"], result["usage"]["input_tokens"], latency_ms


DECISION_MODELS = {"jev": Jev, "decision2": Decision2}


def answer_probabilities(answer, keys):
    """A System One answer -> probabilities in class-key order (noul keys are ["false", "true"])."""
    if "noul" in answer:
        return np.array([1 - answer["noul"], answer["noul"]])
    if "probabilities" not in answer:
        raise ValueError(f"no probabilities in answer {answer}")
    p = np.array([float(answer["probabilities"].get(key, 0.0)) for key in keys])
    return p / p.sum()


def run_decision_model(name, dataset, examples, keys, clients):
    """Answers for every example; read from results/tabular/raw when that file holds exactly these requests."""
    path = OUT / "raw" / f"{name}_{dataset}.jsonl"
    requests_wanted = [(e["id"], e["state"], e["question"]) for e in examples]
    if path.is_file():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [(r["id"], r["state"], r["question"]) for r in records] == requests_wanted:
            return records
    if name not in clients:
        clients[name] = DECISION_MODELS[name]()
    client = clients[name]
    if name != "jev":
        client.system_one(examples[0]["state"], examples[0]["question"])  # warmup: kernels, caches
    records = []
    for i, example in enumerate(examples, 1):
        answer, input_tokens, latency_ms = client.system_one(example["state"], example["question"])
        records.append({**example, "probabilities": answer_probabilities(answer, keys).tolist(),
                        "input_tokens": input_tokens, "latency_ms": latency_ms})
        if i % 100 == 0:
            print(f"  {name}: {i}/{len(examples)}", flush=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return records


# ============================================================
# Scoring (src/metrics.py rules, in numpy)
# ============================================================
def metrics_of(P, gold, modality, n_bins=10):
    P = P / P.sum(axis=1, keepdims=True)
    if modality == "score":  # the rounded expected level
        predicted = np.clip(np.rint(P @ np.arange(P.shape[1])), 0, P.shape[1] - 1).astype(int)
    else:
        predicted = P.argmax(axis=1)
    confidence = P[np.arange(len(P)), predicted]
    correct = (predicted == gold).astype(float)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = sum((bins == b).mean() * abs(confidence[bins == b].mean() - correct[bins == b].mean())
              for b in range(n_bins) if (bins == b).any())
    result = {"accuracy": correct.mean(), "macro_f1": f1_score(gold, predicted, average="macro", zero_division=0),
              "brier": ((P - np.eye(P.shape[1])[gold]) ** 2).sum(axis=1).mean(), "ece": ece}
    if modality == "score":
        result |= {"mae": np.abs(predicted - gold).mean(),
                   "qwk": cohen_kappa_score(gold, predicted, weights="quadratic")}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    models = ["prior", *CLASSICAL, *DECISION_MODELS]
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--models", nargs="+", choices=models, default=models)
    p.add_argument("--n-test", type=int, default=N_TEST)
    args = p.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    rows, clients = [], {}
    for dataset in args.datasets:
        spec, modality = DATASETS[dataset], DATASETS[dataset]["modality"]
        X, y, keys = load_table(dataset)
        pool, test = train_test_split(np.arange(len(X)), test_size=args.n_test, stratify=y, random_state=SEED)
        gold, question = y[test], question_of(spec, keys)
        shares = np.bincount(y, minlength=len(keys)) / len(y)
        print(f"\n=== {dataset}: {modality}, {len(X)} rows x {X.shape[1]} features, test {len(test)}, "
              f"train pool {len(pool)}; class shares {dict(zip(keys, shares.round(3).tolist()))}", flush=True)
        print(f"Example request: {json.dumps({'state': row_to_state(X.iloc[test[0]]), 'questions': {'q': question}})}",
              flush=True)

        def add(model, n_train, results, **extra):
            summary = {key: float(np.mean([r[key] for r in results])) for key in results[0]}
            summary["accuracy_std"] = float(np.std([r["accuracy"] for r in results]))
            rows.append({"dataset": dataset, "modality": modality, "model": model, "n_train": n_train,
                         **summary, **extra, "n_test": len(test)})
            print(f"  {model:<10} n_train={n_train:<6} acc={summary['accuracy']:.3f} brier={summary['brier']:.3f} "
                  f"ece={summary['ece']:.3f}", flush=True)

        if "prior" in args.models:  # always the training class shares: the floor every model has to beat
            prior = np.bincount(y[pool], minlength=len(keys)) / len(pool)
            add("prior", len(pool), [metrics_of(np.tile(prior, (len(test), 1)), gold, modality)])
        sizes = sorted({min(n or len(pool), len(pool)) for n in N_TRAIN})
        for name in (m for m in CLASSICAL if m in args.models):
            for size in sizes:
                results = []
                for repeat in range(1 if size == len(pool) else REPEATS):
                    train = subsample(pool, y, size, SEED + repeat)
                    if name == "svm":
                        train = subsample(train, y, SVM_MAX_TRAIN, SEED + repeat)
                    start = time.perf_counter()
                    try:
                        P = fit_proba(CLASSICAL[name], X.iloc[train], y[train], X.iloc[test], len(keys))
                    except ValueError as error:
                        print(f"  {name} n_train={size}: {error}", flush=True)
                        continue
                    results.append({**metrics_of(P, gold, modality), "fit_s": time.perf_counter() - start})
                if results:
                    add(name, len(train), results)
        for name in (m for m in DECISION_MODELS if m in args.models):
            examples = [{"id": f"{dataset}/{i}", "state": row_to_state(X.iloc[i]), "question": question} for i in test]
            records = run_decision_model(name, dataset, examples, keys, clients)
            tokens = [r["input_tokens"] for r in records]
            add(name, 0, [metrics_of(np.array([r["probabilities"] for r in records]), gold, modality)],
                median_latency_ms=statistics.median(r["latency_ms"] for r in records),
                mean_input_tokens=statistics.fmean(tokens),
                total_cost_usd=sum(tokens) / 1e6 * JEV_USD_PER_MILLION_INPUT_TOKENS if name == "jev" else 0.0)

    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    table.to_csv(OUT / "matrix.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep=""))
    print(f"\nMatrix ({len(table)} rows): {OUT / 'matrix.csv'}")


if __name__ == "__main__":
    main()
