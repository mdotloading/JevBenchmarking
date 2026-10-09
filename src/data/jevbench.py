"""JevBench datasets (https://huggingface.co/datasets/Praveenrajus/jev-bench): registry, loading, gold answers."""
import json
import os
import random

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_ID = "Praveenrajus/jev-bench"

# Every usable config by question type ("modality").
DATASETS_BY_MODALITY = {
    "choice": ("banking77", "clinc150", "massive", "ledgar", "go_emotions", "mmlu", "arc_challenge", "mnli"),
    "noul": ("boolq", "fever_evidence", "paws", "civil_comments", "sms_spam", "strategyqa_closed",
             "strategyqa_grounded", "mnli_binary"),
    "score": ("sst5", "yelp5", "helpsteer2_helpfulness", "helpsteer2_verbosity", "stsb", "measuring_hate_speech"),
}
MODALITY_OF = {name: modality for modality, names in DATASETS_BY_MODALITY.items() for name in names}

# The benchmark's test sets: ID = seen by the specialists/experts, OOD = held out (zero-shot).
BENCHMARK_DATASETS = {"banking77": "ID", "boolq": "ID", "helpsteer2_helpfulness": "ID",
                      "clinc150": "OOD", "fever_evidence": "OOD", "yelp5": "OOD"}

# Never trained on. strategyqa_closed asks the same questions as strategyqa_grounded.
HELD_OUT = frozenset({"clinc150", "fever_evidence", "yelp5", "arc_challenge", "measuring_hate_speech",
                      "strategyqa_grounded", "strategyqa_closed"})

# Training mix of a modality expert: coherent tasks per type, no held-out sources.
EXPERT_MIX = {
    "choice": ("banking77", "massive", "ledgar"),
    "noul": ("boolq", "paws", "mnli_binary"),
    "score": ("helpsteer2_helpfulness", "sst5"),
}


def mnli_as_noul(ds):
    """mnli as a yes/no task: entailment -> 1, contradiction -> 0, neutral dropped."""
    question = json.dumps({"type": "noul", "instructions": "Based only on `premise`, is `hypothesis` definitely true?"})
    ds = ds.filter(lambda row: row["label"] in ("entailment", "contradiction"))
    return ds.map(lambda row: {"primitive": "noul", "question": question,
                               "label": "1" if row["label"] == "entailment" else "0"})


# Configs derived from another config: name -> (source config, transform).
DERIVED = {"mnli_binary": ("mnli", mnli_as_noul)}


def source_of(name):
    return DERIVED[name][0] if name in DERIVED else name


def parse_row(row):
    """A jev-bench row (state/question served as JSON strings) -> a Python example."""
    label = row["label"] if row["primitive"] == "choice" else int(row["label"])
    return {"id": row["id"], "source": row["source"], "primitive": row["primitive"],
            "state": json.loads(row["state"]), "question": json.loads(row["question"]), "label": label}


def load_examples(name, split, limit=None, seed=None):
    """Parsed examples of one split: the first `limit` rows, or with a seed a random sample of `limit` rows."""
    ds = load_dataset(DATASET_ID, source_of(name), split=split, token=HF_TOKEN)
    if name in DERIVED:
        ds = DERIVED[name][1](ds)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [parse_row(row) for row in ds]


def shuffle_options(question, row_id, seed):
    """A choice question with its options in a random but reproducible order (training only), so a model
    reads the options instead of learning "the answer is usually at position X"."""
    items = list(question["criteria"].items())
    random.Random(f"{seed}:{row_id}").shuffle(items)
    return {**question, "criteria": dict(items)}


def gold_key(example):
    """The gold answer as an option key: noul 0/1 -> "false"/"true", score -> "0".."K-1", choice -> the label."""
    label = example["label"]
    if example["primitive"] == "noul":
        return {0: "false", 1: "true"}[label]
    return str(label)
