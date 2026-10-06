from datasets import load_dataset
from dotenv import load_dotenv
import os


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_ID = "Praveenrajus/jev-bench"


def load_banking77_train():
    return load_dataset(
        DATASET_ID,
        "banking77",
        split="train",
        token=HF_TOKEN,
    )


def load_banking77_validation():
    return load_dataset(
        DATASET_ID,
        "banking77",
        split="validation",
        token=HF_TOKEN,
    )


def load_boolq_train():
    return load_dataset(
        DATASET_ID,
        "boolq",
        split="train",
        token=HF_TOKEN,
    )


def load_boolq_validation():
    return load_dataset(
        DATASET_ID,
        "boolq",
        split="validation",
        token=HF_TOKEN,
    )


def load_helpsteer_train():
    return load_dataset(
        DATASET_ID,
        "helpsteer2_helpfulness",
        split="train",
        token=HF_TOKEN,
    )


def load_helpsteer_validation():
    return load_dataset(
        DATASET_ID,
        "helpsteer2_helpfulness",
        split="validation",
        token=HF_TOKEN,
    )