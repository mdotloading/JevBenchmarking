from datasets import load_dataset
import json
from dotenv import load_dotenv
import os
from parser import parse_row, parse_label

load_dotenv()
hf_token = os.getenv("HF_TOKEN")




# Datasets laden via HF datasets
choice_ds = load_dataset(
    "Praveenrajus/jev-bench",
    "banking77",
    split="test",
    token=hf_token
)

noul_ds = load_dataset(
    "Praveenrajus/jev-bench",
    "boolq",
    split="test",
    token=hf_token
)

score_ds = load_dataset(
    "Praveenrajus/jev-bench",
    "helpsteer2_helpfulness",
    split="test",
    token=hf_token
)


#print 1 row from each dataset


for name, ds in [
    ("choice", choice_ds),
    ("noul", noul_ds),
    ("score", score_ds)
]:
    row = ds[0]

    print("\n", name.upper())
    print(f"RAW ROW: {row}")

    state = json.loads(row['state'])
    question = json.loads(row['question'])


    print("STATE:", state)
    print("QUESTION", question)
    print("LABEL:", row['label'])




#Parse the rows into python objects

choice_example = parse_row(choice_ds[0])
noul_example = parse_row(noul_ds[0])
score_example = parse_row(score_ds[0])


for example in [
    choice_example,
    noul_example,
    score_example,
]:
    print("\n---")
    print("ID:", example["id"])
    print("SOURCE:", example["source"])
    print("PRIMITIVE:", example["primitive"])
    print("STATE TYPE:", type(example["state"]))
    print("QUESTION TYPE:", type(example["question"]))
    print("LABEL:", example["label"])
    print("LABEL TYPE:", type(example["label"]))



