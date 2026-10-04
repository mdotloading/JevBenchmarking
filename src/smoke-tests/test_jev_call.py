from datasets import load_dataset
from src.data.parser import parse_row, parse_label
from dotenv import load_dotenv
import os
import requests
import json
import laya
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"




BASE_URL = "https://api.typesafe.ai/v1/systemone"


#import env variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

choice_ds = load_dataset(
    "Praveenrajus/jev-bench",
    "banking77",
    split="test",
    token=hf_token,
)


#parse row 1 of that dataset according to our parser scheme
example = parse_row(choice_ds[0])







#Test call to Jev
headers = {
    "Authorization": f'Bearer {os.getenv("TYPESAFE_API_KEY")}',
    "Content-Type": "application/json"
}


payload = {
    "model": "jev-latest",
    "state": example['state'],
    "questions": {
        "benchmark_test": example['question']
    },
}

response = requests.post(
    BASE_URL,
    headers = headers,
    json = payload,
)


print(f"Response Status: {response.status_code}")
print(f"Response: {response.json()}")




#================LAYA test==============

agent = laya.load(
    "convaiinnovations/laya",
    device = device,
)

laya_state = example['state']
laya_question = {
    "laya_benchmark_test": example["question"]
}

laya_res = agent.predict(
    laya_state,
    laya_question
)


print(laya_res)