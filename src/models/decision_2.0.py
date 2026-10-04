from transformers import AutoModel
import huggingface_hub
from datasets import load_dataset
from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "vllm-sr/Decision-2.0-Kai-0.6B"

choice_ds = load_dataset(
    "Praveenrajus/jev-bench",
    "banking77",
    split="test",
    token=HF_TOKEN,
)


model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code = True,
)



state = "Is there an option to have a virtual card?"

questions = {
    "benchmark_test": {
        "type": "choice",
        "instructions": "Which banking support intent does the customer's message express?",
        "criteria": {
            "getting_virtual_card": "Getting virtual card",
            "virtual_card_not_working": "Virtual card not working",
            "get_disposable_virtual_card": "Get disposable virtual card",
        },
    }
}


result = model.system_one(
    state=state,
    questions=questions,
)


print(type(result))
print(result)
print(result.keys())
print(result["answers"])