import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

#env laden
load_dotenv()

keys = [
    "TYPESAFE_API_KEY",
    "OPENROUTER_API_KEY",
    "HF_TOKEN"
]

# keys ausgeben lassen
print("Keys:\n")

for key in keys:
    value = os.getenv(key)

    if value:
        print(f"{key} found")
    else:
        print(f"{key} missing")


#HuggingFAce check

api = HfApi(
    token=os.getenv("HF_TOKEN")
)

user = api.whoami()

print(f"Authenticated as: {user["name"]}")



import platform 
import torch

print("\n SYSTEM")
print(f"OS: {platform.system()}")
print(f"Python {platform.python_version()}")
print(f"PyTorch {torch.__version__}")

print("\n ACCELERATOR")
print(f"CUDA available: {torch.cuda.is_available()}")


if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
    
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    print(f"VRAM: {vram_gb:.2f} GB")

else: 
    print("No CUDA GPU available")