import torch
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM
)


MODEL_ID = "Qwen/Qwen3.5-0.8B"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID)
model = model.to(device)
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are a classification system"
            "Return exactly one allowed label"
            "and nothing else"
        ),
    },
    {
        "role": "user",
        "content": (
            "Customer message:\n"
            "I was charged twice for my monthly subscription\n"
            "Allowed labels:\n"
            "billing\n"
            "technical\n"
            "sales\n"
            "return only the label."
        ),
    },
]


inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

inputs = {
    key: value.to(device) 
    for key, value in inputs.items()
}


with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )



input_length = inputs["input_ids"].shape[-1]

generated_tokens = outputs[0][input_length:]

answer = processor.decode(
    generated_tokens,
    skip_special_tokens=True,
)


print("ANSWER: ", answer)