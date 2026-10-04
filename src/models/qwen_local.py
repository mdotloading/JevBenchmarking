import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "Qwen/Qwen3.5-0.8B"

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    device_map = "auto"
)

model.eval()


def build_prompt(example):
    primitive = example['primitive']
    state = example['state']
    question = example['question']

    if primitive == "choice":
        criteria = question['criteria']

        options = "\n".join(
            criteria.keys()
        )
        return f"""
State:
{state}

Task:
{question['instructions']}

Allowed labels: 
{options}

Return exactly one allowed label. 
Return nothing else.
""".strip()
    
    elif primitive == "noul":
        return f"""
State:
{state}

Task: 
{question['instructions']}

Return exactly:
1 for yes/true
0 for no/false

Return only 0 or 1
""".strip()
    

    elif primitive == "score":
        criteria = question['criteria']

        levels = "\n".join(
            f"{i}: {description}"
            for i, description in enumerate(criteria)
        )

        return f"""
State:
{state}

Task:
{question['instructions']}

Score levels:
{levels}

Return exactly one integer score.
Return nothing else.
""".strip()
    
    else:
        raise ValueError(
            f"Unknown primitive: {primitive}"
        )
    




def predict_qwen(example):
    prompt = build_prompt(example)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a decision system."
                "Follow the requested output format exactly"
            ),
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to(model.device)

    input_tokens = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    generated_tokens = outputs[0][input_tokens:]

    output_tokens = len(generated_tokens)

    raw_answer = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


    primitive = example['primitive']

    if primitive == "choice":
        prediction = raw_answer.strip()

    elif primitive == "noul":
        try:
            prediction = int(raw_answer.strip())
        except ValueError:
            prediction = None
    
    elif primitive == "score":
        try:
            prediction = int(raw_answer.strip())
        except ValueError:
            prediction = None

    
    return {
        "example_id": example["id"],
        "source": example["source"],
        "primitive": primitive,
        "model": MODEL_ID,
        "prediction": prediction,
        "label": example["label"],
        "correct": prediction == example["label"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": 0.0,
    }