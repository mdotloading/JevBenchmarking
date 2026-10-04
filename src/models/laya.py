import torch
import laya


device = "cuda" if torch.cuda.is_available() else "cpu"

print(
    "Loading Laya"
)

agent = laya.load(
    "convaiinnovations/laya",
    device = device
)

print(
    "Laya loaded successfully"
)


def predict_laya(example):

    state = example['state']
    question = {
        "benchmark_test": example['question']
    }


    result = agent.predict(
        state,
        question
    )

    answer = result['answers']['benchmark_test']
    primitive = example['primitive']

    if primitive == "choice":
        prediction = answer['choice']

    elif primitive == "noul":
        prediction = int(answer['noul'] >= 0.5)

    elif primitive == "score":
        prediction = round(answer['score'])

    else:
        raise ValueError(f"Unkown primitive: {primitive}")

    return {
        "model": result['model'],
        "source": example['source'],
        "prediction": prediction,
        "primitive": example['primitive'],
        "label": example['label'],
        "correct": example['label'] == prediction,
        "input_tokens": result['usage']['input_tokens'],
        "output_tokens": result['usage']['output_tokens'],
        "cost": 0.0,
        "example_id": example['id']
    }
