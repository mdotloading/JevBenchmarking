import requests



URL = "http://127.0.0.1:8080/v1/systemone"

headers = {
    "Content-Type": "application/json"
}



def predict_jpt(example):

    payload = {
        "state": example['state'],

        "questions":{
            "benchmark_test": example['question']
        },
    }

    response = requests.post(
        URL,
        headers = headers,
        json = payload
    )

    response.raise_for_status()
    result = response.json()

    answer = result['answers']['benchmark_test']
    primitive = example['primitive']

    if primitive == "choice":
        prediction = answer['choice']

    elif primitive == "noul":
        prediction = int(answer['noul'] >= 0.5)

    elif primitive == "score":
        prediction = round(answer['score'])

    else:
        raise ValueError(f"Unknown primitive: {primitive}")


    return {
    "example_id": example["id"],
    "source": example["source"],
    "primitive": primitive,
    "model": result["model"],
    "prediction": prediction,
    "label": example["label"],
    "correct": prediction == example["label"],
    "input_tokens": result["usage"]["input_tokens"],
    "output_tokens": result["usage"]["output_tokens"],
    "cost": 0.0,
}

