import requests
import os
from dotenv import load_dotenv


load_dotenv()


type_safe_token = os.getenv("TYPESAFE_API_KEY")



if not type_safe_token:
    raise RuntimeError("TYPESAFE_API_KEY not found")




URL = "https://api.typesafe.ai/v1/systemone"




headers = {
    "Authorization": f"Bearer {type_safe_token}",
    "Content-Type": "application/json"
}





def predict_jev(example):
    """
    Extract the JSON response payload in a unified format, regardless of the modality
    """

    # Pass Payload for POST request
    payload = {
        "model": "jev-latest",
        "state": example['state'],
        "questions": {
            "benchmark_test": example['question']
        }
    }

    #await response
    response = requests.post(
        URL,
        headers = headers,
        json = payload
    )


    response.raise_for_status()
    result = response.json()

    #extract the results for benchmark_test
    answer = result['answers']['benchmark_test']
    #decide which of the 3 modalities we are dealing with
    primitive = example['primitive']

    if primitive == "choice":
        prediction = answer['choice']

    elif primitive == "noul":
        prediction = int(answer['noul'] >= 0.5)

    elif primitive == "score":
        prediction = round(answer['score'])
    
    else:
        ValueError(f"Unknown primitive: {primitive}")

    return {
        "model": result['model'],
        "source": example['source'],
        "prediction": prediction,
        "label": example["label"],
        "primitive": example['primitive'],
        "correct": prediction == example['label'],
        "input_tokens": result['usage']['input_tokens'],
        "output_tokens": result['usage']['output_tokens'],
        "cost": (result['usage']['input_tokens'] / 1e6) * 0.042,
        "example_id": example['id']
    }

    