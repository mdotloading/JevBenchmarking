"""Jev over the TypeSafe System One API (needs TYPESAFE_API_KEY in .env)."""
import os
import random
import time

import requests
from dotenv import load_dotenv
from llm2jev.prompt import options_of

load_dotenv()

URL = "https://api.typesafe.ai/v1/systemone"
MODEL_NAME = "jev-latest"
USD_PER_MILLION_INPUT_TOKENS = 0.042
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}  # 529: the API is overloaded, try again later
MAX_ATTEMPTS = 8
BACKOFF_SECONDS = (2, 60)  # first wait, longest wait; doubles each attempt, with jitter
TIMEOUT_SECONDS = (10, 120)  # connect, read


class JevAPI:
    def __init__(self):
        key = os.getenv("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY not found")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    def post_with_retry(self, payload):
        """POST once, retrying transient failures with exponential backoff.
        -> (response, request_ms), timing only the successful attempt."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            start = time.perf_counter()
            try:
                response = self.session.post(URL, json=payload, timeout=TIMEOUT_SECONDS)
                problem = f"HTTP {response.status_code}" if response.status_code in RETRY_STATUS else None
            except (requests.ConnectionError, requests.Timeout) as error:
                response, problem = None, type(error).__name__
            if problem is None:
                response.raise_for_status()
                return response, (time.perf_counter() - start) * 1000
            if attempt == MAX_ATTEMPTS:
                if response is not None:
                    response.raise_for_status()
                raise requests.ConnectionError(f"Jev API unreachable after {MAX_ATTEMPTS} attempts ({problem})")
            wait = min(BACKOFF_SECONDS[0] * 2 ** (attempt - 1), BACKOFF_SECONDS[1])
            retry_after = response.headers.get("Retry-After", "") if response is not None else ""
            if retry_after.isdigit():
                wait = max(wait, min(int(retry_after), 120))
            wait *= random.uniform(0.75, 1.25)
            print(f"  Jev API: {problem}, retry {attempt}/{MAX_ATTEMPTS - 1} in {wait:.0f}s", flush=True)
            time.sleep(wait)

    def predict(self, example):
        """-> option keys, their probabilities, input tokens, cost and the request time (without retries)."""
        payload = {"model": MODEL_NAME, "state": example["state"], "questions": {"q": example["question"]}}
        response, request_ms = self.post_with_retry(payload)
        result = response.json()
        answer = result["answers"]["q"]
        keys, _ = options_of(example["question"])
        if "noul" in answer:
            probabilities = [answer["noul"] if key == "true" else 1 - answer["noul"] for key in keys]
        else:
            probabilities = [float(answer["probabilities"].get(key, 0.0)) for key in keys]
        input_tokens = result["usage"]["input_tokens"]
        return {"keys": keys, "probabilities": probabilities, "input_tokens": input_tokens,
                "cost": input_tokens / 1e6 * USD_PER_MILLION_INPUT_TOKENS, "request_ms": request_ms}
