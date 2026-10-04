import os 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = os.getenv("OPENROUTER_API_KEY")
)


response = client.chat.completions.create(
    model = "qwen/qwen3.8-27b:free",
    messages = [
        {
            "role": "system",
            "content": (
                "You are a classification system."
                "Return only the requested answer"
            ),
        },
        {
            "role": "user",
            "content": (
                "Customer message:\n"
                "I was charged twice for my monthly subscription.\n"
                "Choose exactly one department:\n"
                "billing\n"
                "technical\n"
                "sales\n"

                "Answer with only the department name.")

        },
    ],
    max_tokens = 10,
    temperature=0,
)


print(response.choices[0].message)