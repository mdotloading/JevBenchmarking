import os
from dotenv import load_dotenv
import requests

load_dotenv()

headers = {
    "Authorization": f"Bearer {os.getenv("TYPESAFE_API_KEY")}",
    "Content-Type": "application/json",
}

url = "https://api.typesafe.ai/v1/systemone"


payload = {
    "model": "jev-latest",
    "state": {
        "subject": "Duplicate payment",
        "message": "I was charged twice for my monthly subscription"
    },
    "questions": {
        "routing": {
            "type": "choice",
            "instructions": (
                "Which department should handle this customer request"
            ),
            "criteria": {
                "billing": "Payments, invoices, refunds, and duplicate charges.",
                "technical": "Software bugs, login problems, and technical failures.",
                "sales": "New contracts, pricing, and product purchases.",
            },
        },
    },
}



response = requests.post(
    url,
    headers = headers,
    json= payload
)


print("STATUS:", response.status_code)
print(response.json())