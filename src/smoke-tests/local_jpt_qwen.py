import requests


url = "http://127.0.0.1:8080/v1/systemone"

headers = {
    "Content-Type": "application/json"
}


payload = {
    "state": {
        "subject": "duplicate payment",
        "message": ("I was charged twice"
                    "for my monthly subscription"
                    ),
    },
    "questions": {
        "routing": {
            "type": "choice",
            "instructions": ("Determine which department is responsible"
                             "for this request"
            ),
            "criteria": {
                "billing": "Payments, invoices, refunds, and duplicate charges.",
                "technical": "Software bugs, login problems, and technical failures.",
                "sales": "New contracts, pricing, and product purchases."
            }
            
        }
    }
}


response = requests.post(
    url,
    headers = headers,
    json = payload,
)

print(f"STATUS: {response.status_code}")
print(response.json())