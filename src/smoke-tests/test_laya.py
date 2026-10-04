import laya
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Laya...")


agent = laya.load(
    "convaiinnovations/laya",
    device=device
    )


print("Laya loaded successfully.")


# state = {
#     "subject": "Duplicate payment",
#     "message": "I was charged twice for my monthly subscription"
# }

# questions = {
#     "routing":{
#         "type":"choice",
#         "instructions": "Which department should handle this customer request?",
#         "criteria":{
#             "billing": "Payments, invoices, refunds and duplicate charges",
#             "technical": "Software bugs, login problems and technical failures",
#             "sales": "New contracts, pricing, product purchases"
#         },
#     }
# }

#================================================= NOUL ======================================================


# state = {
#     "account_age_days": 1,
#     "number_of_purchases": 5,
#     "different_cards_used": 3,
#     "chargebacks": 1
# }


# questions = {
#     "fraud_review":{
#         "type": "noul",
#         "instructions": ("Should this account be manually reviewed"
#         "for potential fraud"
#         ),
#     }
# }

#=============================================== Score =======================================================


state = {
    "message": (
        "Our production payment integration has been "
        "completely unavailable for two hours. "
        "Customers cannot complete purchases."
    )
}

questions = {
    "severity": {
        "type": "score",
        "instructions": (
            "Rate the severity of this support incident."
        ),
        "criteria": [
            "Informational, no meaningful impact.",
            "Minor inconvenience.",
            "Moderate degradation.",
            "Major business impact.",
            "Critical production outage.",
        ],
    }
}



result = agent.predict(
    state,
    questions,
)

print(result)