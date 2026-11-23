import csv
print("AI Fraud Detection System")
print("--------------------------------")


def calculate_fraud_risk(transaction):

    risk_score = 0

    # Rule 1: High transaction amount
    if transaction["amount"] > 100000:
        risk_score += 30

    # Rule 2: New device
    if transaction["device"].lower() == "new":
        risk_score += 25

    # Rule 3: Unknown location
    if transaction["location"].lower() == "unknown":
        risk_score += 25

    # Rule 4: Unusual transaction time
    hour = int(transaction["transaction_time"].split(":")[0])

    if 0 <= hour < 5:
        risk_score += 20

    # Determine risk level
    if risk_score >= 60:
        risk_level = "HIGH"
    elif risk_score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return risk_score, risk_level


# Transaction data
transaction = {
    "transaction_id": "TX1001",
    "customer_id": "C1001",
    "amount": 250000,
    "location": "unknown",
    "device": "new",
    "transaction_time": "02:30",
    "merchant": "Amazon"
}


# Display transaction
print("Transaction ID:", transaction["transaction_id"])
print("Customer ID:", transaction["customer_id"])
print("Amount:", transaction["amount"])
print("Location:", transaction["location"])
print("Device:", transaction["device"])
print("Transaction Time:", transaction["transaction_time"])
print("Merchant:", transaction["merchant"])


# Run fraud detection
risk_score, risk_level = calculate_fraud_risk(transaction)


# Display result
print("--------------------------------")
print("Fraud Risk Score:", risk_score)
print("Risk Level:", risk_level)

print("--------------------------------")
print("Reading transaction dataset")

with open("data/transactions.csv", "r") as file:

    reader = csv.DictReader(file)

    for transaction in reader:
        print(transaction)