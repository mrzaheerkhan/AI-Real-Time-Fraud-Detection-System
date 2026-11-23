import pandas as pd
import random


# Number of transactions
NUMBER_OF_TRANSACTIONS = 1000


locations = [
    "Bangalore",
    "Mumbai",
    "Delhi",
    "Chennai",
    "Hyderabad",
    "Pune"
]

devices = [
    "known",
    "new"
]

merchants = [
    "Amazon",
    "Flipkart",
    "Walmart",
    "Myntra",
    "Paytm"
]


transactions = []


for i in range(NUMBER_OF_TRANSACTIONS):

    transaction_id = f"TX{1000 + i}"
    customer_id = f"C{1000 + random.randint(1, 300)}"

    amount = random.randint(500, 300000)

    location = random.choice(locations)

    device = random.choice(devices)

    hour = random.randint(0, 23)
    minute = random.randint(0, 59)

    transaction_time = f"{hour:02d}:{minute:02d}"

    merchant = random.choice(merchants)


    # -----------------------------------------
    # Fraud probability
    # -----------------------------------------

    fraud_score = 0


    # Large transaction
    if amount > 150000:
        fraud_score += 30


    # New device
    if device == "new":
        fraud_score += 25


    # Night transaction
    if hour < 5:
        fraud_score += 25


    # Random variation
    fraud_score += random.randint(0, 20)


    # Determine fraud
    if fraud_score >= 60:
        is_fraud = 1
    else:
        is_fraud = 0


    transactions.append([
        transaction_id,
        customer_id,
        amount,
        location,
        device,
        transaction_time,
        merchant,
        is_fraud
    ])


# Create DataFrame

df = pd.DataFrame(
    transactions,
    columns=[
        "transaction_id",
        "customer_id",
        "amount",
        "location",
        "device",
        "transaction_time",
        "merchant",
        "is_fraud"
    ]
)


# Save dataset

df.to_csv(
    "data/transactions.csv",
    index=False
)


print("Transaction dataset generated successfully!")
print("--------------------------------")
print("Total transactions:", len(df))
print("Fraud transactions:", df["is_fraud"].sum())
print(
    "Normal transactions:",
    len(df) - df["is_fraud"].sum()
)