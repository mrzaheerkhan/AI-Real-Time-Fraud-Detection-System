import pandas as pd

# Read the transaction dataset
df = pd.read_csv("data/transactions.csv")

print("Original Dataset")
print("--------------------------------")
print(df)

# Convert transaction time into hour
df["hour"] = df["transaction_time"].str.split(":").str[0].astype(int)

# Create new device feature
df["device_new"] = (df["device"].str.lower() == "new").astype(int)

# Create unknown location feature
df["location_unknown"] = (df["location"].str.lower() == "unknown").astype(int)

# Create night transaction feature
df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 5)).astype(int)

print("--------------------------------")
print("Dataset After Feature Engineering")
print(df)

print("--------------------------------")
print("ML Features")
print(df[[
    "amount",
    "device_new",
    "location_unknown",
    "hour",
    "is_night",
    "is_fraud"
]])