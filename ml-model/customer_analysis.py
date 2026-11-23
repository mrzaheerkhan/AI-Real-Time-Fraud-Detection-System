import pandas as pd

# Load transaction dataset
df = pd.read_csv("data/transactions.csv")

# Calculate customer behaviour
customer_stats = df.groupby("customer_id").agg(
    average_amount=("amount", "mean"),
    maximum_amount=("amount", "max"),
    total_transactions=("transaction_id", "count")
).reset_index()

# Display customer analysis
print("CUSTOMER BEHAVIOUR ANALYSIS")
print(customer_stats.head(10))

# Save customer behaviour data
customer_stats.to_csv("data/customer_analysis.csv", index=False)

print("\nCustomer analysis completed successfully!")

# Merge customer statistics with original transactions
df = df.merge(customer_stats, on="customer_id", how="left")

# Calculate spending deviation
df["spending_deviation"] = (
    df["amount"] / df["average_amount"]
)

# Identify unusual transactions
df["unusual_transaction"] = (
    df["spending_deviation"] > 2
).astype(int)

# Display results
print("\nUNUSUAL TRANSACTION ANALYSIS")

print(df[
    ["customer_id", "amount", "average_amount",
     "spending_deviation", "unusual_transaction"]
].head(10))

# Save updated dataset
df.to_csv("data/customer_transactions.csv", index=False)