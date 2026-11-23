import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Load dataset
# ==========================================
df = pd.read_csv("data/transactions.csv")

# ==========================================
# 2. Data preprocessing (leakage-safe)
# ==========================================
# Optional chronological sort if a timestamp column exists
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

# Extract hour safely (supports "HH:MM" strings or datetime)
if "transaction_time" in df.columns:
    hour_raw = df["transaction_time"].astype(str).str.split(":").str[0]
    df["hour"] = pd.to_numeric(hour_raw, errors="coerce").fillna(0).astype(int).clip(0, 23)
elif "timestamp" in df.columns:
    df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
else:
    df["hour"] = 0

df["device_new"] = (
    df["device"].fillna("").astype(str).str.strip().str.lower() == "new"
).astype(int)

df["location_unknown"] = (
    df["location"].fillna("").astype(str).str.strip().str.lower() == "unknown"
).astype(int)

df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 5)).astype(int)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

feature_cols = ["amount", "device_new", "location_unknown", "hour", "is_night"]

# Leakage-safe historical spending deviation per customer (uses only prior txns)
if "customer_id" in df.columns:
    prior_mean = (
        df.groupby("customer_id")["amount"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    # Fallback for a customer's very first transaction
    df["spending_deviation"] = np.where(
        prior_mean > 0,
        df["amount"] / prior_mean,
        1.0,
    )
    feature_cols.append("spending_deviation")

# ==========================================
# 3. Select features & target
# ==========================================
X = df[feature_cols]
y = df["is_fraud"].astype(int)

print("Input Features:")
print(X.head(10))
print("--------------------------------")
print("Target distribution:")
print(y.value_counts())

# ==========================================
# 4. Split data (stratified when both classes exist)
# ==========================================
stratify_target = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_target,
)

print("--------------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 5. Train Random Forest model
# ==========================================
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("--------------------------------")
print("Training model...")
model.fit(X_train, y_train)
print("Training completed!")

# ==========================================
# 6. Save model & feature metadata
# ==========================================
artifact = {
    "model": model,
    "feature_cols": feature_cols,
}
joblib.dump(artifact, "fraud_model.pkl")
print("Model saved successfully to fraud_model.pkl!")

# ==========================================
# 7. Evaluate model
# ==========================================
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("--------------------------------")
print("Actual values:   ", list(y_test))
print("Predicted values:", list(y_pred))
print("Fraud probability:", [round(float(p), 3) for p in y_prob])
print("--------------------------------")
print("Accuracy: ", accuracy)
print("Precision:", precision)
print("Recall:   ", recall)
print("F1 Score: ", f1)
if y_test.nunique() > 1:
    print("ROC-AUC:  ", roc_auc_score(y_test, y_prob))
print("--------------------------------")
print("Confusion Matrix:")
print(cm)