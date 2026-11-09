from pathlib import Path
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

PROJECT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT / "data" / "clean"
OUTPUT_DIR = PROJECT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load cleaned data
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 2. Keep completed orders
completed_orders = orders[
    orders["status"].str.lower() == "completed"
].copy()

# 3. Calculate RFM values
snapshot_date = pd.Timestamp("2025-10-01")

rfm = completed_orders.groupby("customer_id").agg(
    Recency=("order_ts", lambda x: (snapshot_date - x.max()).days),
    Frequency=("order_id", "count"),
    Monetary=("order_amount", "sum")
).reset_index()

rfm["Monetary"] = rfm["Monetary"].round(2)

# 4. Scale RFM features
rfm_features = rfm[["Recency", "Frequency", "Monetary"]]

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# 5. K-Means customer segmentation
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# 6. Assign business segment labels
cluster_profiles = rfm.groupby("Cluster").agg(
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean")
)

# Identify clusters by their average behaviour.
at_risk_cluster = cluster_profiles["Recency"].idxmax()
premium_cluster = cluster_profiles["Monetary"].idxmax()
frequent_cluster = cluster_profiles["Frequency"].idxmax()

labels = {}

for cluster in cluster_profiles.index:
    if cluster == at_risk_cluster:
        labels[cluster] = "At-Risk Customers"
    elif cluster == premium_cluster:
        labels[cluster] = "Premium Customers"
    elif cluster == frequent_cluster:
        labels[cluster] = "Frequent Active Customers"
    else:
        labels[cluster] = "Regular Customers"

rfm["Segment"] = rfm["Cluster"].map(labels)

# 7. Save customer segments
segments_path = OUTPUT_DIR / "customer_segments.csv"
rfm.to_csv(segments_path, index=False)

# 8. Prepare lapse prediction target
# Recency above 60 days means lapsed.
rfm["Lapsed"] = (rfm["Recency"] > 60).astype(int)

X = rfm[["Frequency", "Monetary"]]
y = rfm["Lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 9. Train balanced Logistic Regression
lapse_model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)

lapse_model.fit(X_train, y_train)

y_pred = lapse_model.predict(X_test)

# 10. Evaluate model
print("PART B LAPSE MODEL PERFORMANCE")
print("-" * 40)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 3))
print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. Save model
model_path = OUTPUT_DIR / "lapse_model.joblib"
joblib.dump(lapse_model, model_path)

# 12. Generate retention list
rfm["Lapse_Probability"] = lapse_model.predict_proba(X)[:, 1]

retention_list = rfm[
    rfm["Lapse_Probability"] >= 0.5
][[
    "customer_id", "Recency", "Frequency",
    "Monetary", "Segment", "Lapse_Probability"
]].copy()

retention_path = OUTPUT_DIR / "retention_list.csv"
retention_list.to_csv(retention_path, index=False)

# 13. Print final results
print("\nCUSTOMER SEGMENTATION")
print("-" * 40)
print("Customers analysed:", len(rfm))
print(rfm["Segment"].value_counts().to_string())

print("\nRETENTION LIST")
print("Customers flagged:", len(retention_list))

print("\nFILES SAVED")
print(segments_path)
print(model_path)
print(retention_path)
