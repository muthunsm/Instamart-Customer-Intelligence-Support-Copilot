from pathlib import Path
import sqlite3
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT / "data" / "raw"
CLEAN_DIR = PROJECT / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load raw data
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")

# 2. Remove exact duplicate rows
customers = customers.drop_duplicates().copy()
products = products.drop_duplicates().copy()
orders = orders.drop_duplicates().copy()

# 3. Clean customer data
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)
customers.loc[
    ~customers["age"].between(18, 60), "age"
] = pd.NA
customers["age"] = customers["age"].fillna(
    customers["age"].median()
)

customers["gender"] = (
    customers["gender"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .str.title()
)
customers.loc[
    customers["gender"].isin(["", "Nan", "None"]),
    "gender"
] = "Unknown"

customers["city"] = (
    customers["city"]
    .astype(str)
    .str.strip()
    .str.title()
)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# 4. Resolve repeated order IDs
# Keep the final completed/cancelled record instead of
# a pending record when both exist.
final_ids = orders.loc[
    orders["status"].astype(str).str.lower().ne("pending"),
    "order_id"
].dropna().unique()

orders = orders[
    ~(
        orders["status"].astype(str).str.lower().eq("pending")
        & orders["order_id"].isin(final_ids)
    )
].copy()

orders = orders.drop_duplicates(
    subset=["order_id"], keep="last"
).copy()

# 5. Clean order values
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

orders["quantity"] = pd.to_numeric(
    orders["quantity"], errors="coerce"
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)

products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# Fill missing order amounts using product price,
# quantity and discount.
price_lookup = products.set_index(
    "product_id"
)["price_inr"]

missing_amount = orders["order_amount"].isna()

calculated_amount = (
    orders["product_id"].map(price_lookup)
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
)

orders.loc[missing_amount, "order_amount"] = (
    calculated_amount[missing_amount]
)

# 6. Clean delivery times
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

completed_mask = (
    orders["status"].astype(str).str.lower() == "completed"
)

delivery_median = orders.loc[
    completed_mask, "delivery_minutes"
].median()

orders.loc[
    completed_mask & orders["delivery_minutes"].isna(),
    "delivery_minutes"
] = delivery_median

# 7. Clean timestamps
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# 8. Save cleaned CSV files
customers.to_csv(
    CLEAN_DIR / "customers.csv", index=False
)
products.to_csv(
    CLEAN_DIR / "products.csv", index=False
)
orders.to_csv(
    CLEAN_DIR / "orders.csv", index=False
)

# 9. Create SQLite database
db_path = PROJECT / "data" / "instamart.db"

with sqlite3.connect(db_path) as connection:
    customers.to_sql(
        "customers", connection,
        if_exists="replace", index=False
    )
    products.to_sql(
        "products", connection,
        if_exists="replace", index=False
    )
    orders.to_sql(
        "orders", connection,
        if_exists="replace", index=False
    )

# 10. Print validation results
print("PART A DATA FOUNDATION COMPLETED")
print("-" * 40)
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))
print("Order statuses:")
print(orders["status"].value_counts().to_string())
print("Missing customer values:", customers.isna().sum().sum())
print("Missing product values:", products.isna().sum().sum())
print("Missing order amounts:", orders["order_amount"].isna().sum())
print("Duplicate order IDs:", orders["order_id"].duplicated().sum())
print("Database:", db_path)
print("Clean CSV files:", CLEAN_DIR)
