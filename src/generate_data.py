import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 1. Basic configuration (exactly 100 txns per user)
N_USERS = 2000                 # number of unique card users
TXNS_PER_USER = 100            # transactions per user
N_TXNS  = N_USERS * TXNS_PER_USER   # total number of transactions = 200,000

# 2. Categorical value spaces

card_types = ["VISA", "MASTER", "RUPAY"]

merchant_categories = [
    "grocery",
    "electronics",
    "fashion",
    "online_shopping",
    "travel",
    "restaurants",
    "fuel",
    "pharmacy",
    "entertainment",
    "utilities",
    "education",
    "subscriptions",
    "gaming",
    "insurance",
    "hotel",
    "airlines",
    "digital_wallet_topup",
    "emi_payment",
    "jewellery",
    "electronics_repair",
]

locations = [
    # Metro and Tier-1
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad",
    "Chennai", "Kolkata", "Pune", "Jaipur", "Surat",

    # Tier-2 / Tier-3 and other major cities
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
    "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad",
    "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut",
    "Rajkot", "Kalyan", "Vasai", "Varanasi", "Srinagar",
    "Aurangabad", "Dhanbad", "Amritsar", "Navi_Mumbai", "Allahabad",
    "Ranchi", "Howrah", "Coimbatore", "Jabalpur", "Gwalior",
    "Vijayawada", "Jodhpur", "Madurai", "Raipur", "Kota",
    "Guwahati", "Chandigarh", "Solapur", "Hubli", "Mysore",
]

device_types = ["mobile_app", "web", "pos_terminal", "atm"]

# 3. Final schema (columns)
schema_columns = [
    "transaction_id",
    "user_id",
    "card_type",
    "timestamp",
    "amount",
    "merchant_category",
    "location",
    "device_type",
    "previous_transaction_gap",
    "is_fraud",
]

schema_columns

from datetime import datetime, timedelta

rng = np.random.default_rng(42)  # reproducible randomness

# 1) Create user_id list with exactly 100 txns per user
user_ids = np.repeat(np.arange(1, N_USERS + 1), TXNS_PER_USER)  # length = N_TXNS

# 2) Generate timestamps in a 90-day window
start_time = datetime.now() - timedelta(days=90)
# For each transaction, pick random minute offset within 90 days
minute_offsets = rng.integers(0, 90 * 24 * 60, size=N_TXNS)
timestamps = [start_time + timedelta(minutes=int(m)) for m in minute_offsets]

# 3) Probabilistic sampling of categories
card_type_col = rng.choice(card_types, size=N_TXNS, p=[0.6, 0.3, 0.1])  # VISA more common
merchant_col = rng.choice(merchant_categories, size=N_TXNS)
location_col = rng.choice(locations, size=N_TXNS)
device_col = rng.choice(device_types, size=N_TXNS, p=[0.6, 0.2, 0.15, 0.05])  # mobile dominant

# 4) Amounts ~ lognormal for realism
# mean and sigma chosen so most values look like normal card spends
amounts = rng.lognormal(mean=7.5, sigma=0.8, size=N_TXNS)  # tweak later if needed
amounts = np.round(amounts, 2)

# 5) Build initial DataFrame (no gaps / labels yet)
df = pd.DataFrame({
    "transaction_id": np.arange(1, N_TXNS + 1),
    "user_id": user_ids,
    "card_type": card_type_col,
    "timestamp": timestamps,
    "amount": amounts,
    "merchant_category": merchant_col,
    "location": location_col,
    "device_type": device_col,
})

# Sort by user and time to make behaviour realistic
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

df.head(), df.shape
df.info()
# Compute time gap (in minutes) since previous txn for the same user
df["previous_transaction_gap"] = (
    df.groupby("user_id")["timestamp"]
      .diff()
      .dt.total_seconds()
      .div(60)
)

# First transaction per user has NaN gap; set it to a large value (e.g., 1 day)
df["previous_transaction_gap"] = df["previous_transaction_gap"].fillna(60 * 24)

# Temporarily set is_fraud = 0 for all; we will overwrite with rules later
df["is_fraud"] = 0

df.head(), df.describe(include="all")
# ---------------------------
# STEP 4: RULE-BASED FRAUD INJECTION WITH 10 RULES (SIMPLIFIED)
# ---------------------------

# 0) Prepare behaviour helpers

# user average amount (for spike rule)
user_avg_amount = df.groupby("user_id")["amount"].transform("mean")

# previous location and previous device for each user
df["prev_location"] = df.groupby("user_id")["location"].shift(1)
df["prev_device"]   = df.groupby("user_id")["device_type"].shift(1)

# 1) Define each rule as a boolean mask

# 1. High amount spike vs user average
rule1 = df["amount"] > 3 * user_avg_amount

# 2. Absolute very high amount
rule2 = df["amount"] > 100000

# 3. High-velocity large transaction (small gap + high amount)
rule3 = (df["previous_transaction_gap"] < 5) & (df["amount"] > 5000)

# 4. Too many transactions in very short time (approximation)
#    If gap < 2 minutes, we treat it as high frequency behaviour
rule4 = df["previous_transaction_gap"] < 2

# 5. Risky merchant + high amount
risky_merchants = ["online_shopping", "gaming", "airlines", "jewellery"]
rule5 = df["merchant_category"].isin(risky_merchants) & (df["amount"] > 20000)

# 6. Unusually large fuel/pharmacy spend
fuel_like = ["fuel", "pharmacy"]
rule6 = df["merchant_category"].isin(fuel_like) & (df["amount"] > 10000)

# 7. New city for this user + high amount
rule7 = (
    df["prev_location"].notna()
    & (df["location"] != df["prev_location"])
    & (df["amount"] > 15000)
)

# 8. Location change + short gap (impossible travel)
rule8 = (
    df["prev_location"].notna()
    & (df["location"] != df["prev_location"])
    & (df["previous_transaction_gap"] < 30)
    & (df["amount"] > 5000)
)

# 9. New device + high amount
rule9 = (
    df["prev_device"].notna()
    & (df["device_type"] != df["prev_device"])
    & (df["amount"] > 10000)
)

# 10. New city + new device + high amount (combined high risk)
rule10 = (
    df["prev_location"].notna()
    & df["prev_device"].notna()
    & (df["location"] != df["prev_location"])
    & (df["device_type"] != df["prev_device"])
    & (df["amount"] > 15000)
)

# 2) Combine rules into a rule_score (0–1)

rule_weights = {
    "rule1": 0.10,
    "rule2": 0.15,
    "rule3": 0.10,
    "rule4": 0.05,
    "rule5": 0.10,
    "rule6": 0.05,
    "rule7": 0.10,
    "rule8": 0.10,
    "rule9": 0.10,
    "rule10": 0.15,
}

rule_score = (
    rule_weights["rule1"] * rule1.astype(int)
  + rule_weights["rule2"] * rule2.astype(int)
  + rule_weights["rule3"] * rule3.astype(int)
  + rule_weights["rule4"] * rule4.astype(int)
  + rule_weights["rule5"] * rule5.astype(int)
  + rule_weights["rule6"] * rule6.astype(int)
  + rule_weights["rule7"] * rule7.astype(int)
  + rule_weights["rule8"] * rule8.astype(int)
  + rule_weights["rule9"] * rule9.astype(int)
  + rule_weights["rule10"] * rule10.astype(int)
)

df["rule_score"] = rule_score.clip(0, 1)

# 3) Decide fraud labels based purely on rules (for synthetic data)

df["is_fraud"] = 0  # base

# We'll consider transactions with rule_score >= 0.4 as strong candidates
fraud_mask = df["rule_score"] >= 0.4
candidate_indices = df[fraud_mask].index

if len(candidate_indices) > 0:
    # sample 50% of strong candidates as actual fraud to keep fraud rate reasonable
    selected_indices = df.loc[candidate_indices].sample(frac=0.5, random_state=42).index
    df.loc[selected_indices, "is_fraud"] = 1

# 4) Clean up helper columns if you don't need them later
df = df.drop(columns=["prev_location", "prev_device"])

# Check fraud rate and some sample rows
fraud_rate = df["is_fraud"].mean()
fraud_rate, df["is_fraud"].value_counts(), df[["amount","previous_transaction_gap",
                                              "merchant_category","location",
                                              "device_type","rule_score","is_fraud"]].head(10)
# Save final synthetic dataset
df.to_csv("credit_card_data.csv", index=False)
df.shape, df["is_fraud"].mean()
