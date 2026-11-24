"""Behavioural feature engineering shared by training, batch scoring and the API.

Using ONE implementation for both batch (notebook/training) and single-row (API)
prevents train/serve skew, the classic "stateless API vs stateful ML" bug.
"""
import numpy as np
import pandas as pd

from . import config as C

NUM_FEATURES = [  # used by Isolation Forest + Autoencoder (scaled, numeric only)
    "amount_log", "amount_zscore", "amount_ratio", "gap_log",
    "device_changed", "location_changed",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "user_txns_per_day",
]
FEATURES = NUM_FEATURES + ["rule_score", "merchant_code", "card_code", "device_code"]

FEATURE_TEXT = {
    "amount_log": "High transaction amount",
    "amount_zscore": "Amount far above this user's normal spend",
    "amount_ratio": "Amount is a large multiple of user's average",
    "gap_log": "Unusual time gap since previous transaction",
    "device_changed": "Device changed since last transaction",
    "location_changed": "Location changed since last transaction",
    "hour_sin": "Unusual time of day", "hour_cos": "Unusual time of day",
    "dow_sin": "Unusual day of week", "dow_cos": "Unusual day of week",
    "user_txns_per_day": "User transaction frequency profile",
    "rule_score": "Business fraud rule triggered",
    "merchant_code": "Merchant category risk",
    "card_code": "Card type", "device_code": "Device type risk",
}


def build_meta(df: pd.DataFrame) -> dict:
    """Category vocabularies, frozen at training time so the API encodes identically."""
    return {
        "card_types": sorted(df["card_type"].unique().tolist()),
        "merchant_categories": sorted(df["merchant_category"].unique().tolist()),
        "device_types": sorted(df["device_type"].unique().tolist()),
        "locations": sorted(df["location"].unique().tolist()),
    }


def build_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user historical aggregates (becomes user_profiles.csv)."""
    g = df.groupby("user_id")
    p = g["amount"].agg(mean_amount="mean", std_amount="std", n_txns="size")
    span = (g["timestamp"].max() - g["timestamp"].min()).dt.total_seconds() / 86400
    p["txns_per_day"] = p["n_txns"] / span.clip(lower=1)
    p["std_amount"] = p["std_amount"].fillna(0).clip(lower=1.0)
    return p.reset_index()


def last_txn_cache(df: pd.DataFrame) -> dict:
    """user_id -> state of that user's most recent transaction (becomes last_txn_cache.pkl)."""
    last = df.sort_values("timestamp").groupby("user_id").tail(1)
    return {
        int(r.user_id): {"timestamp": r.timestamp, "location": r.location, "device_type": r.device_type}
        for r in last.itertuples()
    }


def add_previous_txn(df: pd.DataFrame) -> pd.DataFrame:
    """Batch helper: attach previous-transaction context per user."""
    df = df.sort_values(["user_id", "timestamp"]).copy()
    g = df.groupby("user_id")
    df["prev_ts"] = g["timestamp"].shift()
    df["prev_location"] = g["location"].shift()
    df["prev_device"] = g["device_type"].shift()
    return df


def featurize(df: pd.DataFrame, profiles: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """df needs: user_id, amount, timestamp, card_type, merchant_category, location,
    device_type, prev_ts, prev_location, prev_device. Returns a frame with FEATURES."""
    d = df.merge(profiles, on="user_id", how="left")
    # cold start: unseen user -> global averages
    d["mean_amount"] = d["mean_amount"].fillna(profiles["mean_amount"].mean())
    d["std_amount"] = d["std_amount"].fillna(profiles["std_amount"].mean())
    d["txns_per_day"] = d["txns_per_day"].fillna(profiles["txns_per_day"].mean())

    gap = (d["timestamp"] - d["prev_ts"]).dt.total_seconds() / 60
    d["gap"] = gap.clip(lower=0).fillna(C.DEFAULT_GAP_MIN)
    d["device_changed"] = ((d["prev_device"].notna()) & (d["prev_device"] != d["device_type"])).astype(int)
    d["location_changed"] = ((d["prev_location"].notna()) & (d["prev_location"] != d["location"])).astype(int)

    d["amount_log"] = np.log1p(d["amount"])
    d["amount_zscore"] = ((d["amount"] - d["mean_amount"]) / d["std_amount"]).clip(-10, 50)
    d["amount_ratio"] = (d["amount"] / d["mean_amount"].clip(lower=1)).clip(upper=100)
    d["gap_log"] = np.log1p(d["gap"])
    hour = d["timestamp"].dt.hour + d["timestamp"].dt.minute / 60
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    dow = d["timestamp"].dt.dayofweek
    d["dow_sin"], d["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    d["user_txns_per_day"] = d["txns_per_day"]

    # deterministic business rules  RuleScore = sum(w_k * rule_k), capped at 1
    w = C.RULE_WEIGHTS
    d["rule_score"] = (
        w["spike"] * (d["amount"] > C.SPIKE_MULTIPLIER * d["mean_amount"])
        + w["very_large"] * (d["amount"] > C.VERY_LARGE_AMOUNT)
        + w["velocity"] * ((d["gap"] < C.VELOCITY_GAP_MIN) & (d["amount"] > C.VELOCITY_AMOUNT))
        + w["travel"] * ((d["location_changed"] == 1) & (d["gap"] < C.TRAVEL_GAP_MIN))
    ).clip(upper=1.0).astype(float)

    d["merchant_code"] = pd.Categorical(d["merchant_category"], categories=meta["merchant_categories"]).codes
    d["card_code"] = pd.Categorical(d["card_type"], categories=meta["card_types"]).codes
    d["device_code"] = pd.Categorical(d["device_type"], categories=meta["device_types"]).codes
    return d[FEATURES].astype(float).reset_index(drop=True)
