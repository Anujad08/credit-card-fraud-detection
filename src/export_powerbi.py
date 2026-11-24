"""Score ALL transactions with the trained pipeline and write a star-schema for Power BI.
   python -m src.export_powerbi"""
import json

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from .scoring import FraudScorer



def main():
    C.POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    sc = FraudScorer()
    df = pd.read_csv(C.DATA_PATH, parse_dates=["timestamp"])
    b = F.add_previous_txn(df)
    X = F.featurize(b, sc.profiles, sc.meta)
    comp = sc.components(X)
    score = sc.hybrid(comp)
    dec = sc.decisions(score)

    # top SHAP driver per transaction (explainability column for the dashboard)
    sv = sc.explainer.shap_values(X[F.FEATURES])
    top = np.array(F.FEATURES)[sv.argmax(axis=1)]
    top_text = pd.Series(top).map(F.FEATURE_TEXT).where(score >= sc.cfg["thresholds"]["verify"], "—")

    test_idx = set(pd.read_csv(C.ARTIFACT_DIR / "test_index.csv")["row_index"])
    val_idx = set(pd.read_csv(C.ARTIFACT_DIR / "val_index.csv")["row_index"])
    split = np.where(b.index.isin(test_idx), "test", np.where(b.index.isin(val_idx), "validation", "train"))

    fact = pd.DataFrame({
        "transaction_id": b["transaction_id"].values, "user_id": b["user_id"].values,
        "timestamp": b["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").values,
        "date_key": b["timestamp"].dt.strftime("%Y%m%d").astype(int).values,
        "hour": b["timestamp"].dt.hour.values,
        "card_type": b["card_type"].values, "merchant_category": b["merchant_category"].values,
        "location": b["location"].values, "device_type": b["device_type"].values,
        "amount": b["amount"].values, "gap_minutes": np.round(np.expm1(X["gap_log"].values), 1),
        "amount_zscore": X["amount_zscore"].round(3).values,
        "device_changed": X["device_changed"].astype(int).values, "location_changed": X["location_changed"].astype(int).values,
        "is_fraud": b["is_fraud"].values, "rule_score": X["rule_score"].round(3).values,
        "xgb_prob": comp["xgb"].round(5), "iso_score": comp["iso"].round(4), "ae_score": comp["ae"].round(4),
        "risk_score": score.round(2), "decision": dec, "top_risk_driver": top_text.values,
        "model_split": split,
    })
    # business outcome columns (cost matrix) so DAX stays simple
    flagged = fact["decision"] != "APPROVE"
    fact["outcome"] = np.select(
        [(fact.is_fraud == 1) & flagged, (fact.is_fraud == 1) & ~flagged, (fact.is_fraud == 0) & flagged],
        ["TP", "FN", "FP"], default="TN")
    fact["loss_inr"] = np.select([fact.outcome == "FN", fact.outcome == "FP"], [C.COST_FN, C.COST_FP], default=0)
    fact = fact.sort_values("timestamp")
    fact.to_csv(C.POWERBI_DIR / "fact_transactions.csv", index=False)

    dates = pd.date_range(df["timestamp"].min().normalize(), df["timestamp"].max().normalize())
    pd.DataFrame({"date_key": dates.strftime("%Y%m%d").astype(int), "date": dates.strftime("%Y-%m-%d"),
                  "year": dates.year, "month_no": dates.month, "month": dates.strftime("%b"),
                  "day": dates.day, "weekday": dates.strftime("%a"), "weekday_no": dates.dayofweek + 1,
                  "week": dates.isocalendar().week.values}).to_csv(C.POWERBI_DIR / "dim_date.csv", index=False)
    pd.DataFrame({"user_id": sc.profiles["user_id"], "avg_amount": sc.profiles["mean_amount"].round(2),
                  "std_amount": sc.profiles["std_amount"].round(2), "txns_per_day": sc.profiles["txns_per_day"].round(3)}
                 ).to_csv(C.POWERBI_DIR / "dim_user.csv", index=False)

    m = json.load(open(C.ARTIFACT_DIR / "metrics.json"))
    rows = []
    for name, key in [("Rules only", "rule_only"), ("XGBoost only", "xgboost_only"), ("Hybrid (VERIFY+BLOCK)", "hybrid_verify_or_block")]:
        c = m["cost"][key]; rows.append({"model": name, "TP": c["TP"], "FP": c["FP"], "FN": c["FN"], "TN": c["TN"], "total_cost_inr": c["cost"]})
    pd.DataFrame(rows).to_csv(C.POWERBI_DIR / "model_cost_comparison.csv", index=False)
    pd.read_csv(C.ARTIFACT_DIR / "global_shap_importance.csv").to_csv(C.POWERBI_DIR / "shap_global_importance.csv", index=False)

    print(fact.groupby(["model_split", "decision"]).size().unstack(fill_value=0))
    print("files:", [p.name for p in C.POWERBI_DIR.glob("*.csv")])


if __name__ == "__main__":
    main()
