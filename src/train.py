"""End-to-end training:  python -m src.train
Outputs models + state files to artifacts/, figures to reports/figures/, metrics.json"""
import json
import os
import pickle
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from xgboost import XGBClassifier

from . import config as C
from . import features as F
from .scoring import decide, total_cost, tune_hybrid


def seed_everything():
    random.seed(C.RANDOM_STATE); np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)


def main():
    seed_everything()
    C.ARTIFACT_DIR.mkdir(exist_ok=True); C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: load + clean ----
    df = pd.read_csv(C.DATA_PATH, parse_dates=["timestamp"]).drop_duplicates("transaction_id").dropna()
    print(f"rows={len(df):,}  fraud={df.is_fraud.sum()} ({df.is_fraud.mean():.4%})")

    # ---- Phase 2: behavioural features ----
    meta = F.build_meta(df)
    batch = F.add_previous_txn(df)
    # user profiles must come from TRAINING rows only (no leakage into the test set)
    idx_tr, idx_tmp = train_test_split(batch.index, test_size=0.4, stratify=batch.is_fraud, random_state=C.RANDOM_STATE)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.5, stratify=batch.loc[idx_tmp, "is_fraud"], random_state=C.RANDOM_STATE)
    profiles = F.build_profiles(batch.loc[idx_tr])
    X = F.featurize(batch, profiles, meta); X.index = batch.index
    y = batch["is_fraud"]
    Xtr, Xva, Xte = X.loc[idx_tr], X.loc[idx_va], X.loc[idx_te]
    ytr, yva, yte = y.loc[idx_tr], y.loc[idx_va], y.loc[idx_te]
    print(f"train/val/test = {len(Xtr):,}/{len(Xva):,}/{len(Xte):,}  frauds {ytr.sum()}/{yva.sum()}/{yte.sum()}")

    # ---- Phase 3a: XGBoost (supervised) -- imbalance via scale_pos_weight, NOT SMOTE ----
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=spw, eval_metric="aucpr", random_state=C.RANDOM_STATE, n_jobs=-1)
    xgb.fit(Xtr[F.FEATURES], ytr)

    # ---- Phase 3b: Isolation Forest + Autoencoder (unsupervised) ----
    scaler = StandardScaler().fit(Xtr[F.NUM_FEATURES])
    S = {k: scaler.transform(v[F.NUM_FEATURES]) for k, v in {"tr": Xtr, "va": Xva, "te": Xte}.items()}
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=C.RANDOM_STATE, n_jobs=-1).fit(S["tr"])

    normal = S["tr"][ytr.values == 0]
    n_in = normal.shape[1]
    ae = keras.Sequential([
        keras.Input(shape=(n_in,)),
        keras.layers.Dense(16, activation="relu"), keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(16, activation="relu"), keras.layers.Dense(n_in),
    ])
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    ae.fit(normal, normal, epochs=15, batch_size=512, validation_split=0.1, shuffle=True, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])  # trained on NORMAL only

    # normalisation constants (so each component maps to 0..1)
    iso_tr = -iso.score_samples(S["tr"])
    ae_mse_tr = np.mean((normal - ae.predict(normal, batch_size=4096, verbose=0)) ** 2, axis=1)
    norm = {"iso_lo": float(np.percentile(iso_tr, 1)), "iso_hi": float(np.percentile(iso_tr, 99.9)),
            "ae_scale": float(np.percentile(ae_mse_tr, 99.5))}

    def comps(Xs, Sx):
        iso_raw = -iso.score_samples(Sx)
        mse = np.mean((Sx - ae.predict(Sx, batch_size=4096, verbose=0)) ** 2, axis=1)
        return {"xgb": xgb.predict_proba(Xs[F.FEATURES])[:, 1],
                "iso": np.clip((iso_raw - norm["iso_lo"]) / (norm["iso_hi"] - norm["iso_lo"]), 0, 1),
                "ae": np.clip(mse / norm["ae_scale"], 0, 1), "rule": Xs["rule_score"].to_numpy()}

    cva, cte = comps(Xva, S["va"]), comps(Xte, S["te"])

    # ---- Phase 4: hybrid score + cost-optimised thresholds (tuned on VALIDATION only) ----
    tuned = tune_hybrid(cva, yva.values)
    w, vt, bt = tuned["weights"], tuned["thresholds"]["verify"], tuned["thresholds"]["block"]
    hyb_te = 100 * sum(w[k] * cte[k] for k in w)

    # baseline: XGBoost alone with its own best cost threshold chosen on validation
    xgb_ts = np.linspace(0.05, 0.95, 91)
    t_x = min(xgb_ts, key=lambda t: total_cost(yva.values, cva["xgb"] >= t)["cost"])
    base = total_cost(yte.values, cte["xgb"] >= t_x)
    hyb = total_cost(yte.values, hyb_te >= vt)
    rule_only = total_cost(yte.values, cte["rule"] >= 0.35)
    block_only = total_cost(yte.values, hyb_te >= bt)
    decisions = pd.Series([decide(s, vt, bt) for s in hyb_te]).value_counts().to_dict()

    metrics = {
        "split_sizes": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "test_frauds": int(yte.sum()),
        "scale_pos_weight": spw,
        "roc_auc": {"xgboost": float(roc_auc_score(yte, cte["xgb"])), "hybrid": float(roc_auc_score(yte, hyb_te)),
                    "isolation_forest": float(roc_auc_score(yte, cte["iso"])), "autoencoder": float(roc_auc_score(yte, cte["ae"]))},
        "pr_auc": {"xgboost": float(average_precision_score(yte, cte["xgb"])), "hybrid": float(average_precision_score(yte, hyb_te)),
                   "isolation_forest": float(average_precision_score(yte, cte["iso"])), "autoencoder": float(average_precision_score(yte, cte["ae"]))},
        "cost": {"rule_only": rule_only, "xgboost_only": {**base, "threshold": float(t_x)},
                 "hybrid_verify_or_block": {**hyb, "verify_threshold": vt}, "hybrid_block_only": {**block_only, "block_threshold": bt}},
        "hybrid_decision_counts_test": {k: int(v) for k, v in decisions.items()},
        "weights": w, "thresholds": {"verify": vt, "block": bt},
    }
    p, r = hyb["TP"] / max(hyb["TP"] + hyb["FP"], 1), hyb["TP"] / max(hyb["TP"] + hyb["FN"], 1)
    metrics["hybrid_precision"], metrics["hybrid_recall"] = p, r
    print(json.dumps(metrics, indent=2))

    # ---- Phase 5: SHAP global plot ----
    sample = Xte[F.FEATURES].sample(min(3000, len(Xte)), random_state=C.RANDOM_STATE)
    sv = shap.TreeExplainer(xgb).shap_values(sample)
    shap.summary_plot(sv, sample, show=False, plot_size=(9, 6)); plt.tight_layout()
    plt.savefig(C.FIG_DIR / "shap_summary.png", dpi=130); plt.close()
    gi = pd.DataFrame({"feature": F.FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    gi["description"] = gi["feature"].map(F.FEATURE_TEXT)

    # figures: PR curve + cost comparison + class imbalance
    for name, sc in [("XGBoost", cte["xgb"]), ("Hybrid", hyb_te / 100)]:
        pr, rc, _ = precision_recall_curve(yte, sc); plt.plot(rc, pr, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (test)"); plt.legend()
    plt.savefig(C.FIG_DIR / "pr_curve.png", dpi=130); plt.close()
    plt.bar(["Rules only", "XGBoost only", "Hybrid"], [rule_only["cost"], base["cost"], hyb["cost"]], color=["#9aa5b1", "#4c78a8", "#2a9d8f"])
    plt.ylabel("Total cost (Rs)"); plt.title("Financial cost on test set")
    plt.savefig(C.FIG_DIR / "cost_comparison.png", dpi=130); plt.close()
    df.is_fraud.value_counts().rename({0: "Genuine", 1: "Fraud"}).plot.bar(logy=True, color=["#4c78a8", "#e4572e"])
    plt.title("Class imbalance (log scale)"); plt.tight_layout(); plt.savefig(C.FIG_DIR / "class_imbalance.png", dpi=130); plt.close()

    # ---- Phase 6: serialise models + state (solves the stateless-API problem) ----
    joblib.dump(xgb, C.ARTIFACT_DIR / "xgb_model.pkl")
    joblib.dump(iso, C.ARTIFACT_DIR / "isolation_forest.pkl")
    joblib.dump(scaler, C.ARTIFACT_DIR / "scaler.pkl")
    ae.save(C.ARTIFACT_DIR / "autoencoder.keras")
    profiles.to_csv(C.ARTIFACT_DIR / "user_profiles.csv", index=False)
    with open(C.ARTIFACT_DIR / "last_txn_cache.pkl", "wb") as f:
        pickle.dump(F.last_txn_cache(df), f)
    with open(C.ARTIFACT_DIR / "config.json", "w") as f:
        json.dump({"meta": meta, "norm": norm, "weights": w, "thresholds": {"verify": vt, "block": bt}}, f, indent=2)
    with open(C.ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    gi.to_csv(C.ARTIFACT_DIR / "global_shap_importance.csv", index=False)
    pd.Series(idx_te).to_csv(C.ARTIFACT_DIR / "test_index.csv", index=False, header=["row_index"])
    pd.Series(idx_va).to_csv(C.ARTIFACT_DIR / "val_index.csv", index=False, header=["row_index"])
    print("saved artifacts ->", C.ARTIFACT_DIR)


if __name__ == "__main__":
    main()
