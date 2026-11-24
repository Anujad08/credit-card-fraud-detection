"""Hybrid risk scoring + SHAP reasons. Loaded once by the API (lifespan) and by the Power BI export."""
import json
import os
import pickle
from itertools import product

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import features as F

DEFAULT_WEIGHTS = {"xgb": 0.60, "iso": 0.10, "ae": 0.10, "rule": 0.20}


def decide(score: float, verify_t: float, block_t: float) -> str:
    return "BLOCK" if score >= block_t else "VERIFY" if score >= verify_t else "APPROVE"


def total_cost(y, flagged) -> dict:
    y, flagged = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool)
    fn = int((y & ~flagged).sum()); fp = int((~y & flagged).sum()); tp = int((y & flagged).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": int((~y & ~flagged).sum()),
            "cost": fn * C.COST_FN + fp * C.COST_FP}


class FraudScorer:
    def __init__(self, art=C.ARTIFACT_DIR):
        from tensorflow import keras  # imported lazily: heavy
        import shap

        art = os.fspath(art)
        self.xgb = joblib.load(f"{art}/xgb_model.pkl")
        self.iso = joblib.load(f"{art}/isolation_forest.pkl")
        self.scaler = joblib.load(f"{art}/scaler.pkl")
        self.ae = keras.models.load_model(f"{art}/autoencoder.keras", compile=False)
        self.profiles = pd.read_csv(f"{art}/user_profiles.csv")
        with open(f"{art}/last_txn_cache.pkl", "rb") as f:
            self.cache = pickle.load(f)
        with open(f"{art}/config.json") as f:
            self.cfg = json.load(f)
        self.meta = self.cfg["meta"]
        self.explainer = shap.TreeExplainer(self.xgb)
        self._prof_idx = set(self.profiles["user_id"].astype(int))

    # ---------- component scores (all normalised to 0..1) ----------
    def components(self, X: pd.DataFrame) -> dict:
        xgb_p = self.xgb.predict_proba(X[F.FEATURES])[:, 1]
        Xs = self.scaler.transform(X[F.NUM_FEATURES])
        n = self.cfg["norm"]
        iso_raw = -self.iso.score_samples(Xs)
        iso = np.clip((iso_raw - n["iso_lo"]) / (n["iso_hi"] - n["iso_lo"]), 0, 1)
        mse = np.mean((Xs - self.ae.predict(Xs, batch_size=4096, verbose=0)) ** 2, axis=1)
        ae = np.clip(mse / n["ae_scale"], 0, 1)
        return {"xgb": xgb_p, "iso": iso, "ae": ae, "rule": X["rule_score"].to_numpy()}

    def hybrid(self, comp: dict, weights=None) -> np.ndarray:
        w = weights or self.cfg["weights"]
        return 100 * sum(w[k] * comp[k] for k in w)

    def decisions(self, score):
        t = self.cfg["thresholds"]
        return np.array([decide(s, t["verify"], t["block"]) for s in np.atleast_1d(score)])

    # ---------- explainability ----------
    def reasons(self, X: pd.DataFrame, top_k=3):
        sv = self.explainer.shap_values(X[F.FEATURES])
        out = []
        for row in sv:
            order = np.argsort(-row)
            items = [{"feature": F.FEATURES[i], "text": F.FEATURE_TEXT[F.FEATURES[i]], "impact": round(float(row[i]), 4)}
                     for i in order if row[i] > 0.05][:top_k]
            # collapse duplicate wording (e.g. hour_sin + hour_cos)
            seen, uniq = set(), []
            for it in items:
                if it["text"] not in seen:
                    seen.add(it["text"]); uniq.append(it)
            out.append(uniq or [{"feature": "-", "text": "Behaviour consistent with user history", "impact": 0.0}])
        return out

    # ---------- single transaction (API path) ----------
    def score_one(self, p: dict) -> dict:
        uid = int(p["user_id"])
        ts = pd.Timestamp(p.get("timestamp") or pd.Timestamp.now())
        prev = self.cache.get(uid)  # O(1) dictionary lookup
        row = pd.DataFrame([{
            "user_id": uid, "amount": float(p["amount"]), "timestamp": ts, "card_type": p["card_type"],
            "merchant_category": p["merchant_category"], "location": p["location"], "device_type": p["device_type"],
            "prev_ts": prev["timestamp"] if prev else pd.NaT,
            "prev_location": prev["location"] if prev else None,
            "prev_device": prev["device_type"] if prev else None,
        }])
        X = F.featurize(row, self.profiles, self.meta)
        comp = self.components(X)
        score = float(self.hybrid(comp)[0])
        return {
            "risk_score": round(score, 2),
            "decision": str(self.decisions(score)[0]),
            "components": {k: round(float(v[0]), 4) for k, v in comp.items()},
            "reasons": self.reasons(X)[0],
            "known_user": uid in self._prof_idx,
            "features": {k: round(float(v), 4) for k, v in X.iloc[0].items()},
        }


def tune_hybrid(comp_val: dict, y_val) -> dict:
    """Grid-search fusion weights + APPROVE/VERIFY/BLOCK thresholds to minimise Rs cost on validation data."""
    weight_grid = [
        {"xgb": 0.60, "iso": 0.10, "ae": 0.10, "rule": 0.20},
        {"xgb": 0.70, "iso": 0.05, "ae": 0.10, "rule": 0.15},
        {"xgb": 0.50, "iso": 0.10, "ae": 0.15, "rule": 0.25},
        {"xgb": 0.80, "iso": 0.00, "ae": 0.10, "rule": 0.10},
        {"xgb": 0.40, "iso": 0.15, "ae": 0.20, "rule": 0.25},
    ]
    best = None
    for w in weight_grid:
        s = 100 * sum(w[k] * comp_val[k] for k in w)
        for vt, bt in product(range(10, 61, 2), range(40, 91, 5)):
            if bt <= vt + 5:
                continue
            c = total_cost(y_val, s >= vt)["cost"]
            # tie-break: prefer the spec's 40/70 thresholds, then the higher VERIFY cut-off (fewer alerts)
            key = (c, abs(vt - 40) + abs(bt - 70))
            if best is None or key < best[0]:
                best = (key, w, vt, bt)
    _, w, vt, bt = best
    return {"weights": w, "thresholds": {"verify": vt, "block": bt}}
