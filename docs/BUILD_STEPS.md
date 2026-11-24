# End-to-End Build Guide: Hybrid Credit Card Fraud Detection (+ Power BI)

This guide follows the same phases as your project flow PDF (Phases 1–8), keeps the same tech stack
(Python, XGBoost, Isolation Forest, Autoencoder, SHAP, FastAPI, Streamlit, JSONL audit log) and adds
**Power BI** as an analytics/BI layer. Every command below was run on your 200,000-row dataset and works.

```
CSV (200k txns) ─► Features ─► XGBoost + IsoForest + Autoencoder + Rules ─► Hybrid score 0-100
                                                     │                         │
                              SHAP reasons ◄─────────┘          APPROVE / VERIFY / BLOCK
                                   │                                          │
        FastAPI  /score_transaction ──► Streamlit (live demo)      fact/dim CSVs ──► POWER BI (analytics)
                    │
              fraud_audit.jsonl
```

| Layer | Purpose | Audience |
|---|---|---|
| FastAPI + Streamlit | Real-time scoring of ONE transaction | Fraud operations / live demo |
| **Power BI** (new) | Historical analytics, fraud trends, model & cost performance, investigation | Risk managers / business |

---

## Step 0 – Prerequisites (30 min)

- **Python 3.10, 3.11 or 3.12** (TensorFlow does not yet install on every newer Python version, especially on Windows, so 3.11 is the safe pick). Check: `python --version`
- **Git** and a **GitHub account**. Check: `git --version`
- **Power BI Desktop** (free, **Windows only**): Microsoft Store → "Power BI Desktop". On Mac, use a Windows VM or Power BI Service in a browser (limited authoring).
- VS Code (optional but used in the demo playbook).

## Step 1 – Create the project and virtual environment

```bash
cd credit-card-fraud-detection          # the folder you downloaded
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Mac/Linux:      source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
**Expected:** installs finish without errors (TensorFlow is the big one, 3–5 minutes).
**Always run commands from the repo root** (the folder containing `src/`, `api/`, `dashboard/`). Most "No module named src" errors come from running from inside a sub-folder.

## Step 2 – Data (Phase 1: dataset)

`data/credit_card_data.csv` is already in place: 200,000 rows, 2,000 users × 100 transactions, 239 frauds (0.1195%), 90 days.
Columns: `transaction_id, user_id, card_type, timestamp, amount, merchant_category, location, device_type, previous_transaction_gap, is_fraud, rule_score`.

Quick check:
```bash
python -c "import pandas as pd; d=pd.read_csv('data/credit_card_data.csv'); print(d.shape, d.is_fraud.mean())"
```
→ `(200000, 11) 0.001195`

## Step 3 – Preprocessing + behavioural features (Phases 1–2) → `src/features.py`

What it does (explain this in interviews):
- **Per-user profile** (`user_profiles.csv`): mean/std of amount, transactions per day. Built **from training rows only**, so there is no leakage into the test set.
- **Behavioural features:** amount Z-score vs the user's own average, amount ratio, log time-gap since previous transaction, **device changed**, **location changed**, **cyclic time** (hour & weekday as sin/cos), user velocity.
- **Business rules** (from your report §5.1.2): amount > 3× user average, amount > ₹100,000, velocity (gap < 5 min and amount > ₹5,000), impossible travel (location change within 30 min) → weighted `RuleScore`, capped at 1.
- **One function is used for training AND for the API**, which prevents train/serve skew (your "stateless API vs stateful ML" point).
- **Imbalance:** no SMOTE. XGBoost uses `scale_pos_weight` (≈ 838 here).

## Step 4 – Train the hybrid pipeline (Phases 3–6) → `src/train.py`

```bash
python -m src.train
```
Takes ~20 seconds. It performs:
1. Stratified **60/20/20 train / validation / test** split (143 / 48 / 48 frauds).
2. **XGBoost** (supervised) with `scale_pos_weight`.
3. **Isolation Forest** + **Autoencoder** (Keras, trained on *normal* transactions only; high reconstruction MSE = anomaly).
4. **Hybrid risk score** = `100 × (0.6·XGB + 0.1·IsoForest + 0.1·Autoencoder + 0.2·Rules)`; the weights and the VERIFY/BLOCK thresholds are chosen by **grid-search on the validation set** to minimise `Cost = FN×₹5000 + FP×₹200`, then reported on the untouched **test set**.
5. **SHAP** global summary plot.
6. Saves to `artifacts/`: `xgb_model.pkl, isolation_forest.pkl, scaler.pkl, autoencoder.keras, user_profiles.csv, last_txn_cache.pkl, config.json, metrics.json` and figures to `reports/figures/`.

**Results you should see** (test set, 40,000 txns, 48 frauds):

| Model | Frauds caught | False alerts | Total cost |
|---|---|---|---|
| Rules only | 48/48 | 2,425 | ₹485,000 |
| XGBoost only | 48/48 | 115 | ₹23,000 |
| **Hybrid** | 48/48 | 114 | **₹22,800** |

> **Be honest in interviews:** the report PDF quotes ₹23,600 (hybrid) vs ₹29,000 (XGBoost). Your re-run gives different numbers because the split, rule weights and thresholds differ. Quote the numbers *your* run prints (`artifacts/metrics.json`), not the report's. The hybrid advantage here is modest because this synthetic data is very easy for XGBoost (ROC-AUC 0.999). The hybrid's real value is catching *novel* patterns the labels never showed. If asked, say that directly, since interviewers respect it.
>
> The weight formula in your PDF was garbled (`$` only), so the fusion weights above are my choice, tuned on validation data. Explain them as "tuned by cost on a validation set".

## Step 5 – FastAPI backend (Phase 7) → `api/app.py`

```bash
uvicorn api.app:app --reload          # terminal 1 (leave running)
```
Open http://127.0.0.1:8000/docs (Swagger UI). Key points:
- `lifespan` loads all models **once** at startup (not per request).
- **Pydantic** schema rejects bad input with HTTP **422** before it touches the models.
- Endpoint is plain `def`, not `async def`, so CPU-bound inference runs in a thread pool and does not block the event loop.
- Every call is appended to `logs/fraud_audit.jsonl` through a `RotatingFileHandler`.
- Endpoints: `POST /score_transaction`, `GET /audit_log`, `GET /health`, `GET /meta`.

Test with curl (Mac/Linux) or the Swagger "Try it out" button:
```bash
curl -X POST http://127.0.0.1:8000/score_transaction -H "Content-Type: application/json" \
 -d '{"user_id":42,"amount":95000,"location":"Chennai","device_type":"pos_terminal","merchant_category":"electronics"}'
```

## Step 6 – Streamlit dashboard (Phase 8) → `dashboard/streamlit_app.py`

```bash
streamlit run dashboard/streamlit_app.py     # terminal 2
```
It has a transaction form, Plotly risk gauge, APPROVE/VERIFY/BLOCK alert, SHAP reasons, per-model score bars and a live audit-log tab. It contains **no ML code**, only HTTP calls to the API.

Three demo scenarios (use the radio buttons; verified results):

| Scenario | Input | Score | Decision |
|---|---|---|---|
| 🟢 Normal | ₹1,500, Mumbai, mobile_app | 3.5 | APPROVE |
| 🟡 Medium | ₹25,000, Delhi, web | 39.4 | VERIFY |
| 🔴 Fraud | ₹95,000, Chennai, pos_terminal | 76.0 | BLOCK |

## Step 7 – Automated tests

```bash
python -m pytest -q tests
```
Expected: **7 passed** (health, approve, block, monotonic score, 422 validation, cold-start unknown user, audit log). Mention "unit + integration + validation testing" exactly as in your report §5.3.

## Step 8 – ⭐ Power BI layer (the new tech stack)

### 8.1 Generate the BI dataset
```bash
python -m src.export_powerbi
```
Scores **all 200,000** transactions with the trained pipeline and writes a **star schema** to `powerbi/data/`:

| File | Grain | Key columns |
|---|---|---|
| `fact_transactions.csv` | 1 row / transaction | amount, is_fraud, risk_score, decision, top_risk_driver (SHAP), outcome (TP/FP/FN/TN), loss_inr, model_split |
| `dim_date.csv` | 1 row / day | year, month, weekday, week |
| `dim_user.csv` | 1 row / user | avg_amount, txns_per_day |
| `model_cost_comparison.csv` | 3 rows | rules vs XGBoost vs hybrid, with confusion counts and cost |
| `shap_global_importance.csv` | 1 row / feature | mean_abs_shap |

`model_split` = train / validation / test. **Filter model-performance pages to `test`**, otherwise the numbers are optimistic (the model has seen training rows).

### 8.2 Build the report in Power BI Desktop
Follow **`powerbi/POWERBI_GUIDE.md`**. It contains the exact Power Query steps, relationships, ready-to-paste **DAX measures**, the 4 report pages with visual-by-visual layout, and a theme file (`powerbi/fraud_theme.json`).

### 8.3 Save and keep in the repo
Save as `powerbi/Fraud_Detection_Dashboard.pbix`. Export screenshots to `docs/images/` (File → Export → PDF, or Win+Shift+S) and embed them in the README. Recruiters look at the pictures first.

## Step 9 – Push to GitHub

```bash
git init
git add .
git status                       # check: no .venv, no artifacts/, no logs/ (all in .gitignore)
git commit -m "Hybrid credit card fraud detection: ML pipeline, FastAPI, Streamlit, Power BI"
git branch -M main
```
Create an **empty** repo on github.com (no README/licence), then:
```bash
git remote add origin https://github.com/<your-username>/credit-card-fraud-detection.git
git push -u origin main
```
(or with GitHub CLI: `gh repo create credit-card-fraud-detection --public --source=. --push`).

Checklist before you push:
- The 17 MB CSV is fine for GitHub (hard limit 100 MB per file, warning at 50 MB). `powerbi/data/*.csv` is git-ignored (31 MB, regenerated by Step 8.1); the `.pbix` is small enough to commit.
- Models in `artifacts/` are ignored; anyone can recreate them in 20 s with `python -m src.train`. If you want a demo to work right after cloning, remove `artifacts/*` from `.gitignore` and commit the folder (it is only a few MB).
- Add repo **topics**: `fraud-detection xgboost shap fastapi streamlit power-bi machine-learning`.
- Put 2–3 screenshots (Streamlit decision, Power BI overview, SHAP plot) at the top of the README.
- Never commit secrets or `.env` files.

## Step 10 – How to present it (matches your Demonstration Playbook PDF)

1. Slide/diagram: architecture (above).
2. Terminal 1 `uvicorn api.app:app` → "models load once at startup".
3. Terminal 2 `streamlit run ...` → "UI has zero ML logic".
4. Run scenarios 🟢 → 🟡 → 🔴; show SHAP reasons, then the audit log.
5. **New:** open the Power BI report → "the API handles one transaction in real time; Power BI is how the risk team monitors fraud trends, model performance and money saved across all 200k transactions".

### Power BI interview Q&A
- **Why Power BI if you have Streamlit?** Streamlit = operational, single-transaction. Power BI = analytical, aggregated, self-service for non-technical stakeholders, with slicers, drill-through and scheduled refresh.
- **Why a star schema?** Fact table of transactions plus date and user dimensions gives fast, simple DAX and clean filtering.
- **Why filter to the test split?** To avoid reporting inflated performance on training data.
- **Which DAX is most important?** `Net Loss`, `Precision`, `Recall` with `CALCULATE` on `outcome`; plus `Cost Saved vs No Model`.
- **How would this go live?** Replace the CSV with a SQL/Delta table that the API writes to, Power BI in DirectQuery or scheduled refresh through a gateway.

## Troubleshooting (common errors)

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from repo root; use `python -m src.train` (not `python src/train.py`). |
| TensorFlow fails to install | Use Python 3.10–3.12 (not 3.13+/3.8). Create the venv with that version. |
| `FileNotFoundError: artifacts/xgb_model.pkl` | You skipped Step 4. Run `python -m src.train`. |
| Streamlit says "Cannot reach the API" | Start uvicorn first (Step 5) and keep that terminal open. |
| Port 8000 busy | `uvicorn api.app:app --port 8001` and set `FRAUD_API_URL=http://127.0.0.1:8001`. |
| API 422 | Input type/category invalid (e.g., amount text, unknown `merchant_category`). Correct behaviour. |
| Power BI shows dates as text | Power Query → select `date` → Data type = Date. |
| `git push` rejected (file >100 MB) | Remove the file from the commit, then add it to `.gitignore`. |

## Project folder map
```
data/credit_card_data.csv       raw dataset
src/{config,features,scoring,train,export_powerbi}.py
api/app.py                      FastAPI service
dashboard/streamlit_app.py      Streamlit UI
tests/test_api.py               pytest
powerbi/  POWERBI_GUIDE.md, fraud_theme.json, data/ (generated)
artifacts/ logs/ reports/figures/   generated outputs
docs/BUILD_STEPS.md             this file
```

## Optional upgrades (the "Phase 10" talking points from your PDF; not built here)
Redis for state, Kafka ingestion, Docker/Kubernetes, ADWIN drift detection, GitHub Actions + MLflow retraining. A `Dockerfile` and a GitHub Actions workflow running `pytest` are the cheapest additions if you want to show more.
