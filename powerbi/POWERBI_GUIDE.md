# Power BI Guide: Fraud Monitoring Dashboard

Power BI Desktop (Windows) is needed to author the `.pbix`. A `.pbix` cannot be generated outside Power BI, so this guide
tells you exactly what to click and paste. Budget about 60–90 minutes.

Prerequisite: run `python -m src.export_powerbi` first so `powerbi/data/*.csv` exists.

## 1. Load the data (Power Query)

1. Power BI Desktop → **Home → Get data → Text/CSV** → pick `fact_transactions.csv` → **Transform Data**.
2. Repeat for `dim_date.csv`, `dim_user.csv`, `model_cost_comparison.csv`, `shap_global_importance.csv`.
3. In Power Query set data types:
   - `fact_transactions`: `timestamp` → **Date/Time**; `amount, risk_score, xgb_prob, iso_score, ae_score, amount_zscore, rule_score, loss_inr` → **Decimal**; `transaction_id, user_id, date_key, hour, is_fraud, device_changed, location_changed` → **Whole number**; others → Text.
   - `dim_date`: `date` → **Date**.
4. Add a column in `fact_transactions` → **Custom column** `risk_band` :
   `if [risk_score] >= 70 then "High" else if [risk_score] >= 30 then "Medium" else "Low"`
5. **Close & Apply**.

> Tip: create a Power Query parameter `DataFolder` (Home → Manage Parameters) so the path is not hard-coded, which makes the `.pbix` portable.

## 2. Model (Model view)

- `dim_date[date_key]` **1 → \*** `fact_transactions[date_key]`
- `dim_user[user_id]` **1 → \*** `fact_transactions[user_id]`
- Right-click `dim_date` → **Mark as date table** → `date`.
- Sort `dim_date[month]` by `month_no`, and `dim_date[weekday]` by `weekday_no`.
- Leave `model_cost_comparison` and `shap_global_importance` unrelated (they are small reference tables).

## 3. DAX measures

Create a table `_Measures` (Home → Enter data → empty, name it `_Measures`) and add these measures:

```DAX
Total Transactions = COUNTROWS(fact_transactions)
Total Amount = SUM(fact_transactions[amount])
Fraud Transactions = SUM(fact_transactions[is_fraud])
Fraud Rate % = DIVIDE([Fraud Transactions], [Total Transactions])
Fraud Amount = CALCULATE(SUM(fact_transactions[amount]), fact_transactions[is_fraud] = 1)
Avg Risk Score = AVERAGE(fact_transactions[risk_score])

Approved Txns = CALCULATE([Total Transactions], fact_transactions[decision] = "APPROVE")
Verify Txns   = CALCULATE([Total Transactions], fact_transactions[decision] = "VERIFY")
Blocked Txns  = CALCULATE([Total Transactions], fact_transactions[decision] = "BLOCK")
Alert Rate %  = DIVIDE([Verify Txns] + [Blocked Txns], [Total Transactions])

TP = CALCULATE([Total Transactions], fact_transactions[outcome] = "TP")
FP = CALCULATE([Total Transactions], fact_transactions[outcome] = "FP")
FN = CALCULATE([Total Transactions], fact_transactions[outcome] = "FN")
TN = CALCULATE([Total Transactions], fact_transactions[outcome] = "TN")

Precision = DIVIDE([TP], [TP] + [FP])
Recall = DIVIDE([TP], [TP] + [FN])
F1 Score = DIVIDE(2 * [Precision] * [Recall], [Precision] + [Recall])
False Positive Rate = DIVIDE([FP], [FP] + [TN])

Net Loss (INR) = SUM(fact_transactions[loss_inr])
Loss Without Model (INR) = [Fraud Transactions] * 5000
Cost Saved (INR) = [Loss Without Model (INR)] - [Net Loss (INR)]
Fraud Caught Amount = CALCULATE([Fraud Amount], fact_transactions[outcome] = "TP")
Fraud Rate vs Prior Month =
VAR cur = [Fraud Rate %]
VAR prev = CALCULATE([Fraud Rate %], DATEADD(dim_date[date], -1, MONTH))
RETURN DIVIDE(cur - prev, prev)
```
(The ₹5,000 / ₹200 values come from your cost matrix. Better still, add two What-If parameters so viewers can change them live: Modeling → New parameter → `CostFN` 5000 and `CostFP` 200, then `Net Loss` = `[FN]*[CostFN Value] + [FP]*[CostFP Value]`.)

Important: precision, recall and cost are **only meaningful on unseen data**. Put a slicer on `fact_transactions[model_split]` and default it to `test` on the Model Performance page.

## 4. Report pages

Apply the theme first: **View → Themes → Browse for themes → `powerbi/fraud_theme.json`**.
Add a shared slicer panel (sync across pages): `dim_date[date]` (Between), `card_type`, `device_type`.

### Page 1 – Executive Overview
- **Cards:** Total Transactions, Fraud Transactions, Fraud Rate %, Fraud Amount, Net Loss (INR), Cost Saved (INR)
- **Line chart:** X = `dim_date[date]`, Y = `Fraud Transactions` (+ secondary `Avg Risk Score`)
- **Donut:** `decision` by `Total Transactions` (APPROVE / VERIFY / BLOCK). Because APPROVE dwarfs the rest, add a bar chart on a log axis or filter APPROVE out.
- **Bar:** `merchant_category` by `Fraud Amount` (Top N = 10)

### Page 2 – Fraud Patterns
- **Matrix heatmap:** Rows = `dim_date[weekday]`, Columns = `hour`, Values = `Fraud Rate %`, with conditional formatting (background colour scale).
- **Bar:** `device_type` by `Fraud Rate %`; **Column:** `card_type` by `Fraud Transactions`
- **Map (filled/bubble):** `location` by `Fraud Transactions` (Indian city names geocode automatically; set Data category = City if needed)
- **Scatter:** X = `amount`, Y = `risk_score`, legend = `is_fraud` (shows the separation the model learned)

### Page 3 – Model Performance & Cost (slicer `model_split` = test)
- **Cards:** Precision, Recall, F1 Score, False Positive Rate
- **Matrix confusion matrix:** Values = `TP, FP, FN, TN`
- **Clustered column:** from `model_cost_comparison`: `model` by `total_cost_inr` (Rules only vs XGBoost only vs Hybrid)
- **Bar:** `shap_global_importance`: `description` by `mean_abs_shap` (global explainability)
- **Histogram-style column:** `risk_band` counts (log scale)

### Page 4 – Investigation (analyst view)
- **Table:** `transaction_id, user_id, timestamp, amount, location, device_type, risk_score, decision, top_risk_driver`, filtered to `decision` ≠ APPROVE, sorted by `risk_score` desc, with data bars on `risk_score`.
- **Drill-through** page on `user_id`: user's amount timeline, decision history, `dim_user[avg_amount]` as a reference line. (Right-click a user in the table → Drill through.)
- **Tooltip page** showing `top_risk_driver` and the three model scores.

## 5. Finish
- Rename pages, add titles, align objects, and add a text box: "Source: synthetic data, 200k transactions, hybrid ML scoring (see README)."
- Save as `powerbi/Fraud_Detection_Dashboard.pbix`.
- Export screenshots into `docs/images/`.
- Optional: **Publish** to Power BI Service (needs a work/school account) and note the link in the README.

## Numbers you should be able to reproduce (test split)
Precision ≈ 29.6%, Recall = 100%, TP = 48, FP = 114, FN = 0, hybrid cost ₹22,800. If Power BI shows other values, check the `model_split = test` slicer.
