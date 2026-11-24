"""Streamlit UI: zero ML logic, HTTP calls only.   streamlit run dashboard/streamlit_app.py"""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Fraud Detection Console", page_icon="🛡️", layout="wide")
st.title("🛡️ Hybrid Credit Card Fraud Detection")
st.caption("XGBoost + Isolation Forest + Autoencoder + rules → 0-100 risk score → APPROVE / VERIFY / BLOCK")

try:
    cats = requests.get(f"{API}/meta", timeout=3).json()
except Exception:
    st.error(f"Cannot reach the API at {API}. Start it with:  uvicorn api.app:app --reload")
    st.stop()

PRESETS = {
    "🟢 Normal": dict(amount=1500.0, location="Mumbai", device="mobile_app"),
    "🟡 Medium risk": dict(amount=25000.0, location="Delhi", device="web"),
    "🔴 Fraud attack": dict(amount=95000.0, location="Chennai", device="pos_terminal"),
}
preset = st.radio("Demo scenario", list(PRESETS), horizontal=True)
p = PRESETS[preset]

tab_score, tab_audit = st.tabs(["Score a transaction", "Audit log"])
with tab_score:
    with st.form("tx"):
        c1, c2, c3 = st.columns(3)
        user_id = c1.number_input("User ID", 1, 100000, 42)
        amount = c2.number_input("Amount (₹)", 1.0, 10_000_000.0, p["amount"], key=f"a{preset}")
        location = c3.text_input("Location", p["location"], key=f"l{preset}")
        c4, c5, c6 = st.columns(3)
        devices = ["pos_terminal", "web", "mobile_app", "atm"]
        device = c4.selectbox("Device", devices, index=devices.index(p["device"]), key=f"d{preset}")
        card = c5.selectbox("Card", ["VISA", "MASTER", "RUPAY"])
        merchant = c6.selectbox("Merchant category", cats["merchant_categories"], index=cats["merchant_categories"].index("grocery"))
        go_btn = st.form_submit_button("Score transaction", type="primary")

    if go_btn:
        body = dict(user_id=int(user_id), amount=float(amount), card_type=card, merchant_category=merchant,
                    location=location, device_type=device)
        r = requests.post(f"{API}/score_transaction", json=body, timeout=15)
        if r.status_code != 200:
            st.error(f"API rejected the request ({r.status_code}): {r.json()}")
        else:
            res = r.json()
            color = {"APPROVE": "#2a9d8f", "VERIFY": "#f4a261", "BLOCK": "#e63946"}[res["decision"]]
            left, right = st.columns([1, 1])
            with left:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=res["risk_score"], title={"text": "Hybrid risk score"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                           "steps": [{"range": [0, 30], "color": "#e8f5f3"}, {"range": [30, 70], "color": "#fdf0e2"}, {"range": [70, 100], "color": "#fbe3e5"}]}))
                fig.update_layout(height=280, margin=dict(t=40, b=0))
                st.plotly_chart(fig, width="stretch")
            with right:
                {"APPROVE": st.success, "VERIFY": st.warning, "BLOCK": st.error}[res["decision"]](f"Decision: **{res['decision']}**")
                st.markdown("**Why? (SHAP explanation)**")
                for rs in res["reasons"]:
                    st.write(f"• {rs['text']}  `(impact {rs['impact']:+.2f})`")
                st.caption(f"Latency {res['latency_ms']} ms · request {res['request_id'][:8]} · known user: {res['known_user']}")
            comp = pd.DataFrame({"model": list(res["components"]), "score (0-1)": list(res["components"].values())})
            st.bar_chart(comp, x="model", y="score (0-1)", height=200)

with tab_audit:
    if st.button("Refresh"):
        pass
    log = requests.get(f"{API}/audit_log?limit=100", timeout=5).json()
    if log:
        df = pd.json_normalize(log)
        st.dataframe(df[["logged_at", "input.user_id", "input.amount", "input.location", "risk_score", "decision", "reasons"]], width="stretch")
    else:
        st.info("No transactions scored yet.")
