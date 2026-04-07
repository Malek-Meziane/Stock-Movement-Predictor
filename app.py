import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_loader import load_stock_data
from src.features import engineer_features
from src.preprocessing import split_and_scale, FEATURE_COLS
from src.model import (train_logistic_regression,
                       train_random_forest,
                       train_xgboost)

st.set_page_config(
    page_title="Stock Movement Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("Stock Movement Predictor")
st.caption("Predicts significant price moves using technical indicators")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    ticker = st.text_input(
        "Stock ticker",
        value="AAPL",
        help="e.g. AAPL, TSLA, MSFT, MC.PA"
    )

    start_date = st.date_input(
        "Start date",
        value=pd.to_datetime("2018-01-01")
    )

    end_date = st.date_input(
        "End date",
        value=pd.to_datetime("2024-01-01")
    )

    threshold = st.slider(
        "Prediction threshold",
        min_value=0.3,
        max_value=0.7,
        value=0.5,
        step=0.05,
        help="Higher = only flag high confidence predictions"
    )

    run_button = st.button("Run Analysis", type="primary")

# ── Stop if button not clicked ────────────────────────────
if not run_button:
    st.info("Configure settings in the sidebar and click Run Analysis")
    st.stop()

# ── Cached pipeline ───────────────────────────────────────
@st.cache_resource
@st.cache_resource(show_spinner=False)
def run_pipeline(ticker, start, end):
    df = load_stock_data(ticker, str(start), str(end))
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    lr  = train_logistic_regression(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    return df, X_train, X_test, y_train, y_test, scaler, lr, rf, xgb, explainer, shap_values

# ── Run pipeline ──────────────────────────────────────────
with st.spinner(f"Loading {ticker} data and training models..."):
    df, X_train, X_test, y_train, y_test, scaler, lr, rf, xgb, explainer, shap_values = run_pipeline(
        ticker, start_date, end_date
    )

# ── Dataset overview ──────────────────────────────────────
st.subheader("Dataset overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total trading days", f"{len(df):,}")
col2.metric("Training days", f"{int(len(df)*0.8):,}")
col3.metric("Test days", f"{int(len(df)*0.2):,}")
col4.metric("Significant move threshold", "±1%")

st.divider()

# ── Model comparison ──────────────────────────────────────
st.subheader("Model performance comparison")

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
    })

st.dataframe(pd.DataFrame(results), use_container_width=True)

st.divider()

# ── Price chart ───────────────────────────────────────────
st.subheader(f"{ticker} price history")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df["Close"], linewidth=1, color="#1F4E79")
ax.set_ylabel("Price (USD)")
ax.set_xlabel("")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
st.pyplot(fig)
plt.close()

st.divider()

# ── SHAP global importance ────────────────────────────────
st.subheader("What drives the predictions?")

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=FEATURE_COLS,
    show=False
)
st.pyplot(fig)
plt.close()

st.divider()

# ── Tomorrow's prediction ─────────────────────────────────
st.subheader(f"Tomorrow's prediction for {ticker}")

latest_features = df[FEATURE_COLS].iloc[-1:].values
latest_scaled   = scaler.transform(latest_features)

prob_up    = xgb.predict_proba(latest_scaled)[0][1]
prediction = "UP" if prob_up >= threshold else "DOWN"
confidence = prob_up if prob_up >= threshold else 1 - prob_up

col1, col2, col3 = st.columns(3)
col1.metric("Prediction", prediction)
col2.metric("Confidence", f"{confidence:.1%}")
col3.metric("Model", "XGBoost")

if prediction == "UP":
    st.success("Model predicts a significant UP move tomorrow")
else:
    st.error("Model predicts a significant DOWN move tomorrow")

st.divider()

# ── Try another stock ─────────────────────────────────────
st.subheader("Try another stock")
st.write("Change the ticker in the sidebar and click Run Analysis again.")
st.write("Works with any stock: TSLA, MSFT, GOOGL, MC.PA, TTE.PA...")

st.caption("⚠️ This is not financial advice. Model accuracy is ~55% on historical test data.")