# Stock Movement Predictor

A machine learning project that predicts significant Apple stock 
price movements using technical indicators, with SHAP explainability.

## Demo
![Demo](demo.gif)

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 0.5442 | 0.5432 |
| Random Forest | 0.5034 | 0.4644 |
| XGBoost | 0.5510 | 0.4391 |

> Models predict significant moves (>±1%) rather than daily direction.
> ~55% accuracy is consistent with the Efficient Market Hypothesis —
> technical indicators alone provide a slight but real edge.

## How it works

1. Downloads real stock data via yfinance (any ticker)
2. Engineers 23 technical indicators (RSI, MACD, Bollinger Bands, 
   Moving Averages, Volume)
3. Trains 3 models: Logistic Regression → Random Forest → XGBoost
4. Evaluates with walk-forward chronological split (no data leakage)
5. Explains predictions with SHAP feature importance
6. Predicts tomorrow's movement with confidence score

## Key technical choices

- **Chronological train/test split** — never trains on future data,
  preventing data leakage common in naive implementations
- **±1% move threshold** — filters noise, focuses on tradeable signals
- **SHAP explainability** — reveals which indicators actually matter
- **Adjustable threshold slider** — tune precision/recall tradeoff

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/stock-predictor.git
cd stock-predictor
python -m venv .venv
.venv\Scripts\activate.bat   # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
stock_predictor/
├── src/
│   ├── data_loader.py      ← yfinance data download
│   ├── features.py         ← 23 technical indicators
│   ├── preprocessing.py    ← chronological split + scaling
│   ├── model.py            ← Logistic Regression, RF, XGBoost
│   └── explainability.py   ← SHAP explanations
├── app.py                  ← Streamlit interface
└── requirements.txt

## What I'd add next

- Sentiment analysis from news headlines as additional features
- Walk-forward cross validation for more robust evaluation
- Portfolio backtesting to simulate real trading performance
- Support for multiple stocks simultaneously