# Bank P/B Estimator (Linear Regression)

This Streamlit app estimates a bank’s Price-to-Book (P/B) ratio using a simple cross-sectional linear regression model.

## Data

* Fundamentals: FY2024 (ROE, Assets, Equity/Assets)
* Target: P/B ratio captured as-of the date used in my dataset

## Model

Target: `log(P/B)`
Features: `ROE`, `log(Assets)`, `Equity/Assets`

The app also shows “under/overvalued” names using the log residual:
`residual = log(P/B)_actual − log(P/B)_predicted`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

* Financial statement items (FY2024) and valuation multiples (P/B) were collected from StockAnalysis (StockAnalysis.com), as-of 2026/01/17
* Fields used: Net Income, Total Assets, Shareholders’ Equity, and Price-to-Book (P/B).
