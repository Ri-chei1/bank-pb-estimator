import streamlit as st
import pandas as pd
import numpy as np

with st.expander("Data sources"):
    st.write(
        "FY2024 fundamentals and P/B ratios were collected from StockAnalysis (StockAnalysis.com). "
    )

st.set_page_config(page_title="Bank P/B Estimator", layout="wide")
st.title("Bank P/B Estimator (Linear Regression)")

# ---- Fixed metadata (NO SIDEBAR) ----
FUNDAMENTALS_YEAR = 2024
PB_ASOF_DATE = "2024"   # change if you want e.g. "Jan 2026"
DATA_PATH = "bank_pbs.csv"

st.caption(
    f"Fundamentals: FY{FUNDAMENTALS_YEAR} (Net Income, Assets, Equity). "
    f"Target: P/B ratios collected as-of {PB_ASOF_DATE}."
)

# ----------------------------
# Closed-form Linear Regression (no sklearn model)
# ----------------------------
class LinearRegressionClosed:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1, 1)

        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        beta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)  # (p+1, 1)

        self.intercept_ = float(beta[0, 0])
        self.coef_ = beta[1:, 0]  # 1D array

    def predict(self, X):
        X = np.asarray(X, float)
        return X @ self.coef_ + self.intercept_


@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # convert numeric columns that may contain commas
    comma_cols = ["Net Income", "Assets", "Shareholder's Equity", "PB", "ROE", "Log_Assets", "Log_pb", "eq_ratio"]
    for c in comma_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

    # If ROE is missing/mostly empty, recompute it (ROE = Net Income / Equity)
    if "ROE" not in df.columns or df["ROE"].isna().mean() > 0.2:
        if "Net Income" in df.columns and "Shareholder's Equity" in df.columns:
            df["ROE"] = df["Net Income"] / df["Shareholder's Equity"]

    # If Log_Assets / Log_pb missing, compute them
    if "Log_Assets" not in df.columns and "Assets" in df.columns:
        df["Log_Assets"] = np.log(df["Assets"])

    if "Log_pb" not in df.columns and "PB" in df.columns:
        df["Log_pb"] = np.log(df["PB"])

    # Equity ratio feature
    if "eq_ratio" not in df.columns and "Assets" in df.columns and "Shareholder's Equity" in df.columns:
        df["eq_ratio"] = df["Shareholder's Equity"] / df["Assets"]

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Ticker", "ROE", "Log_Assets", "Log_pb", "eq_ratio"])
    return df


# ----------------------------
# Load dataset (fixed path, no sidebar)
# ----------------------------
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load '{DATA_PATH}'. Error: {e}")
    st.stop()

# ----------------------------
# Train model
# ----------------------------
X = df[["ROE", "Log_Assets", "eq_ratio"]].to_numpy()
y = df["Log_pb"].to_numpy()

reg = LinearRegressionClosed()
reg.fit(X, y)

st.subheader("Model")
st.write(
    f"Target: **log(P/B)** (as-of {PB_ASOF_DATE})  |  "
    f"Features: **ROE (FY{FUNDAMENTALS_YEAR})**, **log(Assets) (FY{FUNDAMENTALS_YEAR})**, "
    f"**Equity/Assets (FY{FUNDAMENTALS_YEAR})**"
)
st.code(
    f"log(P/B) = {reg.intercept_:.4f} + {reg.coef_[0]:.4f}*ROE + {reg.coef_[1]:.4f}*log(Assets) + {reg.coef_[2]:.4f}*(Equity/Assets)"
)

# ----------------------------
# Evaluate (simple in-sample)
# ----------------------------
y_pred = reg.predict(X)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
mse = np.mean((y - y_pred) ** 2)

col1, col2 = st.columns(2)
col1.metric("In-sample R²", f"{r2:.3f}")
col2.metric("In-sample MSE", f"{mse:.4f}")

# ----------------------------
# Prediction tools
# ----------------------------
st.subheader("Estimate P/B")

tab1, tab2 = st.tabs(["Pick a ticker", "Manual input"])

with tab1:
    ticker = st.selectbox("Choose a bank", sorted(df["Ticker"].unique()))
    row = df[df["Ticker"] == ticker].iloc[0]

    x_row = np.array([[row["ROE"], row["Log_Assets"], row["eq_ratio"]]])
    pred_log_pb = reg.predict(x_row)[0]
    pred_pb = float(np.exp(pred_log_pb))

    actual_pb = float(np.exp(row["Log_pb"]))
    resid = float(row["Log_pb"] - pred_log_pb)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Actual P/B", f"{actual_pb:.2f}")
    c2.metric("Predicted P/B", f"{pred_pb:.2f}")

    # log residual
    c3.metric("Residual (log)", f"{resid:.3f}")

    # convert to % gap: exp(resid) - 1
    gap_pct = (np.exp(resid) - 1) * 100

    # tolerance band (adjust if you want)
    tol = 0.05  # ~5% band around fair value

    if abs(gap_pct) <= tol * 100:
        c4.metric("Model Thinks", "Fairly Valued")
    elif gap_pct > 0:
        c4.metric("Model Thinks", f"Overvalued")
    else:
        c4.metric("Model Thinks", f"Undervalued")

    st.write("Inputs used:")
    st.write(
        {
            f"ROE (FY{FUNDAMENTALS_YEAR})": float(row["ROE"]),
            f"Assets (FY{FUNDAMENTALS_YEAR})": float(np.exp(row["Log_Assets"])),
            f"Equity/Assets (FY{FUNDAMENTALS_YEAR})": float(row["eq_ratio"]),
            "P/B as-of": PB_ASOF_DATE,
        }
    )

with tab2:
    roe = st.number_input(
        f"ROE (FY{FUNDAMENTALS_YEAR}, decimal; 0.12 = 12%)",
        value=0.10, step=0.01, format="%.4f"
    )
    assets = st.number_input(
        f"Total Assets (FY{FUNDAMENTALS_YEAR}, enter unit in millions)",
        value=1_000_000.0, step=50_000.0
    )
    eq_ratio = st.number_input(
        f"Equity/Assets (FY{FUNDAMENTALS_YEAR}, e.g., 0.08)",
        value=0.08, step=0.01, format="%.4f"
    )

    if assets <= 0:
        st.warning("Assets must be > 0.")
    else:
        log_assets = float(np.log(assets))
        x_new = np.array([[roe, log_assets, eq_ratio]])
        pred_log_pb = reg.predict(x_new)[0]
        pred_pb = float(np.exp(pred_log_pb))
        st.success(f"Estimated P/B (as-of {PB_ASOF_DATE}) ≈ **{pred_pb:.2f}**")

# ----------------------------
# Under/Over-valuation table
# ----------------------------
st.subheader(f"Under/Overvalued US Bank Stocks (FY{FUNDAMENTALS_YEAR})")

out = df[["Ticker", "Log_pb", "ROE", "Log_Assets", "eq_ratio"]].copy()
out["pred_log_pb"] = y_pred
out["resid"] = out["Log_pb"] - out["pred_log_pb"]
out["actual_pb"] = np.exp(out["Log_pb"])
out["pred_pb"] = np.exp(out["pred_log_pb"])

top_n = st.slider("How many to show?", 5, 30, 10)

c1, c2 = st.columns(2)
with c1:
    st.write("Most **Overvalued** Stocks Based on Model")
    above = (
        out.sort_values("resid", ascending=False)
           .head(top_n)[["Ticker", "actual_pb", "pred_pb", "resid"]]
           .reset_index(drop=True)
    )
    above.index = above.index + 1
    st.dataframe(above, use_container_width=True)

with c2:
    st.write("Most **Undervalued** Stocks Based on Model")
    below = (
        out.sort_values("resid", ascending=True)
           .head(top_n)[["Ticker", "actual_pb", "pred_pb", "resid"]]
           .reset_index(drop=True)
    )
    below.index = below.index + 1
    st.dataframe(below, use_container_width=True)
