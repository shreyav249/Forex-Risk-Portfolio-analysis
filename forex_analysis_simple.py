# Forex Risk & Portfolio Analysis - SIMPLE VERSION
# USD, EUR, GBP, JPY against the Indian rupee (RBI reference rates)
# Run from the project folder:  python src/forex_analysis_simple.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RF = 0.06      # assumed risk-free rate (6% a year)
DAYS = 252     # trading days in a year

# 1. LOAD AND CLEAN THE DATA -------------------------------------------
df = pd.read_csv("data/rbi_forex_reference_rates_wide.csv", parse_dates=["date"], index_col="date")
df = df[["USD", "EUR", "GBP", "JPY"]].dropna()           # keep 4 currencies, drop missing rows
df.columns = ["USDINR", "EURINR", "GBPINR", "JPYINR"]    # INR per 1 unit (JPY is per 100 yen)
df = df.loc[df.index[-1] - pd.DateOffset(years=5):]      # use the last 5 years
print("Rows, columns:", df.shape)
print(df.describe().round(2))

# 2. RETURNS AND TIME-SERIES CHECK -------------------------------------
ret = np.log(df / df.shift(1)).dropna()                  # daily log returns
print("\nLag-1 autocorrelation of prices :", df.apply(lambda s: s.autocorr(1)).round(3).to_dict())
print("Lag-1 autocorrelation of returns:", ret.apply(lambda s: s.autocorr(1)).round(3).to_dict())
print("Skewness:", ret.skew().round(2).to_dict())
print("Excess kurtosis:", ret.kurt().round(2).to_dict())   # above 0 = fat tails

# 3. MOVING AVERAGES (trend) -------------------------------------------
sma50 = df.rolling(50).mean()
sma200 = df.rolling(200).mean()
up = sma50.iloc[-1] > sma200.iloc[-1]                     # True = short average above long average
print("\nTrend today (SMA50 vs SMA200):\n", up.map({True: "Uptrend (rupee weaker)", False: "Downtrend (rupee stronger)"}).to_string())

# 4. RISK METRICS -------------------------------------------------------
q5 = ret.quantile(0.05)                                   # 5th percentile of daily returns
risk = pd.DataFrame({
    "return_%": ret.mean() * DAYS * 100,                  # annual return
    "volatility_%": ret.std() * np.sqrt(DAYS) * 100,      # annual volatility
    "VaR95_%": -q5 * 100,                                 # worst 1-day loss on 95% of days
    "CVaR95_%": -ret[ret <= q5].mean() * 100,             # average loss on the worst 5% of days
    "max_drawdown_%": (df / df.cummax() - 1).min() * 100})  # biggest peak-to-trough fall
risk["sharpe"] = (risk["return_%"] / 100 - RF) / (risk["volatility_%"] / 100)
print("\n", risk.round(2))
print("\nCorrelation of returns:\n", ret.corr().round(2))

# 5. MARKOWITZ PORTFOLIO (long-only, weights add up to 100%) ------------
mu, cov = ret.mean() * DAYS, ret.cov() * DAYS             # yearly mean returns and covariance
vol = lambda w: np.sqrt(w @ cov.values @ w)               # portfolio volatility
neg_sharpe = lambda w: -(w @ mu.values - RF) / vol(w)     # minimise this to maximise Sharpe
start = np.ones(4) / 4
rules = dict(bounds=[(0, 1)] * 4, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
min_var = minimize(lambda w: vol(w) ** 2, start, **rules).x
max_sharpe = minimize(neg_sharpe, start, **rules).x

portfolios = {"Equal weight": start, "Min variance": min_var, "Max Sharpe": max_sharpe}
print("\nWeights (%):\n", (pd.DataFrame(portfolios, index=df.columns).T * 100).round(1))
for name, w in portfolios.items():
    print(f"{name}: return {w @ mu.values * 100:.2f}%, volatility {vol(w) * 100:.2f}%")

# 6. TEXT SENTIMENT (simple word list) ---------------------------------
good = ["gains", "strengthens", "supports", "inflows", "rallies", "steady", "improved", "cools", "easing", "lifts", "holds"]
bad = ["weakens", "falls", "slides", "outflows", "deficit", "pressure", "pressures", "selloff", "volatile", "low", "slump", "fall"]
heads = pd.read_csv("data/sample_headlines.csv")
def score(text):
    words = text.lower().replace(",", " ").split()
    return sum(w in good for w in words) - sum(w in bad for w in words)
heads["score"] = heads["headline"].apply(score)
print("\nAverage sentiment (positive = good for the rupee):\n", heads.groupby("currency")["score"].mean().round(2))
