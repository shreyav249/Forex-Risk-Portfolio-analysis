# Forex-Risk-Portfolio-analysis

Beginner-level Python capstone project analysing **USDINR, EURINR, GBPINR and JPYINR** using RBI reference rates (1999–2026): time-series analysis, moving averages, risk metrics (volatility, VaR, CVaR, drawdown), Markowitz portfolio optimisation and simple financial text analytics, plus a live-rate monitoring website.

## Project structure
```
data/        RBI reference-rate CSV + sample_headlines.csv (illustrative headlines)
src/         forex_analysis_simple.py  (short, beginner-friendly version, ~65 lines)
             forex_analysis.py         (full version: charts, tables saved to outputs/)
notebooks/   Forex_Risk_Portfolio_Analysis.ipynb  (same code, outputs included)
outputs/     result tables (.csv) and figures/ (.png)
website/     index.html  (live dashboard)
report/      Capstone_Report.docx  (7-10 page report)
```

## How to run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/forex_analysis_simple.py                 # short version, prints all results
python src/forex_analysis.py                        # full version: regenerates all tables and figures
jupyter notebook notebooks/Forex_Risk_Portfolio_Analysis.ipynb
```

## Live website
Open `website/index.html` in a browser (needs internet). It fetches current ECB reference rates from the free Frankfurter API, shows rate cards, a chart with 1M/3M/6M/1Y windows and live risk metrics (volatility, VaR, CVaR, drawdown). If the API cannot be reached it shows an embedded RBI snapshot instead.
To publish: push to GitHub, then *Settings → Pages → deploy from `/website`* (or copy `index.html` to the repo root).

## Method summary
| Step | Method |
|---|---|
| Returns | Daily log returns, last 5 years |
| Time series | Distribution tests (Jarque-Bera), autocorrelation, long-run trend |
| Moving average | SMA20 / SMA50 / SMA200, crossover signal |
| Risk | Volatility, Sharpe (6% risk-free), VaR 95/99%, CVaR, max drawdown, correlation |
| Markowitz | 5,000 random portfolios + SciPy optimiser (min variance, max Sharpe) |
| Text analytics | Lexicon-based sentiment on headlines |

## Notes
- JPY is quoted per 100 yen (RBI convention).
- `sample_headlines.csv` is illustrative; replace it with real news for genuine conclusions.
- Educational project, not investment advice.
