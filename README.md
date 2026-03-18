# BTC Price Prediction — Databricks Medallion Architecture

A fully automated Bitcoin price prediction pipeline built on Databricks using the Medallion Architecture (Bronze → Silver → Gold). The pipeline downloads real BTC price data, engineers features, trains a Linear Regression model, and appends hourly predictions to a Delta table — all orchestrated via Databricks Jobs.

---

## Project Overview

| Layer  | Table                                        | Description                                          |
|--------|----------------------------------------------|------------------------------------------------------|
| Bronze | btc_catalog.default.btc_bronze               | Raw BTC-USD daily price data from Yahoo Finance      |
| Silver | btc_catalog.default.btc_silver_features      | Engineered features: RSI14, MA7, Return_1d, Label    |
| Gold   | btc_catalog.default.btc_gold_predictions     | Model predictions appended every hour                |

---

## Architecture

```
Yahoo Finance (yfinance)
        ↓
  Bronze Notebook
  (Raw ingestion → Delta table)
        ↓
  Silver Notebook
  (Feature engineering → RSI, MA, Returns)
        ↓
  Gold Notebook
  (Linear Regression model → Predictions)
        ↓
  Databricks Job (Hourly schedule)
```

---

## Notebooks

| File                              | Description                                                        |
|-----------------------------------|--------------------------------------------------------------------|
| 01_bronze_layer.ipynb             | Downloads raw BTC-USD data from Yahoo Finance and saves to Delta   |
| 02_silver_layer.ipynb             | Engineers features like RSI14, MA7, Return_1d and creates labels   |
| 03_gold_layer.ipynb               | Trains Linear Regression model and stores predictions              |
| 04_gold_linear_regression.ipynb   | Extended model with evaluation metrics                             |

---

## Features Engineered

- **RSI14** — Relative Strength Index over 14 days
- **MA7** — 7-day Moving Average
- **Return_1d** — Daily percentage return
- **Label** — Binary up/down movement for next day

---

## Dashboard

The project includes a live Databricks Dashboard with 5 visualizations:

- Actual BTC Close Price Over Time
- Actual vs Predicted BTC Price (line chart)
- Prediction Error Over Time (bar chart)
- Up vs Down Predictions (donut chart)
- Recent Predictions Table

---

## Tech Stack

- **Databricks** — Unified data platform
- **Apache Spark** — Distributed data processing
- **Delta Lake** — ACID-compliant storage layer
- **MLlib / Scikit-learn** — Machine learning
- **yfinance** — Bitcoin price data source
- **Python** — Primary language

---

## How to Run

1. Clone this repository
2. Import notebooks into your Databricks workspace
3. Set up a Databricks catalog named `btc_catalog`
4. Run notebooks in order: Bronze → Silver → Gold
5. Schedule using Databricks Jobs for hourly predictions

---

## Author

**Ankit Shinde**
- GitHub: [@ankit-cmd-01](https://github.com/ankit-cmd-01)

---

## License

This project is open source and available under the [MIT License](LICENSE).
