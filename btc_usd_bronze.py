# Databricks notebook source
# MAGIC %pip install yfinance

# COMMAND ----------

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ==============================
# GET LAST TIMESTAMP (UTC SAFE)
# ==============================
def get_last_timestamp():
    try:
        last_date = spark.sql("""
            SELECT MAX(datetime) as max_date
            FROM btc_usd_catalog.btc_usd_raw.btc_usd_raw
        """).collect()[0]["max_date"]

        return last_date

    except:
        return None


# ==============================
# FETCH DATA (FIXED)
# ==============================
def fetch_data(ticker="BTC-USD", interval="1h"):

    last_timestamp = get_last_timestamp()

    # 🔥 FIX 1: Add buffer (1 hour back)
    if last_timestamp:
        start = pd.to_datetime(last_timestamp) - timedelta(hours=2)
    else:
        start = datetime.utcnow() - timedelta(days=720)

    # 🔥 FIX 2: Use current UTC time (NOT date.today)
    end = datetime.utcnow()

    print("Fetching from:", start)
    print("Fetching till:", end)

    raw_data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval
    )

    if raw_data.empty:
        print("Download Failed [X]")
        return None

    raw_data.reset_index(inplace=True)

    # Flatten multi-index columns
    raw_data.columns = raw_data.columns.get_level_values(0)

    # Lowercase
    raw_data.columns = [col.lower() for col in raw_data.columns]

    return raw_data


# ==============================
# PANDAS → SPARK
# ==============================
def pandas_to_spark(raw_data):
    spark_df = spark.createDataFrame(raw_data)
    return spark_df


# ==============================
# SAVE DATA
# ==============================
def save_data():

    raw_data = fetch_data()

    if raw_data is None:
        print("Something went wrong [X]")
        return

    spark_data = pandas_to_spark(raw_data)

    # 🔥 OPTIONAL: Remove duplicates before writing
    spark_data = spark_data.dropDuplicates(["datetime"])

    spark_data.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable("btc_usd_catalog.btc_usd_raw.btc_usd_raw")

    print("Data saved successfully [✓]")


# ==============================
# RUN
# ==============================
save_data()