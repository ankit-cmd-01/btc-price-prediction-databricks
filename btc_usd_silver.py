# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, year, month, dayofmonth, hour, dayofweek,
    round, when, last
)
from pyspark.sql.window import Window

# ==============================
# INIT
# ==============================
spark = SparkSession.builder.getOrCreate()

BRONZE_TABLE = "btc_usd_catalog.btc_usd_bronze.btc_usd_bronze"
SILVER_TABLE = "btc_usd_catalog.btc_usd_silver.btc_usd_silver"


# ==============================
# CHECK EMPTY (SERVERLESS SAFE)
# ==============================
def is_empty(df):
    return df.limit(1).count() == 0


# ==============================
# GET LAST PROCESSED TIMESTAMP
# ==============================
def get_last_processed_time():
    try:
        return spark.sql(f"""
            SELECT MAX(datetime) as max_dt
            FROM {SILVER_TABLE}
        """).first()["max_dt"]
    except:
        return None


# ==============================
# READ INCREMENTAL (FIXED)
# ==============================
def read_bronze_incremental():
    last_ts = get_last_processed_time()

    bronze_df = spark.read.table(BRONZE_TABLE)

    if last_ts:
        # 🔥 Get last row from Silver (IMPORTANT for forward fill)
        prev_row = spark.sql(f"""
            SELECT *
            FROM {SILVER_TABLE}
            ORDER BY datetime DESC
            LIMIT 1
        """)

        # 🔥 Get new data (>= to avoid missing boundary)
        new_data = bronze_df.filter(col("datetime") >= last_ts)

        # 🔥 Combine previous + new
        df = prev_row.unionByName(new_data, allowMissingColumns=True)
    else:
        df = bronze_df

    return df


# ==============================
# TRANSFORM DATA (FIXED)
# ==============================
def transform_data(df):
    if is_empty(df):
        return df

    # 🔹 Select required columns
    df = df.select("datetime", "open", "high", "low", "close", "volume")

    # 🔹 Clean data
    df = df.dropDuplicates(["datetime"]).filter(col("close").isNotNull())

    # 🔹 Sort (VERY IMPORTANT)
    df = df.orderBy("datetime")

    # ==============================
    # 🔥 FIXED VOLUME HANDLING
    # ==============================

    # Step 1: Convert 0 → NULL
    df = df.withColumn(
        "volume",
        when(col("volume") == 0, None).otherwise(col("volume"))
    )

    # Step 2: Forward fill using full history
    window = Window.orderBy("datetime").rowsBetween(Window.unboundedPreceding, 0)

    df = df.withColumn(
        "volume",
        last("volume", ignorenulls=True).over(window)
    )

    # Step 3: Handle edge case
    df = df.fillna({"volume": 1})

    # ==============================
    # FEATURE ENGINEERING
    # ==============================
    df = df.select(
        "*",
        round(col("close") - col("open"), 4).alias("price_change"),
        round((col("close") - col("open")) / col("open") * 100, 4).alias("price_change_pct"),
        round(col("high") - col("low"), 4).alias("range"),
        round((col("open") + col("high") + col("low") + col("close")) / 4, 4).alias("avg_price"),
        year("datetime").alias("year"),
        month("datetime").alias("month"),
        dayofmonth("datetime").alias("day"),
        hour("datetime").alias("hour"),
        dayofweek("datetime").alias("day_of_week")
    )

    return df


# ==============================
# SAVE TO SILVER
# ==============================
def save_to_silver(df):
    if is_empty(df):
        print("No data to write")
        return

    (
        df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(SILVER_TABLE)
    )


# ==============================
# MAIN PIPELINE
# ==============================
def run_silver_pipeline():
    bronze_df = read_bronze_incremental()

    if is_empty(bronze_df):
        print("No new records to process")
        return

    silver_df = transform_data(bronze_df)

    save_to_silver(silver_df)

    print("✅ Silver layer updated successfully")


# ==============================
# RUN
# ==============================
run_silver_pipeline()