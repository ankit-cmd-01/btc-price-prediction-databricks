# Databricks notebook source
from pyspark.sql.functions import col, lag, avg, stddev, round, try_divide
from pyspark.sql.window import Window

silver_table = "btc_usd_catalog.btc_usd_silver.btc_usd_silver"
gold_table = "btc_usd_catalog.btc_usd_gold.btc_usd_ml_gold"

def get_last_processed_time():
    try:
        last_time = spark.sql(f"""
            SELECT MAX(datetime) as max_dt
            FROM {gold_table}
        """).collect()[0]["max_dt"]
        return last_time
    except:
        return None

def read_silver_incremental():
    last_ts = get_last_processed_time()

    if last_ts:
        df = spark.sql(f"""
            SELECT *
            FROM {silver_table}
            WHERE datetime > '{last_ts}'
        """)
    else:
        df = spark.table(silver_table)

    return df

def create_ml_features(df):

    window_spec = Window.orderBy("datetime")
    window_6 = Window.orderBy("datetime").rowsBetween(-5, 0)
    window_12 = Window.orderBy("datetime").rowsBetween(-11, 0)
    window_24 = Window.orderBy("datetime").rowsBetween(-23, 0)

    df = df.withColumn("lag_1", lag("close",1).over(window_spec))
    df = df.withColumn("lag_3", lag("close",3).over(window_spec))
    df = df.withColumn("lag_6", lag("close",6).over(window_spec))

    df = df.withColumn(
        "return_1h",
        round(try_divide(col("close") - col("lag_1"), col("lag_1")), 6)
    )

    df = df.withColumn(
        "price_range",
        round(col("high") - col("low"), 6)
    )

    df = df.withColumn(
        "volume_change",
        round(
            try_divide(
                col("volume") - lag("volume",1).over(window_spec),
                lag("volume",1).over(window_spec)
            ), 6
        )
    )

    df = df.withColumn("sma_6", avg("close").over(window_6))
    df = df.withColumn("sma_12", avg("close").over(window_12))
    df = df.withColumn("sma_24", avg("close").over(window_24))

    df = df.withColumn("rolling_std_12", stddev("close").over(window_12))

    df = df.withColumn(
        "target_price",
        lag("close",-1).over(window_spec)
    )

    df = df.dropna()

    return df

def save_to_gold(df):
    df.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable(gold_table)

def run_gold_pipeline():

    silver_df = read_silver_incremental()

    if silver_df.head(1) == []:
        print("No new records to process")
        return

    gold_df = create_ml_features(silver_df)

    save_to_gold(gold_df)

    print("Gold layer updated successfully")

run_gold_pipeline()