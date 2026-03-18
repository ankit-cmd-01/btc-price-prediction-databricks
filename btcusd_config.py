# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS btc_usd_catalog;
# MAGIC CREATE SCHEMA IF NOT EXISTS btc_usd_catalog.btc_usd_raw;
# MAGIC CREATE SCHEMA IF NOT EXISTS btc_usd_catalog.btc_usd_bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS btc_usd_catalog.btc_usd_silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS btc_usd_catalog.btc_usd_gold;

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG btc_usd_catalog;
# MAGIC USE SCHEMA btc_usd_gold;
# MAGIC
# MAGIC CREATE VOLUME mlflow_volume;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE btc_usd_catalog.btc_usd_gold.btc_usd_dashboard AS
# MAGIC SELECT 
# MAGIC     datetime,
# MAGIC     close AS actual_close,
# MAGIC     target_price AS predicted_close
# MAGIC FROM btc_usd_catalog.btc_usd_gold.btc_usd_ml_gold
# MAGIC ORDER BY datetime;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM btc_usd_catalog.btc_usd_gold.btc_usd_dashboard
# MAGIC ORDER BY datetime DESC
# MAGIC LIMIT 10;

# COMMAND ----------

df = spark.sql("""
SELECT 
    from_utc_timestamp(datetime, 'Asia/Kolkata') as datetime_ist,
    actual_close,
    predicted_close
FROM btc_usd_catalog.btc_usd_gold.btc_usd_dashboard
ORDER BY datetime DESC
LIMIT 5
""")

display(df)