# Databricks notebook source
# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 2. LOAD DATA FROM DELTA TABLE
# ==============================
df = spark.sql("""
SELECT 
    datetime,
    actual_close,
    predicted_close
FROM btc_usd_catalog.btc_usd_gold.btc_usd_dashboard
WHERE datetime >= current_timestamp() - INTERVAL 48 HOURS
ORDER BY datetime
""")

# ==============================
# 3. CONVERT TO PANDAS
# ==============================
pdf = df.toPandas()

# ==============================
# 4. DATA CLEANING (IMPORTANT)
# ==============================
# Convert datetime
pdf['datetime'] = pd.to_datetime(pdf['datetime'])

# Sort properly (safety)
pdf = pdf.sort_values(by='datetime')

# Drop nulls (if any)
pdf = pdf.dropna()

# ==============================
# 5. OPTIONAL: ERROR CALCULATION
# ==============================
pdf['error'] = abs(pdf['actual_close'] - pdf['predicted_close'])

# ==============================
# 6. PLOT GRAPH
# ==============================
plt.figure()

# Actual price line
plt.plot(
    pdf['datetime'], 
    pdf['actual_close'], 
    label='Actual Price',
    linestyle='-'
)

# Predicted price line (best fit)
plt.plot(
    pdf['datetime'], 
    pdf['predicted_close'], 
    label='Predicted Price (LR)',
    linestyle='--'
)

# Labels and title
plt.xlabel("Datetime")
plt.ylabel("BTC Price")
plt.title("BTC-USD Actual vs Predicted (Last 48 Hours)")

# Extras
plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.tight_layout()

# ==============================
# 7. SHOW GRAPH
# ==============================
plt.show()

# COMMAND ----------

plt.figure()

plt.plot(
    pdf['datetime'], 
    pdf['error'], 
    label='Prediction Error'
)

plt.xlabel("Datetime")
plt.ylabel("Error")
plt.title("Model Error Over Time")

plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.tight_layout()
plt.show()