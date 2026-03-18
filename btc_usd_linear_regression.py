# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark
import os


# Initialize Spark
def get_spark():
    return SparkSession.builder.getOrCreate()


# Set MLflow UC volume path (FIXED)
def set_mlflow_uc_volume():
    # Make sure this volume exists
    os.environ["MLFLOW_DFS_TMP"] = "/Volumes/btc_usd_catalog/btc_usd_gold/mlflow_volume/tmp"
    print("MLflow UC volume path set")


# Set MLflow experiment
def set_mlflow_experiment(experiment_name="btc_lr_experiment"):
    spark = get_spark()
    user = spark.sql("SELECT current_user()").first()[0]
    experiment_path = f"/Users/{user}/{experiment_name}"
    mlflow.set_experiment(experiment_path)
    print(f"MLflow experiment set to: {experiment_path}")


# Load dataset
def load_data(table_name):
    spark = get_spark()
    return spark.read.table(table_name)


# Split dataset (time-series safe)
def split_data(df, train_ratio=0.8):

    df = df.orderBy("datetime")  # sort data

    total_count = df.count()  # total rows
    train_count = int(total_count * train_ratio)

    from pyspark.sql.functions import row_number
    window = Window.orderBy("datetime")

    df = df.withColumn("row_num", row_number().over(window))

    train_df = df.filter(col("row_num") <= train_count).drop("row_num")
    test_df = df.filter(col("row_num") > train_count).drop("row_num")

    return train_df, test_df


# Feature columns
def get_feature_columns():
    return [
        "close",
        "volume",
        "price_change_pct",
        "hour",
        "day_of_week",
        "lag_1",
        "lag_3",
        "return_1h",
        "volume_change",
        "sma_6",
        "sma_12",
        "rolling_std_12"
    ]


# Build pipeline
def build_pipeline(feature_cols, reg, elastic_net):

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True
    )

    lr = LinearRegression(
        featuresCol="scaled_features",
        labelCol="target_price",
        regParam=reg,
        elasticNetParam=elastic_net
    )

    return Pipeline(stages=[assembler, scaler, lr])


# Train model
def train_model(pipeline, train_df):
    return pipeline.fit(train_df)


# Evaluate model
def evaluate_model(model, test_df):

    predictions = model.transform(test_df)

    metrics = {}

    for metric in ["rmse", "mse", "mae", "r2"]:
        evaluator = RegressionEvaluator(
            labelCol="target_price",
            predictionCol="prediction",
            metricName=metric
        )
        metrics[metric] = evaluator.evaluate(predictions)

    print("\nModel Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v}")

    return metrics, predictions


# Predict next hour
def predict_next_hour(model, df):

    latest_row = df.orderBy(col("datetime").desc()).limit(1)

    prediction = model.transform(latest_row)

    result = prediction.select("datetime", "close", "prediction").collect()[0]

    print("\nNext Hour Prediction:")
    print(f"Last Time   : {result['datetime']}")
    print(f"Last Close  : {result['close']}")
    print(f"Predicted   : {result['prediction']}")

    return result["prediction"]


# Train with MLflow
def train_with_mlflow(df, experiment_name="btc_lr_experiment"):

    set_mlflow_uc_volume()  # fix UC volume
    set_mlflow_experiment(experiment_name)  # set experiment

    train_df, test_df = split_data(df)
    feature_cols = get_feature_columns()

    best_model = None
    best_rmse = float("inf")

    reg_params = [0.01, 0.1, 1.0]
    elastic_net_params = [0.0, 0.5, 1.0]

    for reg in reg_params:
        for enet in elastic_net_params:

            with mlflow.start_run():

                pipeline = build_pipeline(feature_cols, reg, enet)
                model = train_model(pipeline, train_df)

                metrics, _ = evaluate_model(model, test_df)

                mlflow.log_param("regParam", reg)
                mlflow.log_param("elasticNetParam", enet)

                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                # UC volume fix here
                mlflow.spark.log_model(
                    model,
                    "lr_model",
                    dfs_tmpdir=os.environ["MLFLOW_DFS_TMP"]
                )

                print(f"\nRun: reg={reg}, elasticNet={enet}")

                if metrics["rmse"] < best_rmse:
                    best_rmse = metrics["rmse"]
                    best_model = model

    print(f"\nBest RMSE: {best_rmse}")

    predict_next_hour(best_model, df)

    return best_model


# Main pipeline
def run_training_pipeline():

    table_name = "btc_usd_catalog.btc_usd_gold.btc_usd_ml_gold"

    df = load_data(table_name)

    model = train_with_mlflow(df)

    return model


# Run pipeline
run_training_pipeline()