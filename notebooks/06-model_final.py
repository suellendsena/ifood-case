# Databricks notebook source
# MAGIC %run /Users/suellensenads@gmail.com/ifood-case/src/model/tuning

# COMMAND ----------

tuner = GBTOptunaTuner(dataset_path="/FileStore/train_test/train.parquet")
best_params = tuner.run(spark)

# COMMAND ----------

with open("/dbfs/FileStore/models/tuning_best_params_gbt.json", "r") as f:
    best_params = json.load(f)

# COMMAND ----------

import os
import json
import logging
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import classification_report
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf, col

# COMMAND ----------

# Read feature_config Delta table
df_config = spark.table("feature_config")

# Build feature configuration dictionary
feature_config = {}

for field in df_config.schema.fields:
    feature_name = field.name
    metadata = df_config.select(f"{feature_name}.*").first().asDict()
    feature_config[feature_name] = metadata


# COMMAND ----------

# Paths
features_selected_path = "/dbfs/FileStore/features/features_selected.json"

# Load selected features from JSON
with open(features_selected_path, "r") as f:
    selected_json = json.load(f)
    selected_features = selected_json["support_random_forest"]

def find_specific_variables(config, key, specific_value=True):
    return [k for k, v in config.items() if key in v and v[key] == specific_value]

target_col = find_specific_variables(feature_config, "target")[0]

feature_cols = [f for f in selected_features if f != "index" and f != target_col]


# COMMAND ----------

# Load training data
train_df = spark.read.parquet("dbfs:/FileStore/train_test/train_encoded.parquet")
train_df.show(5)

# COMMAND ----------

train_df.show(5)

# COMMAND ----------

if target_col != "label":
    train_df = train_df.withColumnRenamed(target_col, "label")

# COMMAND ----------

params_path =  "/dbfs/FileStore/models/tuning_best_params_gbt.json"

# Load hyperparameters
with open(params_path, 'r') as f:
    best_params = json.load(f)

# COMMAND ----------

model_output_path = "dbfs:/FileStore/models/model_gbt"

# Train and save model
gbt = GBTClassifier(
    labelCol="label",
    featuresCol="features",
    maxDepth=int(best_params["maxDepth"]),
    maxIter=int(best_params["maxIter"]),
    stepSize=float(best_params["stepSize"]),
    seed=96
)

final_model = gbt.fit(train_df.select("features", "label"))
final_model.write().overwrite().save(model_output_path)

print(f"Final GBT model trained and saved to: {model_output_path}")

# COMMAND ----------

predictions = final_model.transform(train_df.select("features", "label"))

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = evaluator.evaluate(predictions)
print(f"AUC on training set: {auc:.4f}")

extract_prob_1 = udf(lambda v: float(v[1]), FloatType())
predictions = predictions.withColumn("prob_1", extract_prob_1(col("probability")))

pred_pd = predictions.select("prediction", "label", "prob_1").toPandas()

report = classification_report(
    y_true=pred_pd["label"],
    y_pred=pred_pd["prediction"],
    digits=4
)
print("\nClassification Report (Training Set):")
print(report)


# COMMAND ----------

