# Databricks notebook source
# MAGIC %run /Users/suellensenads@gmail.com/ifood-case/src/features/feature_selection

# COMMAND ----------

dataset_path = "/FileStore/train_test/train.parquet"
selector = RandomForestFeatureSelector(dataset_path)
results = selector.run(spark)

# COMMAND ----------

import json

with open("/dbfs/FileStore/features/features_selected.json", "r") as f:
    selected = json.load(f)
    selected_features = selected["support_random_forest"]

# COMMAND ----------

# MAGIC %run /Users/suellensenads@gmail.com/ifood-case/src/features/create_encoders

# COMMAND ----------

builder = SparkEncoderBuilder(dataset_path="/FileStore/train_test/train.parquet")
df_encoded = builder.run(spark)

# COMMAND ----------

import logging
import json
from pyspark.sql.functions import rand, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

df_config = spark.table("feature_config")

feature_config = {}

for field in df_config.schema.fields:
    feature_name = field.name
    metadata = df_config.select(f"{feature_name}.*").first().asDict()
    feature_config[feature_name] = metadata


# COMMAND ----------

features_selected_path = "/dbfs/FileStore/features/features_selected.json"
train_data_path = "/dbfs/FileStore/train_test/train.parquet"

with open(features_selected_path, "r") as f:
    selected_json = json.load(f)
    selected_features = selected_json["support_random_forest"]

def find_specific_variables(config, key, specific_value=True):
    return [k for k, v in config.items() if key in v and v[key] == specific_value]

target_col = find_specific_variables(feature_config, "target")[0]

feature_cols = [f for f in selected_features if f != "index" and f != target_col]


# COMMAND ----------

train_df = spark.read.parquet("dbfs:/FileStore/train_test/train.parquet")
train_df.show(5)

# COMMAND ----------

num_folds = 5
account_folds = (
    train_df
    .select("account_id")
    .distinct()
    .withColumn("fold", (rand(seed=96) * num_folds).cast("int"))
)
df_with_folds = train_df.join(account_folds, on="account_id", how="left")


# COMMAND ----------

models = {
    'DT': DecisionTreeClassifier(labelCol="label", featuresCol="features"),
    'RF': RandomForestClassifier(labelCol="label", featuresCol="features", seed=96),
    'GBT': GBTClassifier(labelCol="label", featuresCol="features", seed=96)
}

# COMMAND ----------


evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

# COMMAND ----------

def apply_manual_sampling(df, label_col="label", minority_class=1):
    df_pos = df.filter(col(label_col) == minority_class)
    df_neg = df.filter(col(label_col) != minority_class)
    neg_fraction = df_pos.count() / df_neg.count()
    df_neg_sampled = df_neg.sample(withReplacement=False, fraction=neg_fraction, seed=96)
    return df_pos.union(df_neg_sampled)

# COMMAND ----------


final_results = {"original": {}, "manual_sampling": {}}

for setting in ["original", "manual_sampling"]:
    print(f"\n===== EVALUATION: {setting.upper()} =====")

    for model_name, classifier in models.items():
        print(f"\n---- Model: {model_name} ----")
        auc_scores = []

        for fold in range(num_folds):
            print(f"Fold {fold + 1}/{num_folds}")

            train_fold = df_with_folds.filter(col("fold") != fold)
            valid_fold = df_with_folds.filter(col("fold") == fold)

            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            train_vec = assembler.transform(train_fold).withColumnRenamed(target_col, "label").select("features", "label")
            valid_vec = assembler.transform(valid_fold).withColumnRenamed(target_col, "label").select("features", "label")

            if setting == "manual_sampling":
                train_vec = apply_manual_sampling(train_vec, label_col="label")

            model = classifier.fit(train_vec)
            predictions = model.transform(valid_vec)
            auc = evaluator.evaluate(predictions)
            auc_scores.append(auc)
            print(f"AUC: {auc:.4f}")

        mean_auc = sum(auc_scores) / num_folds
        final_results[setting][model_name] = mean_auc
        print(f"Mean AUC: {mean_auc:.4f}")

# COMMAND ----------

# Print summary
print("\n===== FINAL RESULTS SUMMARY =====")
for setting in final_results:
    print(f"\nSetting: {setting}")
    for model_name, score in final_results[setting].items():
        print(f"{model_name}: AUC = {score:.4f}")

# COMMAND ----------

