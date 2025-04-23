# Databricks notebook source
import os
import logging
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np

from sklearn.metrics import precision_recall_curve, classification_report, f1_score

from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassificationModel

# COMMAND ----------

df = spark.read.parquet("dbfs:/FileStore/train_test/test.parquet")
df.show(5)

# COMMAND ----------

test_path = os.path.join("dbfs", "FileStore", "data", "train_test", "test.parquet")
features_selected_path = os.path.join("dbfs", "FileStore", "src", "features", "selected", "features_selected.json")
model_path = os.path.join("dbfs", "FileStore", "models", "final_gbt_model")

# COMMAND ----------

df_config = spark.table("feature_config")

feature_config = {}

for field in df_config.schema.fields:
    feature_name = field.name
    metadata = df_config.select(f"{feature_name}.*").first().asDict()
    feature_config[feature_name] = metadata

# COMMAND ----------

features_selected_path = "/dbfs/FileStore/features/features_selected.json"

with open(features_selected_path, "r") as f:
    selected_json = json.load(f)
    selected_features = selected_json["support_random_forest"]

def find_specific_variables(config, key, specific_value=True):
    return [k for k, v in config.items() if key in v and v[key] == specific_value]

target_col = find_specific_variables(feature_config, "target")[0]

feature_cols = [f for f in selected_features if f != "index" and f != target_col]


# COMMAND ----------

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_test = assembler.transform(df.select(*(feature_cols + [target_col]))).select("features", col(target_col).alias("label"))

df_test.show()

# COMMAND ----------

model_path = "dbfs:/FileStore/models/model_gbt"
model = GBTClassificationModel.load(model_path)

predictions = model.transform(df_test)

# COMMAND ----------

predictions

# COMMAND ----------

extract_prob_1 = udf(lambda v: float(v[1]), FloatType())
predictions = predictions.withColumn("prob_1", extract_prob_1(col("probability")))

# COMMAND ----------

pred_pd = predictions.select("prob_1", "prediction", "label").toPandas()
probs = pred_pd["prob_1"].values
true_labels = pred_pd["label"].values

# COMMAND ----------

importances = model.featureImportances.toArray()
final_features = selected_features[:len(importances)]

assembler = VectorAssembler(inputCols=final_features, outputCol="features")
df_test = assembler.transform(df.select(*(final_features + [target_col])))
df_test = df_test.select("features", col(target_col).alias("label"))

engineered_features = [
    'real_amount_per_limit',
    'n_transactions_so_far',
    'amount_cumulative',
    'n_conversions_so_far',
    'pct_current_vs_total_session',
    'amount_pct_limit',
    'limit_factor_vs_tx'
]

feature_importance_df = pd.DataFrame({
    'feature': final_features,
    'importance': importances
})

feature_importance_df = feature_importance_df[feature_importance_df['importance'] > 0]

feature_importance_df['color'] = feature_importance_df['feature'].apply(
    lambda x: '#ff4c4c' if x in engineered_features else '#800000' )

feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x='importance',
    y='feature',
    data=feature_importance_df,
    palette=feature_importance_df['color'].tolist()
)
plt.title('Feature Importance - GBT Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# COMMAND ----------

# Calculate total importance of engineered vs original features
engineered_total = feature_importance_df[
    feature_importance_df['feature'].isin(engineered_features)
]['importance'].sum()

overall_total = feature_importance_df['importance'].sum()

engineered_percentage = engineered_total / overall_total * 100
print(f"Engineered features account for {engineered_percentage:.2f}% of total importance.")


# COMMAND ----------

pred_pd['score'] = (pred_pd['prob_1'] * 1000).astype(int)

pred_pd['rank_score'] = pred_pd['score'].rank(method='first')

pred_pd['decil'] = pd.qcut(pred_pd['rank_score'], q=10, labels=False)

results = (
    pred_pd.groupby('decil').label.sum() / pred_pd.label.sum()
).reset_index()

results['decil'] += 1

results['offers'] = pred_pd[pred_pd.label == 1].groupby('decil').size()
results['total_items'] = pred_pd.groupby('decil').size().astype(int)
results['no_offers'] = results['total_items'] - results['offers']
results['score (>=)'] = round(pred_pd.groupby('decil').score.min(), 3)

results.fillna(0, inplace=True)

results.sort_values(by='decil', ascending=False, inplace=True)

results = results[['decil', 'score (>=)', 'offers', 'no_offers', 'total_items', 'label']]
results = results.rename(columns={'label': 'recall'})

results['recall'] = results['recall'].cumsum()
results['precision'] = results['offers'] / results['total_items']

results['no_offers'] = results['no_offers'].cumsum()
results['total_items'] = results['total_items'].cumsum()

results.reset_index(drop=True, inplace=True)
results


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC A tabela mostra a performance do modelo ao segmentar clientes em 10 decis com base em um score preditivo, onde **Decil 10 representa os clientes com maior score (maior probabilidade de conversão)**.
# MAGIC
# MAGIC * **Decis superiores (10 a 8)** concentram os clientes com maior probabilidade de conversão, apresentando **maior precisão**. Contudo, como esses grupos contêm uma menor fração da base, o **recall acumulado** ainda é limitado — indicando que, embora os clientes desses decis sejam altamente qualificados, eles não representam a maioria dos casos positivos.
# MAGIC
# MAGIC * **Decis intermediários e inferiores (7 a 1)** abrangem um número muito maior de clientes. Com isso, o modelo consegue **capturar praticamente todos os casos positivos** (recall atinge 100% no Decil 1), mas à custa de **baixa precisão** — ou seja, muitos falsos positivos. 
# MAGIC
# MAGIC
# MAGIC Dessa forma, **ações de marketing mais direcionadas devem priorizar os decis superiores**, que possuem maior concentração de clientes com perfil de conversão, otimizando o custo-benefício.
# MAGIC

# COMMAND ----------


f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Best threshold based on F1-score: {best_threshold:.4f}")

pred_pd["prediction_thresholded"] = (pred_pd["prob_1"] >= best_threshold).astype(int)

report_thresh = classification_report(
    y_true=pred_pd["label"],
    y_pred=pred_pd["prediction_thresholded"],
    digits=4
)

print("Classification Report using best threshold (based on F1-score):")
print(report_thresh)


# COMMAND ----------

