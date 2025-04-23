# Databricks notebook source
# MAGIC %run /Users/suellensenads@gmail.com/ifood-case/src/utils/transformers

# COMMAND ----------

# MAGIC %run /Users/suellensenads@gmail.com/ifood-case/src/features/build_features

# COMMAND ----------

from pyspark.sql.functions import col, avg, stddev, count, round 

# COMMAND ----------

input_path = "/FileStore/interim/interim.parquet"
output_path = "/FileStore/processed/processed.parquet"

pipeline = FeatureBuilderPipeline(input_path, output_path)
final_df = pipeline.run(spark, BuildFeatures())


# COMMAND ----------

processed_df = spark.read.parquet("dbfs:/FileStore/processed/processed.parquet")

# COMMAND ----------

processed_df.show()

# COMMAND ----------

processed_df.select("limit_factor_vs_tx", "amount_pct_limit", "pct_current_vs_total_session",
                    "n_conversions_so_far", "amount_cumulative", "n_transactions_so_far", 
                    "real_amount_per_limit", "target_converted").show(5)

# COMMAND ----------

features = [
    "limit_factor_vs_tx",
    "amount_pct_limit",
    "pct_current_vs_total_session",
    "n_conversions_so_far",
    "amount_cumulative",
    "n_transactions_so_far",
    "real_amount_per_limit"
]

for feature in features:
    print(f"\n=== {feature} grouped by target_converted ===")
    processed_df.filter(col('credit_card_limit') > 0).groupBy("target_converted").agg(
        round(avg(col(feature)), 4).alias("mean"),
        round(stddev(col(feature)), 4).alias("stddev"),
        count("*").alias("count")
    ).orderBy("target_converted").show()
