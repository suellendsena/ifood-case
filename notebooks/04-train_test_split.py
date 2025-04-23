# Databricks notebook source
# MAGIC %run /Users/suellensenads@gmail.com/ifood-case/src/data/train_test_split

# COMMAND ----------

dataset_path = "/FileStore/processed/processed.parquet"
train_path = "/FileStore/train_test/train.parquet"
test_path = "/FileStore/train_test/test.parquet"

splitter = DatasetSplitter(dataset_path, train_path, test_path)
df_train, df_test = splitter.run(spark)

# COMMAND ----------

train_df = spark.read.parquet("dbfs:/FileStore/train_test/train.parquet")

# COMMAND ----------

train_df.show(5, truncate=False)

# COMMAND ----------

test_df = spark.read.parquet("dbfs:/FileStore/train_test/test.parquet")

# COMMAND ----------

test_df.show(5, truncate=False)

# COMMAND ----------

