# Databricks notebook source
# MAGIC %run /Users/suellensenads@gmail.com/ifood-case/src/data/extraction_data

# COMMAND ----------

url = "https://data-architect-test-source.s3.sa-east-1.amazonaws.com/ds-technical-evaluation-data.tar.gz"
loader = RawDataLoader(url)
loader.download_tarball()
loader.extract_tarball()
loader.organize_files()

# COMMAND ----------

offers_df = spark.read.json("dbfs:/FileStore/raw/offers.json")
customers_df = spark.read.json("dbfs:/FileStore/raw/profile.json")
transactions_df = spark.read.json("dbfs:/FileStore/raw/transactions.json")

# COMMAND ----------

offers_df.show(5, truncate=False)

# COMMAND ----------

customers_df.show(5, truncate=False)

# COMMAND ----------

transactions_df.show(5, truncate=False)

# COMMAND ----------

transactions_df.printSchema()

# COMMAND ----------

