# Databricks notebook source
from pyspark.sql.functions import (
    col, coalesce, sum, min, max, avg, to_date, dayofmonth, month, year, array_contains, lit
)


# COMMAND ----------

def describe_column(df, categoric_column, numeric_column):
    """
    Computes basic descriptive statistics (min, max, mean) for a numeric column,
    grouped by the categoric column.

    """

    stats_df = (
        df
        .groupBy(categoric_column)
        .agg(
            min(numeric_column).alias(f"min_{numeric_column}"),
            avg(numeric_column).alias(f"mean_{numeric_column}"),
            max(numeric_column).alias(f"max_{numeric_column}")
        )
    )
    
    return stats_df

# COMMAND ----------

offers_df = spark.read.json("dbfs:/FileStore/raw/offers.json")
customers_df = spark.read.json("dbfs:/FileStore/raw/profile.json")
transactions_df = spark.read.json("dbfs:/FileStore/raw/transactions.json")

# COMMAND ----------

offers_df.show(5, truncate=False)

# COMMAND ----------

channel_list = ["web", "email", "mobile", "social"]

for ch in channel_list:
    offers_df = (offers_df.withColumn(f"{ch}", array_contains("channels", lit(ch)).cast("int")))

offers_df = offers_df.drop("channels")

offers_df.show(5, truncate=False)


# COMMAND ----------

customers_df.show(5, truncate=False)

# COMMAND ----------

transactions_df.show(5, truncate=False)

# COMMAND ----------

transactions_df.printSchema()

# COMMAND ----------

transactions_df = (
    transactions_df
    .withColumn("amount", col("value.amount"))
    .withColumn("offer_id_tmp1", col("value.`offer id`")) 
    .withColumn("offer_id_tmp2", col("value.offer_id"))
    .withColumn("reward", col("value.reward"))
    .withColumn("offer_id", coalesce(col("offer_id_tmp2"), col("offer_id_tmp1")))
    .drop("offer_id_tmp1", "offer_id_tmp2", "value")
)


# COMMAND ----------

transactions_df.show(5, truncate=False)

# COMMAND ----------

display("offers", offers_df.count(), "customers", customers_df.count(), "transactions", transactions_df.count())

# COMMAND ----------

offers_df = offers_df.dropDuplicates()
customers_df = customers_df.dropDuplicates()
transactions_df = transactions_df.dropDuplicates()

# COMMAND ----------

# ~ 400 duplicate values in transactions_df

display("offers", offers_df.count(), "customers", customers_df.count(), "transactions", transactions_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA transactions_df

# COMMAND ----------

transactions_df.show(5, truncate=False)

# COMMAND ----------

print(f"shape: {transactions_df.count()}, {len(transactions_df.columns)}")

# COMMAND ----------

transactions_df.select([
    sum(col(c).isNull().cast("int")).alias(c) for c in transactions_df.columns
]).show()

# COMMAND ----------

for col_name in transactions_df.columns:
    print(f"{col_name}: {transactions_df.select(col_name).distinct().count()} distinct values")

# COMMAND ----------

transactions_df.groupBy("event").count().orderBy("count", ascending=False).show()

# COMMAND ----------

describe_column(transactions_df, "event", "amount").orderBy("event").show()

# COMMAND ----------

describe_column(transactions_df, "event", "time_since_test_start").orderBy("event").show()

# COMMAND ----------

describe_column(transactions_df, "event", "reward").orderBy("event").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA customer_df

# COMMAND ----------

customers_df.show(5, truncate=False)

# COMMAND ----------

print(f"shape: {customers_df.count()}, {len(customers_df.columns)}")

# COMMAND ----------

customers_df.select([
    sum(col(c).isNull().cast("int")).alias(c) for c in customers_df.columns
]).show()

# COMMAND ----------

for col_name in customers_df.columns:
    print(f"{col_name}: {customers_df.select(col_name).distinct().count()} distinct values")

# COMMAND ----------

customers_df.groupBy("gender").count().orderBy("count", ascending=False).show()

# COMMAND ----------

describe_column(customers_df, "gender", "credit_card_limit").orderBy("gender").show()

# COMMAND ----------

describe_column(customers_df, "gender", "age").orderBy("gender").show()

# COMMAND ----------

customers_df.filter(col("age") > 110).count()

# COMMAND ----------

customers_df = (
    customers_df
    .withColumn("registered_on", to_date("registered_on", "yyyyMMdd"))
    .withColumn("reg_day", dayofmonth("registered_on"))
    .withColumn("reg_month", month("registered_on"))
    .withColumn("reg_year", year("registered_on"))
)

customers_df.show(5, truncate=False)

# COMMAND ----------

customers_df.groupBy("reg_month").count().orderBy("reg_month").show(12)

# COMMAND ----------

customers_df.groupBy("reg_year").count().orderBy("reg_year").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA offers_df

# COMMAND ----------

offers_df.show(10, truncate=False)

# COMMAND ----------

print(f"shape: {offers_df.count()}, {len(offers_df.columns)}")

# COMMAND ----------

for col_name in offers_df.columns:
    print(f"{col_name}: {offers_df.select(col_name).distinct().count()} distinct values")

# COMMAND ----------

