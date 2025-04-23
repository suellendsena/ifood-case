# Databricks notebook source
# MAGIC %run /Users/suellensenads@gmail.com/ifood-case/src/data/pre_processing

# COMMAND ----------

preprocessor = DatasetPreprocessor()
offers_df = spark.read.json("dbfs:/FileStore/raw/offers.json")
customers_df = spark.read.json("dbfs:/FileStore/raw/profile.json")
transactions_df = spark.read.json("dbfs:/FileStore/raw/transactions.json")

final_df = preprocessor.run(offers_df, customers_df, transactions_df)

# COMMAND ----------

import os

from pyspark.sql.functions import (
    round, avg, col, count, min, max, avg, when
)
from pyspark.sql.window import Window

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

interim_df = spark.read.parquet("dbfs:/FileStore/interim/interim.parquet")


# COMMAND ----------

interim_df.show(5, truncate=False)

# COMMAND ----------

interim_df.filter(col("age") < 0).show(5, truncate=False)

# COMMAND ----------

interim_df.groupBy("target_converted").count().show()

# COMMAND ----------

30617/(30617+108336)

# COMMAND ----------

# MAGIC %md
# MAGIC Classes desbalanceadas, as ofertas convertidas representam apenas 22% das transações.

# COMMAND ----------

(
    interim_df.withColumn(
        "real_amount_group",
        when(col("real_amount") < 0, "Unknown")
        .when(col("real_amount") < 50, "0-49")
        .when(col("real_amount") < 100, "50-99")
        .when(col("real_amount") < 150, "100-149")
        .when(col("real_amount") < 200, "150-199")
        .when(col("real_amount") < 250, "200-249")
        .when(col("real_amount") < 300, "250-299")
        .otherwise("300+")
    )
    .groupBy("real_amount_group")
    .agg(round(avg("target_converted"), 3).alias("conversion_rate"))
    .orderBy("real_amount_group")
).orderBy("conversion_rate").show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC A faixa com valor total da transação possui uma taxa de conversão superior. Mesmo com a oferta, o cliente ainda gasta mais. 

# COMMAND ----------

interim_df.filter(col("age") < 0).groupBy("completed_offer_types").count().orderBy("count").show()

# COMMAND ----------

# MAGIC %md
# MAGIC *Discount* é o tipo de cupom mais utilizado.

# COMMAND ----------

(
    interim_df.groupBy("gender", "target_converted")
    .agg(count("*").alias("n"))
    .groupBy("gender")
    .pivot("target_converted")
    .sum("n")
    .withColumnRenamed("0", "not_converted")
    .withColumnRenamed("1", "converted")
    .fillna(0)
    .withColumn("conversion_rate", round(col("converted") / (col("converted") + col("not_converted")), 3))
).show()


# COMMAND ----------

# MAGIC %md
# MAGIC As proporções de sexo entre cliente convertidos ou não são bem balanceadas.

# COMMAND ----------

(
    interim_df.withColumn(
        "age_group",
        when(col("age") < 0, "Unknown")
        .when(col("age") < 10, "0-9")
        .when(col("age") < 20, "10–19")
        .when(col("age") < 30, "20–29")
        .when(col("age") < 40, "30–39")
        .when(col("age") < 50, "40–49")
        .when(col("age") < 60, "50–59")
        .when(col("age") < 70, "60–69")
        .when(col("age") < 80, "70–79")
        .otherwise("80+")
    )
    .groupBy("age_group")
    .agg(round(avg("target_converted"), 3).alias("conversion_rate"))
    .orderBy("age_group")
).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC As faixas acima de 40 anos concentram uma proporção interessante de clientes convertidos. 

# COMMAND ----------

(
    interim_df.groupBy("reg_year")
    .agg(round(avg("target_converted"), 3).alias("conversion_rate"))
    .orderBy("reg_year")
).show()


# COMMAND ----------

(
    interim_df.groupBy("reg_month")
    .agg(round(avg("target_converted"), 3).alias("conversion_rate"))
    .orderBy("reg_month")
).show()


# COMMAND ----------

# MAGIC %md
# MAGIC Mês e ano não possuem impacto na conversão, não é possível levantar hipótese de sazonalidade.

# COMMAND ----------

(
    interim_df.groupBy("target_converted")
    .agg(round(avg("real_amount"), 2).alias("avg_real_amount"))
).show()


# COMMAND ----------

(
    interim_df.groupBy("target_converted")
    .agg(round(avg("time_since_test_start"), 2).alias("time_since_test_start"))
).show()


# COMMAND ----------

# MAGIC %md
# MAGIC O tempo da promoção possui leve impacto na conversão.

# COMMAND ----------

