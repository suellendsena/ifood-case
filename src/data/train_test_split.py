# Databricks notebook source
import logging
import random
from pyspark.sql.functions import col

class DatasetSplitter:
    def __init__(self, dataset_path: str, train_path: str, test_path: str, seed: int = 96):
        self.dataset_path = dataset_path
        self.train_path = train_path
        self.test_path = test_path
        self.seed = seed

        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, spark_session):
        self.logger.info("Starting dataset split - Promotion classification")

        df = spark_session.read.parquet(self.dataset_path)
        self.logger.info(f"Dataset shape: ({df.count()}, {len(df.columns)})")

        account_ids = [row["account_id"] for row in df.select("account_id").distinct().collect()]
        random.seed(self.seed)
        random.shuffle(account_ids)

        split_index = int(0.8 * len(account_ids))
        train_ids = account_ids[:split_index]
        test_ids = account_ids[split_index:]

        intersect = set(train_ids) & set(test_ids)
        if intersect:
            self.logger.error(f"Leakage detected! Shared account_ids: {len(intersect)}")
            raise ValueError("Data leakage detected: some account_ids appear in both train and test sets.")
        else:
            self.logger.info("Leakage check passed. No shared account_ids between train and test.")

        df_train = df.filter(col("account_id").isin(train_ids))
        df_test = df.filter(col("account_id").isin(test_ids))

        self.logger.info(f"Train set shape: ({df_train.count()}, {len(df_train.columns)})")
        self.logger.info(f"Test set shape: ({df_test.count()}, {len(df_test.columns)})")

        df_train.write.mode("overwrite").parquet(self.train_path)
        df_test.write.mode("overwrite").parquet(self.test_path)

        self.logger.info("Train and test datasets saved successfully.")
        return df_train, df_test


# COMMAND ----------

# MAGIC %md
# MAGIC