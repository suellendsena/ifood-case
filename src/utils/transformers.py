import logging
from pyspark.sql.functions import (
    col, when, lit,
    count, sum as spark_sum, coalesce
)
from pyspark.sql.window import Window
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildFeatures(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    PySpark transformer that adds session-based features.
    """

    def __init__(self):
        super().__init__()

    def _transform(self, X):
        logger.info("Starting feature engineering...")

        session_window = Window.partitionBy("account_id") \
            .orderBy("time_since_test_start") \
            .rowsBetween(Window.unboundedPreceding, -1)

        X = (
            X
            .withColumn("real_amount_per_limit", col("real_amount") / when(col("credit_card_limit") != 0, col("credit_card_limit")).otherwise(1))
            .withColumn("n_transactions_so_far", coalesce(count("*").over(session_window), lit(0)))
            .withColumn("amount_cumulative", coalesce(spark_sum("real_amount").over(session_window), lit(0)))
            .withColumn("n_conversions_so_far", coalesce(spark_sum("target_converted").over(session_window), lit(0)))
            .withColumn("pct_current_vs_total_session", col("real_amount") / when(col("amount_cumulative") > 0, col("amount_cumulative")).otherwise(1))
            .withColumn("amount_pct_limit", col("real_amount") / when(col("credit_card_limit") != 0, col("credit_card_limit")).otherwise(1))
            .withColumn("limit_factor_vs_tx", col("credit_card_limit") / when(col("real_amount") != 0, col("real_amount")).otherwise(1))
        )

        logger.info("Feature engineering completed successfully.")
        return X


class SparkSelector(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer that selects a subset of columns from a DataFrame.
    """
    def __init__(self, features: list, target: str):
        super().__init__()
        self.features = features
        self.target = target

    def _transform(self, df):
        return df.select(*(self.features + [self.target]))


class FillMissingTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer that fills missing values with predefined values (dict-based).
    """
    def __init__(self, fill_dict: dict):
        super().__init__()
        self.fill_dict = fill_dict

    def _transform(self, df):
        for col_name, val in self.fill_dict.items():
            if col_name in df.columns:
                df = df.withColumn(
                    col_name,
                    when(col(col_name).isNull(), val).otherwise(col(col_name))
                )
        return df
