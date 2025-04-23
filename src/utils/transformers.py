from pyspark.sql.functions import (
    col, when
)
from pyspark.sql.window import Window
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


class BuildFeatures(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    PySpark custom transformer that creates session-based features from transaction data.
    
    Features created:
    - `real_amount_per_limit`: Ratio of transaction amount to credit card limit.
    - `pct_current_vs_total_session`: Ratio of the current amount to the session cumulative amount.
    - `amount_pct_limit`: Duplicate of `real_amount_per_limit` for compatibility purposes.
    - `limit_factor_vs_tx`: Ratio of credit card limit to the transaction amount.
    """

    def __init__(self):
        super().__init__()

    def _transform(self, X):
        X = (
            X
            .withColumn(
                "real_amount_per_limit",
                col("real_amount") / when(col("credit_card_limit") != 0, col("credit_card_limit")).otherwise(1)
            )
            .withColumn(
                "pct_current_vs_total_session",
                col("real_amount") / when(col("amount_cumulative") > 0, col("amount_cumulative")).otherwise(1)
            )
            .withColumn(
                "amount_pct_limit",
                col("real_amount") / when(col("credit_card_limit") != 0, col("credit_card_limit")).otherwise(1)
            )
            .withColumn(
                "limit_factor_vs_tx",
                col("credit_card_limit") / when(col("real_amount") != 0, col("real_amount")).otherwise(1)
            )
        )
        return X


class SparkSelector(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer that selects a specified subset of columns from a Spark DataFrame.
    
    Parameters:
    ----------
    features : list
        List of feature column names to keep.
    target : str
        Name of the target column to keep.
    """

    def __init__(self, features: list, target: str):
        super().__init__()
        self.features = features
        self.target = target

    def _transform(self, df):
        """
        Select only the specified features and target column.
        
        Parameters:
        ----------
        df : DataFrame
            Input DataFrame.
        
        Returns:
        -------
        DataFrame
            Subset with selected columns.
        """
        return df.select(*(self.features + [self.target]))


class FillMissingTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer that fills missing values in a DataFrame using a user-defined dictionary.
    
    Parameters:
    ----------
    fill_dict : dict
        Dictionary where keys are column names and values are default values to replace nulls.
    """

    def __init__(self, fill_dict: dict):
        super().__init__()
        self.fill_dict = fill_dict

    def _transform(self, df):
        """
        Replace null values with default values provided in fill_dict.
        
        Parameters:
        ----------
        df : DataFrame
            Input DataFrame.
        
        Returns:
        -------
        DataFrame
            Transformed DataFrame with missing values filled.
        """
        for col_name, val in self.fill_dict.items():
            if col_name in df.columns:
                df = df.withColumn(
                    col_name,
                    when(col(col_name).isNull(), val).otherwise(col(col_name))
                )
        return df
