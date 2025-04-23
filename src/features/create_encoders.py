# Databricks notebook source
import os
import json
import logging

from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler


class SparkEncoderBuilder:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.encoder_path = "/dbfs/FileStore/models/encoders/spark_encoder_pipeline"
        self.output_path = "/FileStore/train_test/train_encoded.parquet"
        self.selected_features_path = "/dbfs/FileStore/features/features_selected.json"

        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_feature_config_from_table(self, spark):
        self.logger.info("Loading feature config from Delta table 'feature_config'")
        df_config = spark.table("feature_config")
        config = {}
        for col_name in df_config.schema.names:
            attrs = df_config.select(col_name).first()[0]
            if attrs:
                config[col_name] = attrs.asDict()
        return config

    def _find_specific_variables(self, config, key, specific_value=True):
        return [
            col_name for col_name, attr in config.items()
            if key in attr and attr[key] == specific_value
        ]

    def _load_selected_features(self):
        self.logger.info(f"Loading selected features from {self.selected_features_path}")
        with open(self.selected_features_path, "r") as f:
            selected = json.load(f)
        return selected.get("support_random_forest", [])

    def run(self, spark):
        self.logger.info("Starting encoder creation process")

        df = spark.read.parquet(self.dataset_path)
        self.logger.info(f"Dataset shape: ({df.count()}, {len(df.columns)})")

        features = self._load_feature_config_from_table(spark)
        selected_features = self._load_selected_features()

        target_col = self._find_specific_variables(features, "target", specific_value=True)
        target_col = target_col[0] if isinstance(target_col, list) else target_col
        self.logger.info(f"Target column: {target_col}")

        hard_remove = self._find_specific_variables(features, "hard_remove", specific_value=True)
        selected = list(set(selected_features) - set(hard_remove))
        self.logger.info(f"Features after removing hard_remove: {selected}")

        fillna_values = {
            "gender": "unknown",
            "credit_card_limit": -1.0,
            "age": -1
        }
        self.logger.info(f"Applying fillna: {fillna_values}")
        for col_name, val in fillna_values.items():
            if col_name in df.columns:
                df = df.withColumn(col_name, when(col(col_name).isNull(), val).otherwise(col(col_name)))

        string_cols = [f.name for f in df.schema.fields if f.name in selected and f.dataType.simpleString() == "string"]
        self.logger.info(f"String columns to index: {string_cols}")

        indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
            for col in string_cols
        ]

        final_features = [
            f"{col}_idx" if col in string_cols else col
            for col in selected
            if col != target_col and col != "account_id"
        ]

        assembler = VectorAssembler(inputCols=final_features, outputCol="features")
        pipeline = Pipeline(stages=indexers + [assembler])
        pipeline_model = pipeline.fit(df)
        df_transformed = pipeline_model.transform(df)

        cols_to_save = ["features", target_col]
        if "account_id" in df.columns:
            cols_to_save.append("account_id")

        df_transformed.select(*cols_to_save).write.mode("overwrite").parquet(self.output_path)

        pipeline_model.write().overwrite().save(self.encoder_path)

        self.logger.info(f"Encoded dataset saved to: {self.output_path}")
        self.logger.info(f"Encoder pipeline saved to: {self.encoder_path}")
        self.logger.info("Finished encoder process.")

        return df_transformed
