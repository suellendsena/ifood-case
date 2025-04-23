# Databricks notebook source
import os
import json
import logging
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier


class RandomForestFeatureSelector:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.output_path = "/dbfs/FileStore/features/features_selected.json"

        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_feature_config_from_table(self, spark):
        self.logger.info("Loading feature configuration from Delta table 'feature_config'")
        df_config = spark.table("feature_config")
        feature_config = {}

        for field in df_config.schema.fields:
            feature_name = field.name
            metadata = df_config.select(f"{feature_name}.*").first().asDict()
            feature_config[feature_name] = metadata

        return feature_config

    def _find_specific_variables(self, config, key, specific_value=True):
        return [
            col_name for col_name, attr in config.items()
            if key in attr and attr[key] == specific_value
        ]

    def run(self, spark):
        self.logger.info("Starting PySpark RandomForest Feature Selection")

        feature_config = self._load_feature_config_from_table(spark)

        self.logger.info(f"Reading dataset from: {self.dataset_path}")
        df = spark.read.parquet(self.dataset_path)
        self.logger.info(f"Dataset shape: ({df.count()}, {len(df.columns)})")

        aux_vars = self._find_specific_variables(feature_config, "auxiliar", True)
        hard_remove = self._find_specific_variables(feature_config, "hard_remove", True)
        drop_cols = list(set(aux_vars + hard_remove) & set(df.columns))
        df = df.drop(*drop_cols)
        self.logger.info(f"Removed {len(drop_cols)} auxiliary/hard-remove features")

        target = self._find_specific_variables(feature_config, "target", True)
        target = target[0] if isinstance(target, list) and target else None
        if not target:
            raise ValueError("No target column identified in feature configuration.")
        self.logger.info(f"Target column: {target}")

        feature_cols = [
            f.name for f in df.schema.fields
            if f.name != target and f.name != "account_id" and f.dataType.simpleString() in ["int", "double"]
        ]
        self.logger.info(f"{len(feature_cols)} numeric features to evaluate")

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_ml = assembler.transform(df.select(*(feature_cols + [target]))).select("features", col(target).alias("label"))

        rf = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=100,
            maxDepth=4,
            minInstancesPerNode=50,
            seed=96
        )
        model = rf.fit(df_ml)

        importances = model.featureImportances.toArray()
        full_ranking = list(zip(feature_cols, importances))
        full_ranking.sort(key=lambda x: x[1], reverse=True)

        importance_threshold = 0.01
        selected_features = [col for col, imp in full_ranking if imp >= importance_threshold]
        rejected_features = [col for col, imp in full_ranking if imp < importance_threshold]

        self.logger.info(f"{len(selected_features)} features selected (importance >= {importance_threshold})")
        self.logger.info(f"{len(rejected_features)} features rejected (importance < {importance_threshold})")

        result = {
            "support_random_forest": selected_features,
            "rejected_random_forest": rejected_features,
            "full_importance_ranking": [[str(f), float(round(i, 6))] for f, i in full_ranking]
        }

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "w") as f:
            json.dump(result, f, indent=2)
        self.logger.info(f"Feature selection result saved to: {self.output_path}")

        return result
