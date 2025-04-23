# Databricks notebook source
import os
import json
import logging
import optuna
import pandas as pd
import mlflow
from pyspark.sql.functions import rand, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


class GBTOptunaTuner:
    def __init__(self, dataset_path: str, num_folds: int = 5):
        self.dataset_path = dataset_path
        self.num_folds = num_folds
        self.feature_config_table = "feature_config"
        self.selected_features_path = "/dbfs/FileStore/features/features_selected.json"
        self.output_params_path = "/dbfs/FileStore/models/tuning_best_params_gbt.json"
        self.output_results_path = "/dbfs/FileStore/models/tuning_results_gbt.csv"

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_feature_config(self, spark):
        df_config = spark.table(self.feature_config_table)
        config = {}
        for col_name in df_config.schema.names:
            attrs = df_config.select(col_name).first()[0]
            if attrs:
                config[col_name] = attrs.asDict()
        return config

    def _find_specific_variables(self, config, key, specific_value=True):
        return [k for k, v in config.items() if key in v and v[key] == specific_value]

    def _load_selected_features(self):
        with open(self.selected_features_path, "r") as f:
            selected = json.load(f)
        return selected.get("support_random_forest", [])

    def _evaluate_model(self, params, df_with_folds, feature_cols, label_col):
        evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        aucs = []

        for fold in range(self.num_folds):
            train = df_with_folds.filter(col("fold") != fold)
            valid = df_with_folds.filter(col("fold") == fold)

            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            train_vec = assembler.transform(train).select("features", col(label_col))
            valid_vec = assembler.transform(valid).select("features", col(label_col))

            model = GBTClassifier(
                featuresCol="features",
                labelCol=label_col,
                maxDepth=params["maxDepth"],
                maxIter=params["maxIter"],
                stepSize=params["stepSize"],
                seed=96
            ).fit(train_vec)

            preds = model.transform(valid_vec)
            auc = evaluator.evaluate(preds)
            aucs.append(auc)

        return sum(aucs) / len(aucs)

    def _objective(self, trial, df_with_folds, feature_cols, label_col):
        try:
            params = {
                "maxDepth": trial.suggest_int("maxDepth", 3, 8),
                "maxIter": trial.suggest_int("maxIter", 10, 50),
                "stepSize": trial.suggest_float("stepSize", 0.01, 0.3)
            }
            trial.set_user_attr("params", params)

            auc = self._evaluate_model(params, df_with_folds, feature_cols, label_col)
            return auc
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return 0.0

    def run(self, spark):
        import mlflow

        self.logger.info("Loading dataset...")
        df = spark.read.parquet(self.dataset_path)
        self.logger.info(f"Dataset loaded with shape: ({df.count()}, {len(df.columns)})")

        self.logger.info("Loading feature configuration and selected features...")
        config = self._load_feature_config(spark)
        target_col = self._find_specific_variables(config, "target")[0]
        features = self._load_selected_features()
        features = [f for f in features if f != "index"]

        self.logger.info(f"Selected features for tuning: {features}")
        self.logger.info(f"Target column: {target_col}")

        self.logger.info("Generating cross-validation folds by account_id...")
        folds = (
            df.select("account_id").distinct()
            .withColumn("fold", (rand(seed=96) * self.num_folds).cast("int"))
        )
        df_with_folds = df.join(folds, on="account_id", how="left").withColumnRenamed(target_col, "label")

        self.logger.info("Starting hyperparameter optimization with Optuna...")

        with mlflow.start_run(run_name="GBT_Tuning_Optuna"):
            study = optuna.create_study(
                direction="maximize",
                study_name="GBTClassifier_GroupCV",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(
                lambda trial: self._objective(trial, df_with_folds, features, label_col="label"),
                n_trials=25,
                n_jobs=1,
                show_progress_bar=True
            )

            best_params = study.best_trial.params
            self.logger.info("Hyperparameter tuning completed.")
            self.logger.info(f"Best parameters found: {best_params}")

            os.makedirs(os.path.dirname(self.output_params_path), exist_ok=True)

            with open(self.output_params_path, "w") as f:
                json.dump(best_params, f, indent=2)

            study_df = study.trials_dataframe()
            study_df.to_csv(self.output_results_path, index=False)

            self.logger.info("Artifacts saved:")
            self.logger.info(f"- Best parameters: {self.output_params_path}")
            self.logger.info(f"- Full tuning results: {self.output_results_path}")

            mlflow.log_params(best_params)
            mlflow.log_artifact(self.output_params_path)
            mlflow.log_artifact(self.output_results_path)

        return best_params


# COMMAND ----------

