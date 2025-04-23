import os
import yaml
import json
import logging
import warnings
import optuna
import click

from pyspark.sql.functions import rand, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from utils.spark_session import get_spark_session
from utils.training_utils import find_specific_variables, evaluate_gbt_model, objective


@click.command()
@click.option('--configfile', default='feature_config.yaml', help='YAML file describing features', type=str)
@click.option('--dataset_name', default='train.parquet', help='Training dataset name', type=str)
def main(configfile, dataset_name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info('Initializing Spark session...')
    spark = get_spark_session("GBT_Tuning_GroupKFold")

    logger.info('Loading dataset...')
    df = spark.read.parquet(os.path.join('data', 'train_test', dataset_name))
    logger.info(f'Dataset shape: {df.count()} rows, {len(df.columns)} columns')

    logger.info('Loading feature config...')
    feature_config = yaml.safe_load(open(os.path.join('src', 'config', configfile), 'r'))
    feature_target = find_specific_variables(feature_config, 'target', specific_value=True)
    feature_target = feature_target[0] if isinstance(feature_target, list) else feature_target

    logger.info('Loading selected features...')
    selected_yaml = yaml.safe_load(open(os.path.join('src', 'features', 'selected', 'features_selected.yaml'), 'r'))
    feature_cols = selected_yaml.get("support_random_forest") or selected_yaml.get("support_boruta") or []
    if "index" in feature_cols:
        feature_cols.remove("index")

    logger.info(f"Features used in tuning: {feature_cols}")

    num_folds = 5
    account_folds = (
        df.select("account_id")
        .distinct()
        .withColumn("fold", (rand(seed=96) * num_folds).cast("int"))
    )
    df_with_folds = df.join(account_folds, on="account_id", how="left")
    df_with_folds = df_with_folds.withColumnRenamed(feature_target, "label")

    logger.info('Starting Optuna study...')
    study = optuna.create_study(direction='maximize', study_name='GBTClassifier_GroupCV', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, df_with_folds, feature_cols, label_col="label", num_folds=num_folds), n_trials=30, n_jobs=1, show_progress_bar=True)

    logger.info('Tuning complete.')
    logger.info(f'Best trial: {study.best_trial.params}')

    os.makedirs('models', exist_ok=True)
    with open('models/tuning_best_params_gbt.json', 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    study.trials_dataframe().to_csv('models/tuning_results_gbt.csv', index=False)

    logger.info('Artifacts saved.')
    spark.stop()


if __name__ == '__main__':
    main()