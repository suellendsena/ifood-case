import os
import yaml
import logging
import click
import warnings

from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

from utils.spark_session import get_spark_session
from utils.training_utils import find_specific_variables

warnings.filterwarnings("ignore")


@click.command()
@click.option('--configfile', default='feature_config.yaml', help='YAML file describing the features', type=str)
@click.option('--dataset_name', default='train.parquet', help='Training dataset name', type=str)
def main(configfile, dataset_name):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting PySpark RandomForest Feature Selection")

    spark = get_spark_session("RandomForestFeatureSelection")

    feature_config_path = os.path.join('src', 'config', configfile)
    feature_config = yaml.safe_load(open(feature_config_path, 'r'))

    dataset_path = os.path.join('data', 'train_test', dataset_name)
    df = spark.read.parquet(dataset_path)
    logger.info(f"Training dataset shape: ({df.count()}, {len(df.columns)})")

    aux_vars = find_specific_variables(feature_config, 'auxiliar', specific_value=True)
    hard_remove = find_specific_variables(feature_config, 'hard_remove', specific_value=True)
    drop_cols = list(set(aux_vars + hard_remove) & set(df.columns))
    df = df.drop(*drop_cols)
    logger.info(f"Removed {len(drop_cols)} auxiliary/hard-remove features")

    target = find_specific_variables(feature_config, 'target', specific_value=True)
    target = target[0] if isinstance(target, list) else target
    logger.info(f"Target column: {target}")

    feature_cols = [
        f.name for f in df.schema.fields
        if f.name != target and f.name != 'account_id' and f.dataType.simpleString() in ['int', 'double']
    ]
    logger.info(f"{len(feature_cols)} numeric features to evaluate (excluding 'account_id')")

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

    logger.info(f"{len(selected_features)} features selected (importance >= {importance_threshold}).")
    logger.info(f"{len(rejected_features)} features rejected (importance < {importance_threshold}).")

    result = {
        "support_random_forest": selected_features,
        "rejected_random_forest": rejected_features,
        "full_importance_ranking": [[str(f), float(round(i, 6))] for f, i in full_ranking]
    }

    output_yaml = os.path.join("src", "features", "selected", "features_selected.yaml")
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)

    with open(output_yaml, "w") as f:
        yaml.dump(result, f, allow_unicode=True)

    logger.info(f"Feature selection result saved to: {output_yaml}")


if __name__ == "__main__":
    main()
