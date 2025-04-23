import os
import yaml
import logging
import click

from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

from utils.spark_session import get_spark_session
from utils.training_utils import find_specific_variables

@click.command()
@click.option('--configfile', default='feature_config.yaml', help='Feature description file', type=str)
@click.option('--dataset_name', default='train.parquet', help='Training dataset name', type=str)
def main(configfile, dataset_name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('Starting encoder creation process (PySpark version)')

    spark = get_spark_session("EncoderBuilder")

    df = spark.read.parquet(os.path.join('data', 'train_test', dataset_name))
    logger.info(f'Dataset loaded. Shape: ({df.count()}, {len(df.columns)})')

    features = yaml.safe_load(open(os.path.join('src', 'config', configfile), 'r'))
    target_col = find_specific_variables(features, 'target', specific_value=True)
    target_col = target_col[0] if isinstance(target_col, list) else target_col

    try:
        selected_features_yaml = yaml.safe_load(open(os.path.join('src', 'features', 'selected', 'features_selected.yaml')))
        selected_features = selected_features_yaml.get("support_random_forest") or selected_features_yaml.get("support_boruta")
    except Exception as e:
        logger.error('Could not read feature selection output file.')
        raise e

    hard_remove = find_specific_variables(features, 'hard_remove', specific_value=True)
    selected_features = list(set(selected_features) - set(hard_remove))

    logger.info(f'Selected features for encoding: {selected_features}')
    string_cols = [f.name for f in df.schema.fields if f.name in selected_features and f.dataType.simpleString() == 'string']
    logger.info(f'String columns identified: {string_cols}')

    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in string_cols
    ]

    final_features = [
        f"{col}_idx" if col in string_cols else col
        for col in selected_features
        if col != target_col
    ]

    assembler = VectorAssembler(inputCols=final_features, outputCol="features")

    pipeline = Pipeline(stages=indexers + [assembler])
    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)

    logger.info('Transformation completed. Saving outputs...')

    df_transformed.select("features", target_col).write.mode("overwrite").parquet(
        os.path.join("data", "train_test", "train_encoded.parquet")
    )

    encoder_path = os.path.join("models", "encoders", "spark_encoder_pipeline")
    pipeline_model.write().overwrite().save(encoder_path)
    logger.info(f'Pipeline model saved to: {encoder_path}')

    logger.info('Success! Encoded data and encoder saved.')


if __name__ == '__main__':
    main()
