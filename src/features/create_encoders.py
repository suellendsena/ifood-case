import os
import yaml
import logging
import click

from pyspark.sql.functions import col, when
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

    use_for_cv = 'account_id' in df.columns

    logger.info(f'Selected features for encoding: {selected_features}')

    fillna_values = {
        "gender": "unknown",
        "credit_card_limit": -1.0,
        "age": -1
    }

    logger.info(f'Applying consistent fillna values: {fillna_values}')
    for col_name, fill_val in fillna_values.items():
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                when(col(col_name).isNull(), fill_val).otherwise(col(col_name))
            )

    os.makedirs('models/encoders', exist_ok=True)
    with open('models/encoders/fillna_values.yaml', 'w') as f:
        yaml.dump(fillna_values, f)

    string_cols = [f.name for f in df.schema.fields if f.name in selected_features and f.dataType.simpleString() == 'string']
    logger.info(f'String columns identified: {string_cols}')

    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in string_cols
    ]

    final_features = [
        f"{col}_idx" if col in string_cols else col
        for col in selected_features
        if col != target_col and col != 'account_id'
    ]

    assembler = VectorAssembler(inputCols=final_features, outputCol="features")

    pipeline = Pipeline(stages=indexers + [assembler])
    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)

    logger.info('Transformation completed. Saving outputs...')

    columns_to_save = ["features", target_col]
    if 'account_id' in df.columns:
        columns_to_save.append('account_id')

    df_transformed.select(*columns_to_save).write.mode("overwrite").parquet(
        os.path.join("data", "train_test", "train_encoded.parquet")
    )

    encoder_path = os.path.join("models", "encoders", "spark_encoder_pipeline")
    pipeline_model.write().overwrite().save(encoder_path)
    logger.info(f'Pipeline model saved to: {encoder_path}')

    logger.info('Success! Encoded data, encoder, and fillna values saved.')


if __name__ == '__main__':
    main()
