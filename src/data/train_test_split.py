import os
import logging
import click
import random

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

@click.command()
@click.option('--dataset_name', default='processed.parquet', help='Name of the processed dataset', type=str)
def main(dataset_name):
    # Logger configuration
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Starting dataset split - Promotion classification')

    # Initialize Spark session
    spark = SparkSession.builder.appName("SplitDataset").getOrCreate()

    # Load dataset
    dataset_path = os.path.join('data', 'processed', dataset_name)
    df = spark.read.parquet(dataset_path)

    logger.info(f'Dataset shape: ({df.count()}, {len(df.columns)})')

    # Get distinct account_ids and shuffle
    account_ids_df = df.select("account_id").distinct()
    account_ids = [row["account_id"] for row in account_ids_df.collect()]
    random.seed(96)
    random.shuffle(account_ids)

    # Manual train/test split
    split_index = int(0.8 * len(account_ids))
    train_ids = account_ids[:split_index]
    test_ids = account_ids[split_index:]

    # Verificação de vazamento de dados
    intersect = set(train_ids) & set(test_ids)
    if intersect:
        logger.error(f"Leakage detected! Shared account_ids: {len(intersect)}")
        raise ValueError("Data leakage detected: some account_ids appear in both train and test sets.")
    else:
        logger.info("Leakage check passed. No shared account_ids between train and test.")

    # Filter the main DataFrame using the split account_id lists
    df_train = df.filter(col("account_id").isin(train_ids))
    df_test = df.filter(col("account_id").isin(test_ids))

    logger.info(f'Train set shape: ({df_train.count()}, {len(df_train.columns)})')
    logger.info(f'Test set shape: ({df_test.count()}, {len(df_test.columns)})')

    # Ensure output directory exists
    os.makedirs(os.path.join('data', 'train_test'), exist_ok=True)

    # Save as Parquet
    df_train.write.mode('overwrite').parquet(os.path.join('data', 'train_test', 'train.parquet'))
    df_test.write.mode('overwrite').parquet(os.path.join('data', 'train_test', 'test.parquet'))

    logger.info('Success! Train and test sets saved as Parquet.')


if __name__ == '__main__':
    main()
