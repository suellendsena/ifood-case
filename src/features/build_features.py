import os
import logging

from utils.spark_session import get_spark_session
from utils.transformers import BuildFeatures

def main():

    logger = logging.getLogger(__name__)
    logger.info("Starting feature creation process.")

    spark = get_spark_session("features")

    input_path = os.path.join("data", "interim", "interim.parquet")
    output_path = os.path.join("data", "processed", "processed.parquet")

    logger.info(f"Reading data from: {input_path}")
    df = spark.read.parquet(input_path)
    logger.info(f"Initial DataFrame shape: {df.count()} rows, {len(df.columns)} columns.")

    logger.info("Transforming data with BuildFeatures.")
    feature_builder = BuildFeatures()
    df_transformed = feature_builder.transform(df)

    logger.info(f"Transformation completed. Final shape: {df_transformed.count()} rows, {len(df_transformed.columns)} columns.")

    logger.info(f"Saving processed data to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_transformed.write.mode("overwrite").parquet(output_path)
    logger.info("Processed data saved successfully.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
