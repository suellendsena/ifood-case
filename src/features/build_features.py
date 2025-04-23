# Databricks notebook source
import logging

class FeatureBuilderPipeline:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, spark_session, transformer):
        self.logger.info("Starting feature creation process.")

        self.logger.info(f"Reading data from: {self.input_path}")
        df = spark_session.read.parquet(self.input_path)
        self.logger.info(f"Initial DataFrame shape: {df.count()} rows, {len(df.columns)} columns.")

        self.logger.info("Transforming data with BuildFeatures.")
        df_transformed = transformer.transform(df)

        self.logger.info(f"Transformation completed. Final shape: {df_transformed.count()} rows, {len(df_transformed.columns)} columns.")

        self.logger.info(f"Saving processed data to: {self.output_path}")
        df_transformed.write.mode("overwrite").parquet(self.output_path)
        self.logger.info("Processed data saved successfully.")

        return df_transformed 
