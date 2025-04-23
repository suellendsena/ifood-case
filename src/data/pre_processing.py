# Databricks notebook source
import logging
from pyspark.sql.functions import (
    col, when, max, collect_set, row_number, concat_ws,
    array_contains, coalesce, lit, to_date, dayofmonth, month, year, size,
    monotonically_increasing_id
)
from pyspark.sql.window import Window

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Preprocessor")


class DatasetPreprocessor:
    def __init__(self, channel_list=None):
        self.channel_list = channel_list or ["web", "email", "mobile", "social"]

    def preprocess_offers(self, offers_df):
        logger.info("Processing 'channels' column in offers...")
        for ch in self.channel_list:
            offers_df = offers_df.withColumn(ch, array_contains("channels", lit(ch)).cast("int"))
        return offers_df.drop("channels")

    def preprocess_transactions(self, transactions_df):
        logger.info("Extracting fields from 'value' column in transactions...")
        return (
            transactions_df
            .withColumn("amount", col("value.amount"))
            .withColumn("offer_id_tmp1", col("value.`offer id`"))
            .withColumn("offer_id_tmp2", col("value.offer_id"))
            .withColumn("reward", col("value.reward"))
            .withColumn("offer_id", coalesce(col("offer_id_tmp2"), col("offer_id_tmp1")))
            .drop("offer_id_tmp1", "offer_id_tmp2", "value")
        )

    def run(self, offers_df, customers_df, transactions_df):
        logger.info("Starting full preprocessing pipeline...")

        offers_df = self.preprocess_offers(offers_df)
        transactions_df = self.preprocess_transactions(transactions_df)

        logger.info("Filtering 'offer received' and 'transaction' events...")
        offers_received_df = transactions_df.filter(col("event") == "offer received").select(
            col("account_id").alias("acc_id"),
            col("offer_id").alias("received_offer_id"),
            col("time_since_test_start").alias("received_time")
        )

        transactions_only_df = transactions_df.filter(col("event") == "transaction").select(
            "account_id", "time_since_test_start", "amount"
        )

        logger.info("Matching each transaction to the latest offer received...")
        window_spec = Window.partitionBy("account_id", "time_since_test_start").orderBy(col("received_time").desc())
        joined_df = (
            transactions_only_df
            .join(
                offers_received_df,
                (transactions_only_df["account_id"] == offers_received_df["acc_id"]) &
                (offers_received_df["received_time"] <= transactions_only_df["time_since_test_start"]),
                how="left"
            )
            .withColumn("row_num", row_number().over(window_spec))
        )

        transactions_with_offer_df = joined_df.filter(col("row_num") == 1).drop("row_num", "acc_id")

        logger.info("Enriching transactions with offer metadata...")
        transactions_with_offer_df = transactions_with_offer_df.join(
            offers_df.withColumnRenamed("id", "offer_id_meta"),
            transactions_with_offer_df["received_offer_id"] == col("offer_id_meta"),
            how="left"
        ).drop("offer_id_meta")

        logger.info("Identifying completed offers...")
        offers_completed_df = transactions_df.filter(col("event") == "offer completed")

        offers_completed_with_meta_df = offers_completed_df.join(
            offers_df.withColumnRenamed("id", "offer_id_meta"),
            offers_completed_df["offer_id"] == col("offer_id_meta"),
            how="left"
        ).select(
            offers_completed_df["account_id"],
            offers_completed_df["time_since_test_start"],
            "discount_value",
            "offer_type"
        )

        logger.info("Aggregating discount and offer types per transaction...")
        discounts_by_transaction_df = offers_completed_with_meta_df.groupBy(
            "account_id", "time_since_test_start"
        ).agg(
            max("discount_value").alias("max_discount"),
            collect_set("offer_type").alias("completed_offer_types")
        )

        logger.info("Calculating real amount spent and conversion target...")
        final_df = transactions_only_df.join(
            discounts_by_transaction_df, ["account_id", "time_since_test_start"], how="left"
        ).withColumn(
            "real_amount",
            when(col("completed_offer_types").isNotNull(), col("amount") + col("max_discount")).otherwise(col("amount"))
        ).withColumn(
            "target_converted",
            when(col("completed_offer_types").isNotNull(), 1).otherwise(0)
        )

        logger.info("Processing customer registration date...")
        customers_df = (
            customers_df
            .withColumn("registered_on", to_date("registered_on", "yyyyMMdd"))
            .withColumn("reg_day", dayofmonth("registered_on"))
            .withColumn("reg_month", month("registered_on"))
            .withColumn("reg_year", year("registered_on"))
            .withColumnRenamed("id", "customer_id")
        )

        logger.info("Merging customer information...")
        final_df = final_df.join(
            customers_df,
            final_df["account_id"] == col("customer_id"),
            how="inner"
        ).drop("customer_id")

        logger.info("Imputing missing or invalid values...")
        final_df = (
            final_df
            .withColumn("gender", when(col("gender").isNull(), "unknown").otherwise(col("gender")))
            .withColumn("credit_card_limit", when(col("credit_card_limit").isNull(), -1).otherwise(col("credit_card_limit")))
            .withColumn("age", when(col("age") == 118, -1).otherwise(col("age")))
        )

        logger.info("Formatting offer types as string...")
        final_df = final_df.withColumn(
            "completed_offer_types",
            when(
                col("completed_offer_types").isNotNull() & (size(col("completed_offer_types")) > 0),
                concat_ws(",", col("completed_offer_types"))
            ).otherwise(None)
        )

        logger.info("Adding unique index column...")
        final_df = final_df.withColumn("index", monotonically_increasing_id())

        logger.info("Preprocessing complete.")
        return final_df