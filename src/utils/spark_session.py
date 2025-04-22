from pyspark.sql import SparkSession

def get_spark_session(app_name="MyApp", local=True):
    builder = SparkSession.builder.appName(app_name)
    if local:
        builder = builder.master("local[*]")
    spark = builder.getOrCreate()
    return spark
