from pyspark.sql import SparkSession
from fraudlib.synth import make_transactions

spark = SparkSession.builder.getOrCreate()

df = make_transactions()

spark_df = spark.createDataFrame(df)

spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("fraud_dev_core.transactions_raw")