from pyspark.sql import functions as F
from pyspark.sql.window import Window



def build_feature_spark(df):
    window = (
        Window
        .partitionBy("customer_id")
        .orderBy(F.col("ts").cast("long"))
        .rangeBetween(-7 * 86400, -1)
    )

    spark_df = (
        df
        .withColumn("amount_mean_7d", F.mean("amount").over(window))
        .withColumn("amount_std_7d", F.stddev("amount").over(window))
        .withColumn("amount_max_7d", F.max("amount").over(window))
        .withColumn("amount_min_7d", F.min("amount").over(window))
        .withColumn("amount_count_7d", F.count("amount").over(window))
    )

    return spark_df


# spark_df.write \
#     .format("delta") \
#     .mode("overwrite") \
#     .saveAsTable("fraud_dev.fraud_dev_core.transactions_raw")