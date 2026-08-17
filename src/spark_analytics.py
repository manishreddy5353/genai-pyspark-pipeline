"""PySpark analytics for synthetic e-commerce sales data."""

import logging
from pathlib import Path

from pyspark import SparkConf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Build sales insights from customer, product, and order parquet data."""

    @staticmethod
    def create_spark_session() -> SparkSession:
        """Create and configure a local Spark session with memory and AQE tuning."""
        config = (
            SparkConf()
            .setAppName("ecommerce-sales-analytics")
            .setMaster("local[*]")
            .set("spark.driver.memory", "4g")
            .set("spark.executor.memory", "4g")
            .set("spark.sql.adaptive.enabled", "true")
            .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryo.registrationRequired", "false")
        )
        return SparkSession.builder.config(conf=config).getOrCreate()

    @staticmethod
    def load_parquet_path(spark: SparkSession, path: str) -> DataFrame:
        """Read a parquet dataset from disk into a Spark DataFrame."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Parquet path not found: {path}")
        return spark.read.parquet(path)

    @staticmethod
    def top_customers_by_revenue(orders_df: DataFrame, products_df: DataFrame, n: int = 10) -> DataFrame:
        """Join orders with products, calculate spend per customer, and return the top N."""
        joined = orders_df.join(products_df.select("product_id", "category", "price"), on="product_id", how="inner")
        revenue_by_customer = joined.withColumn(
            "line_revenue", F.col("quantity") * F.col("price")
        ).groupBy("customer_id").agg(F.round(F.sum("line_revenue"), 2).alias("total_spend"))
        return revenue_by_customer.orderBy(F.desc("total_spend")).limit(n)

    @staticmethod
    def sales_by_category(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Group revenue and units sold by product category."""
        joined = orders_df.join(products_df.select("product_id", "category", "price"), on="product_id", how="inner")
        sales = joined.withColumn(
            "line_revenue", F.col("quantity") * F.col("price")
        ).groupBy("category").agg(
            F.round(F.sum("line_revenue"), 2).alias("revenue"),
            F.sum("quantity").alias("units_sold"),
        )
        return sales.orderBy(F.desc("revenue"))

    @staticmethod
    def monthly_trends(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Calculate month-over-month revenue growth percentage per month."""
        joined = orders_df.join(products_df.select("product_id", "category", "price"), on="product_id", how="inner")
        monthly_revenue = joined.withColumn(
            "line_revenue", F.col("quantity") * F.col("price")
        ).withColumn("month", F.date_format(F.to_date("order_date"), "yyyy-MM"))
        monthly_revenue = monthly_revenue.groupBy("month").agg(
            F.round(F.sum("line_revenue"), 2).alias("monthly_revenue")
        ).orderBy("month")

        window_spec = Window.orderBy("month")
        return monthly_revenue.withColumn(
            "previous_month_revenue",
            F.lag("monthly_revenue").over(window_spec),
        ).withColumn(
            "growth_pct",
            F.when(
                F.col("previous_month_revenue").isNull() | (F.col("previous_month_revenue") == 0),
                F.lit(None),
            ).otherwise(
                ((F.col("monthly_revenue") - F.col("previous_month_revenue")) / F.col("previous_month_revenue")) * 100
            ),
        ).select(
            "month",
            "monthly_revenue",
            F.round(F.col("growth_pct"), 4).alias("growth_pct"),
        )


if __name__ == "__main__":
    spark = SalesAnalytics.create_spark_session()
    try:
        orders_df = SalesAnalytics.load_parquet_path(spark, "data/raw/orders.parquet")
        products_df = SalesAnalytics.load_parquet_path(spark, "data/raw/products.parquet")
        LOGGER.info("Top customers by revenue:\n%s", SalesAnalytics.top_customers_by_revenue(orders_df, products_df, 10).show())
        LOGGER.info("Sales by category:\n%s", SalesAnalytics.sales_by_category(orders_df, products_df).show())
        LOGGER.info("Monthly trends:\n%s", SalesAnalytics.monthly_trends(orders_df, products_df).show())
    finally:
        spark.stop()
