"""Run the Spark analytics pipeline against generated parquet data."""

import logging
import time

from src.spark_analytics import SalesAnalytics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Load parquet data, run all sales analyses, and print execution timings."""
    spark = None
    try:
        LOGGER.info("Creating Spark session")
        spark = SalesAnalytics.create_spark_session()

        customers_path = "data/raw/customers.parquet"
        products_path = "data/raw/products.parquet"
        orders_path = "data/raw/orders.parquet"

        LOGGER.info("Loading parquet files")
        customers_df = SalesAnalytics.load_parquet_path(spark, customers_path)
        products_df = SalesAnalytics.load_parquet_path(spark, products_path)
        orders_df = SalesAnalytics.load_parquet_path(spark, orders_path)

        LOGGER.info("Running top customers analysis")
        start = time.perf_counter()
        top_customers_df = SalesAnalytics.top_customers_by_revenue(orders_df, products_df, n=10)
        elapsed = time.perf_counter() - start
        print(f"Top customers by revenue execution time: {elapsed:.4f} seconds")
        top_customers_df.show(truncate=False)

        LOGGER.info("Running sales by category analysis")
        start = time.perf_counter()
        category_sales_df = SalesAnalytics.sales_by_category(orders_df, products_df)
        elapsed = time.perf_counter() - start
        print(f"Sales by category execution time: {elapsed:.4f} seconds")
        category_sales_df.show(truncate=False)

        LOGGER.info("Running monthly trends analysis")
        start = time.perf_counter()
        monthly_trends_df = SalesAnalytics.monthly_trends(orders_df, products_df)
        elapsed = time.perf_counter() - start
        print(f"Monthly trends execution time: {elapsed:.4f} seconds")
        monthly_trends_df.show(truncate=False)

    except FileNotFoundError as exc:
        LOGGER.exception("Required parquet file missing: %s", exc)
        raise
    except Exception as exc:
        LOGGER.exception("Analytics execution failed: %s", exc)
        raise
    finally:
        if spark is not None:
            LOGGER.info("Stopping Spark session")
            spark.stop()


if __name__ == "__main__":
    main()
