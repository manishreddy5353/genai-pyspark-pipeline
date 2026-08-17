"""Run synthetic e-commerce data generation and save the results as Parquet."""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src import config
from src.data_generator import SyntheticDataGenerator as SyntheticataGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def save_parquet(dataframe: pd.DataFrame, file_name: str) -> Path:
    """Save a DataFrame to the configured raw-data directory as a Parquet file.

    Args:
        dataframe: Data to persist.
        file_name: Target Parquet file name.

    Returns:
        The full path of the saved Parquet file.
    """
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RAW_DATA_DIR / file_name
    dataframe.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path


def format_file_size(file_path: Path) -> str:
    """Return a human-readable size for a file.

    Args:
        file_path: File whose size should be formatted.

    Returns:
        A size string expressed in B, KB, MB, or GB.
    """
    size = float(file_path.stat().st_size)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def main() -> None:
    """Generate default datasets, persist them, and print timing and file sizes."""
    start_time = time.perf_counter()
    try:
        LOGGER.info("Starting synthetic e-commerce data generation")
        generator = SyntheticataGenerator(seed=config.RANDOM_SEED)
        customers, products, orders = generator.generate_all(
            customer_count=config.DEFAULT_CUSTOMER_COUNT,
            product_count=config.DEFAULT_PRODUCT_COUNT,
            order_count=config.DEFAULT_ORDER_COUNT,
        )

        output_files = {
            "customers": save_parquet(customers, "customers.parquet"),
            "products": save_parquet(products, "products.parquet"),
            "orders": save_parquet(orders, "orders.parquet"),
        }
        elapsed_seconds = time.perf_counter() - start_time

        print(f"Generation completed in {elapsed_seconds:.2f} seconds.")
        for dataset_name, output_path in output_files.items():
            print(f"{dataset_name.title()}: {output_path} ({format_file_size(output_path)})")
    except (ImportError, OSError, ValueError) as error:
        LOGGER.exception("Data generation failed: %s", error)
        print(f"Data generation failed: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        LOGGER.exception("Unexpected data-generation failure: %s", error)
        print(f"Unexpected data generation failure: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
