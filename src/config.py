"""Central configuration for paths and default pipeline settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_CUSTOMER_COUNT = 100_000
DEFAULT_PRODUCT_COUNT = 10_000
DEFAULT_ORDER_COUNT = 1_000_000
RANDOM_SEED = 42

CATEGORIES = ("Electronics", "Home & Kitchen", "Books", "Clothing", "Sports")
