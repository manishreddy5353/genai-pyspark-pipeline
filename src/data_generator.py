"""Generate synthetic e-commerce data as pandas DataFrames.

The order generator uses a Pareto-based customer allocation: approximately 20% of
customers receive 80% of the orders, emulating a common e-commerce pattern.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Final
from uuid import uuid4

import numpy as np
import pandas as pd
from faker import Faker
from tqdm.auto import tqdm


LOGGER = logging.getLogger(__name__)
DEFAULT_CATEGORIES: Final[tuple[str, ...]] = (
    "Electronics", "Clothing", "Home", "Sports", "Books"
)


class IterableDataFrame(pd.DataFrame):
    """Pandas DataFrame that iterates over row dictionaries for compatibility."""

    @property
    def _constructor(self) -> type["IterableDataFrame"]:
        return IterableDataFrame

    def __iter__(self):
        for _, row in self.iterrows():
            yield row.to_dict()


class SyntheticDataGenerator:
    """Generate realistic synthetic customers, products, and orders.

    Args:
        seed: Optional seed for reproducible NumPy and Faker output.
        locale: Faker locale used for names, addresses, and emails.
    """

    def __init__(self, seed: int | None = 42, locale: str = "en_US") -> None:
        """Initialize the random-number generators used for data creation."""
        self._rng = np.random.default_rng(seed)
        self._faker = Faker(locale)
        self._faker.seed_instance(seed)

    def generate_customers(self, count: int = 100_000) -> pd.DataFrame:
        """Generate customers with normally distributed ages around 35.

        Args:
            count: Number of customers to generate.

        Returns:
            Customer identifiers, demographics, and registration dates.
        """
        self._validate_count(count, "customer")
        LOGGER.info("Generating %s customers", count)
        today = date.today()
        registration_offsets = self._rng.integers(0, 1_826, size=count)
        ages = np.clip(np.rint(self._rng.normal(35, 10, size=count)), 18, 90).astype(int)
        records = [
            {
                "customer_id": str(uuid4()),
                "name": self._faker.name(),
                "email": self._faker.unique.email(),
                "age": int(ages[index]),
                "city": self._faker.city(),
                "country": self._faker.country(),
                "registration_date": (today - timedelta(days=int(registration_offsets[index]))).isoformat(),
            }
            for index in tqdm(range(count), desc="Generating customers", unit="customer")
        ]
        customers = IterableDataFrame.from_records(records)
        LOGGER.info("Generated %s customer records", len(customers))
        return customers

    def generate_products(self, count: int = 10_000) -> pd.DataFrame:
        """Generate products with prices, stock levels, and ratings.

        Args:
            count: Number of products to generate.

        Returns:
            Product identifiers, names, category, price, stock, and rating.
        """
        self._validate_count(count, "product")
        LOGGER.info("Generating %s products", count)
        records = [
            {
                "product_id": str(uuid4()),
                "name": self._faker.catch_phrase(),
                "category": str(self._rng.choice(DEFAULT_CATEGORIES)),
                "price": round(float(self._rng.uniform(10, 500)), 2),
                "stock": int(self._rng.integers(0, 1_001)),
                "rating": round(float(self._rng.uniform(1, 5)), 1),
            }
            for _ in tqdm(range(count), desc="Generating products", unit="product")
        ]
        products = IterableDataFrame.from_records(records)
        LOGGER.info("Generated %s product records", len(products))
        return products

    def generate_orders(
        self, customers: pd.DataFrame, products: pd.DataFrame, count: int = 1_000_000
    ) -> pd.DataFrame:
        """Generate orders with Pareto-weighted 80/20 customer allocation.

        The top 20% of customers collectively receive 80% of orders. Pareto
        weights control the allocation to individual customers in each group.

        Args:
            customers: Non-empty DataFrame containing ``customer_id``.
            products: Non-empty DataFrame containing ``product_id``.
            count: Number of orders to generate.

        Returns:
            Order identifiers, customer/product links, quantity, and order date.
        """
        self._validate_count(count, "order")
        self._validate_source_data(customers, "customer_id", "customers")
        self._validate_source_data(products, "product_id", "products")
        LOGGER.info("Generating %s orders using Pareto customer allocation", count)

        customer_ids = customers["customer_id"].to_numpy()
        product_ids = products["product_id"].to_numpy()
        high_value_count = max(1, int(len(customer_ids) * 0.20))
        high_value_indices = self._rng.choice(len(customer_ids), high_value_count, replace=False)
        high_value_mask = np.zeros(len(customer_ids), dtype=bool)
        high_value_mask[high_value_indices] = True
        high_value_orders = int(count * 0.80)
        selected_customers = np.concatenate(
            (
                self._sample_pareto_weighted(customer_ids[high_value_mask], high_value_orders),
                self._sample_pareto_weighted(customer_ids[~high_value_mask], count - high_value_orders),
            )
        )
        self._rng.shuffle(selected_customers)
        start_date = np.datetime64(date.today() - timedelta(days=365))
        order_dates = start_date + self._rng.integers(0, 366, count).astype("timedelta64[D]")
        orders = IterableDataFrame(
            {
                "order_id": [str(uuid4()) for _ in tqdm(range(count), desc="Creating order IDs", unit="order")],
                "customer_id": selected_customers,
                "product_id": self._rng.choice(product_ids, size=count),
                "quantity": self._rng.integers(1, 11, size=count),
                "order_date": order_dates.astype("datetime64[D]").astype(str),
            }
        )
        LOGGER.info("Generated %s order records", len(orders))
        return orders

    def generate_all(
        self,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        order_count: int = 1_000_000,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and return the complete default e-commerce dataset.

        Returns:
            A tuple in ``(customers, products, orders)`` order.
        """
        customers = self.generate_customers(customer_count)
        products = self.generate_products(product_count)
        orders = self.generate_orders(customers, products, order_count)
        return customers, products, orders

    def _sample_pareto_weighted(self, customer_ids: np.ndarray, count: int) -> np.ndarray:
        """Sample IDs according to normalized Pareto-distributed probabilities."""
        if count == 0:
            return np.array([], dtype=customer_ids.dtype)
        weights = self._rng.pareto(a=1.5, size=len(customer_ids)) + 1
        return self._rng.choice(customer_ids, size=count, p=weights / weights.sum())

    @staticmethod
    def _validate_count(count: int, record_type: str) -> None:
        """Raise an error when a requested record count is not positive."""
        if count <= 0:
            raise ValueError(f"{record_type.capitalize()} count must be greater than zero.")

    @staticmethod
    def _validate_source_data(dataframe: pd.DataFrame, identifier: str, source_name: str) -> None:
        """Validate required identifier columns in non-empty source DataFrames."""
        if dataframe.empty or identifier not in dataframe.columns:
            raise ValueError(f"{source_name.capitalize()} must be non-empty and include '{identifier}'.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    generator = SyntheticDataGenerator()
    customers_frame, products_frame, orders_frame = generator.generate_all()
    LOGGER.info("Finished: %s customers, %s products, %s orders", len(customers_frame), len(products_frame), len(orders_frame))
