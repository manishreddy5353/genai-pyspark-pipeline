"""Tests for fake e-commerce data generation."""

from src.data_generator import SyntheticDataGenerator


def test_generated_orders_reference_source_records() -> None:
    """Generated orders should reference IDs from provided customers and products."""
    generator = SyntheticDataGenerator(seed=1)
    customers = generator.generate_customers(2)
    products = generator.generate_products(2)
    orders = generator.generate_orders(customers, products, 5)

    customer_ids = {customer["customer_id"] for customer in customers}
    product_ids = {product["product_id"] for product in products}
    assert len(orders) == 5
    assert all(order["customer_id"] in customer_ids for order in orders)
    assert all(order["product_id"] in product_ids for order in orders)
