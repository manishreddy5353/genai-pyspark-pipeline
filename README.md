# E-commerce Data Pipeline

This project generates realistic fake e-commerce data and uses PySpark to produce business-focused analytical datasets.

## Features

- Generates customers, products, and orders as CSV files.
- Produces revenue by category, top products, and customer spending summaries.
- Uses type hints, docstrings, and application logging throughout.

## Project structure

```text
genai-pyspark-pipeline/
├── data/
│   ├── raw/                 # Generated source CSV files
│   └── processed/           # Spark analytical results
├── notebooks/               # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/
│   └── test_data_generator.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

From the project root, generate the sample data:

```bash
python -m src.data_generator --customers 100 --products 50 --orders 500
```

Then run the Spark analysis:

```bash
python -m src.spark_analytics
```

Generated files are stored in `data/raw/`; analytical CSV datasets are stored in `data/processed/`.

## Test

```bash
pytest
```
