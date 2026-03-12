"""Realistic example: e-commerce analytics pipeline with ~1M rows."""

import random
from datetime import date, timedelta

import polars as pl

import flowview as fv

random.seed(42)

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def clean_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize column names and trim string fields."""
    renamed = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})
    str_cols = [
        col for col, dtype in zip(renamed.columns, renamed.dtypes) if dtype == pl.String
    ]
    return renamed.with_columns([pl.col(c).str.strip_chars() for c in str_cols])


def remove_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """Drop duplicate order rows."""
    return df.unique(subset=["order_id"])


def filter_valid_orders(df: pl.DataFrame) -> pl.DataFrame:
    """Remove cancelled orders, zero-quantity, and negative prices."""
    return df.filter(
        (pl.col("status") != "cancelled")
        & (pl.col("quantity") > 0)
        & (pl.col("unit_price") > 0)
    )


def add_financials(df: pl.DataFrame) -> pl.DataFrame:
    """Compute revenue, tax, and net revenue."""
    return df.with_columns(
        (pl.col("unit_price") * pl.col("quantity")).alias("gross_revenue"),
        (pl.col("unit_price") * pl.col("quantity") * pl.col("tax_rate")).alias("tax"),
    ).with_columns(
        (pl.col("gross_revenue") - pl.col("tax")).alias("net_revenue"),
    )


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Extract date parts for analysis."""
    return df.with_columns(
        pl.col("order_date").dt.year().alias("year"),
        pl.col("order_date").dt.month().alias("month"),
        pl.col("order_date").dt.weekday().alias("day_of_week"),
        pl.col("order_date").dt.quarter().alias("quarter"),
    )


def flag_high_value(df: pl.DataFrame) -> pl.DataFrame:
    """Flag orders above the 90th percentile revenue."""
    p90 = df["gross_revenue"].quantile(0.9)
    return df.with_columns(
        (pl.col("gross_revenue") > p90).alias("is_high_value"),
    )


def aggregate_by_category(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize metrics per product category and quarter."""
    return (
        df.group_by("category", "quarter")
        .agg(
            pl.col("gross_revenue").sum().alias("total_revenue"),
            pl.col("net_revenue").sum().alias("total_net_revenue"),
            pl.col("tax").sum().alias("total_tax"),
            pl.col("order_id").n_unique().alias("unique_orders"),
            pl.col("quantity").sum().alias("units_sold"),
            pl.col("is_high_value").sum().alias("high_value_orders"),
            pl.col("gross_revenue").mean().alias("avg_order_value"),
        )
        .sort("category", "quarter")
    )


# ---------------------------------------------------------------------------
# Traced pipeline
# ---------------------------------------------------------------------------


@fv.trace(sample_rows=5)
def ecommerce_analytics(df: pl.DataFrame) -> pl.DataFrame:
    """Full e-commerce analytics pipeline."""
    return (
        df.pipe(clean_columns)
        .pipe(remove_duplicates)
        .pipe(filter_valid_orders)
        .pipe(add_financials)
        .pipe(add_time_features)
        .pipe(flag_high_value)
        .pipe(aggregate_by_category)
    )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

NUM_ROWS = 1_000_000

CATEGORIES = [
    "Electronics",
    "Clothing",
    "Home & Garden",
    "Books",
    "Sports",
    "Toys",
    "Food & Drink",
    "Health",
]
REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America"]
STATUSES = [
    "completed",
    "completed",
    "completed",
    "completed",
    "shipped",
    "shipped",
    "returned",
    "cancelled",
]
CHANNELS = ["web", "web", "mobile", "mobile", "in-store", "marketplace"]


def generate_data(n: int = NUM_ROWS) -> pl.DataFrame:
    """Generate a realistic-ish e-commerce dataset."""
    base_date = date(2023, 1, 1)

    return pl.DataFrame(
        {
            "Order ID": list(range(1, n + 1)),
            "Order Date": [
                base_date + timedelta(days=random.randint(0, 729)) for _ in range(n)
            ],
            "Category": [random.choice(CATEGORIES) for _ in range(n)],
            "Region": [random.choice(REGIONS) for _ in range(n)],
            "Channel": [random.choice(CHANNELS) for _ in range(n)],
            "Status": [random.choice(STATUSES) for _ in range(n)],
            "Quantity": [
                random.randint(0, 20) for _ in range(n)
            ],  # some zeros to be filtered
            "Unit Price": [
                round(random.uniform(-5, 500), 2) for _ in range(n)
            ],  # some negatives to be filtered
            "Tax Rate": [round(random.uniform(0.05, 0.25), 4) for _ in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Generating {NUM_ROWS:,} rows of e-commerce data...\n")
    df = generate_data()

    result = ecommerce_analytics(df)

    print("\nFinal result:")
    print(result)


if __name__ == "__main__":
    main()
