"""Basic example: tracing a Polars pipeline with flowview."""

import polars as pl

import flowview as fv


def clean_names(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize column names to lowercase."""
    return df.rename({col: col.lower().replace(" ", "_") for col in df.columns})


def filter_active(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only active records."""
    return df.filter(pl.col("status") == "active")


def add_revenue(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate revenue from price and quantity."""
    return df.with_columns((pl.col("price") * pl.col("quantity")).alias("revenue"))


def top_categories(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate revenue by category."""
    return df.group_by("category").agg(
        pl.col("revenue").sum().alias("total_revenue"),
        pl.col("revenue").count().alias("order_count"),
    )


@fv.trace
def process_sales(df: pl.DataFrame) -> pl.DataFrame:
    """Full sales processing pipeline."""
    return (
        df.pipe(clean_names).pipe(filter_active).pipe(add_revenue).pipe(top_categories)
    )


def main():
    # Create sample data
    df = pl.DataFrame(
        {
            "Category": ["Electronics", "Books", "Electronics", "Clothing", "Books"]
            * 200,
            "Status": ["active", "active", "inactive", "active", "active"] * 200,
            "Price": [299.99, 14.99, 599.99, 49.99, 29.99] * 200,
            "Quantity": [2, 5, 1, 3, 4] * 200,
        }
    )

    result = process_sales(df)
    print("\nFinal result:")
    print(result)


if __name__ == "__main__":
    main()
