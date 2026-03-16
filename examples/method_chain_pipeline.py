"""Method chain example: tracing idiomatic Polars code with flowview."""

import polars as pl

import flowview as fv


@fv.trace
def process_sales(df: pl.DataFrame) -> pl.DataFrame:
    """Sales pipeline using direct method chains — no .pipe() needed."""
    return (
        df.filter(pl.col("status") == "active")
        .with_columns(
            (pl.col("price") * pl.col("quantity")).alias("revenue"),
            pl.col("category").str.to_lowercase().alias("category"),
        )
        .group_by("category")
        .agg(
            pl.col("revenue").sum().alias("total_revenue"),
            pl.len().alias("order_count"),
        )
        .sort("total_revenue", descending=True)
    )


def main():
    df = pl.DataFrame(
        {
            "status": [
                "active",
                "active",
                "inactive",
                "active",
                "active",
                "active",
                "inactive",
                "active",
                "active",
                "cancelled",
            ]
            * 100,
            "category": [
                "Electronics",
                "Books",
                "Electronics",
                "Clothing",
                "Books",
                "Electronics",
                "Clothing",
                "Books",
                "Electronics",
                "Books",
            ]
            * 100,
            "price": [
                299.99,
                14.99,
                599.99,
                49.99,
                29.99,
                149.99,
                79.99,
                19.99,
                399.99,
                9.99,
            ]
            * 100,
            "quantity": [2, 5, 1, 3, 4, 1, 2, 6, 1, 3] * 100,
        }
    )

    result = process_sales(df)
    print("\nFinal result:")
    print(result)


if __name__ == "__main__":
    main()
