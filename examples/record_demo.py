"""Record the flowview demo as an SVG for the README."""

import polars as pl
from rich.console import Console

import flowview as fv
from flowview import renderer

# Replace the module console with a recording one
recording_console = Console(record=True, width=90)
renderer.console = recording_console


# --- Traced pipeline ---


@fv.trace(sample_rows=3)
def process_sales(df: pl.DataFrame) -> pl.DataFrame:
    """Sales pipeline using method chains."""
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


# --- Run and record ---

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

# Export SVG
svg = recording_console.export_svg(title="flowview")

with open("assets/demo.svg", "w") as f:
    f.write(svg)

print("Saved to assets/demo.svg")
