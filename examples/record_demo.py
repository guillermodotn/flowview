"""Record the flowview demo as an SVG for the README."""

import polars as pl
from rich.console import Console

import flowview as fv
from flowview import renderer

# Replace the module console with a recording one
recording_console = Console(record=True, width=90)
renderer.console = recording_console


# --- Pipeline steps ---


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
    return (
        df.group_by("category")
        .agg(
            pl.col("revenue").sum().alias("total_revenue"),
            pl.len().alias("order_count"),
        )
        .sort("total_revenue", descending=True)
    )


# --- Traced pipeline ---


@fv.trace(sample_rows=3)
def process_sales(df: pl.DataFrame) -> pl.DataFrame:
    """Full sales processing pipeline."""
    return (
        df.pipe(clean_names).pipe(filter_active).pipe(add_revenue).pipe(top_categories)
    )


# --- Run and record ---

df = pl.DataFrame(
    {
        "Category": [
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
        "Status": [
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
        "Price": [
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
        "Quantity": [2, 5, 1, 3, 4, 1, 2, 6, 1, 3] * 100,
    }
)

result = process_sales(df)

# Export SVG
svg = recording_console.export_svg(title="flowview")

with open("assets/demo.svg", "w") as f:
    f.write(svg)

print("Saved to assets/demo.svg")
