"""Record the flowview demo as an SVG for the README."""

import polars as pl
from rich.console import Console

import flowview as fv
from flowview import renderer

# Use a recording console for SVG export
recording_console = Console(record=True, width=80)
renderer.console = recording_console


@fv.trace(sample_rows=2)
def analyze_orders(df: pl.DataFrame) -> pl.DataFrame:
    """Order analytics pipeline."""
    return (
        df.filter(pl.col("status") == "completed")
        .with_columns(
            (pl.col("price") * pl.col("qty")).alias("revenue"),
        )
        .group_by("region")
        .agg(
            pl.col("revenue").sum().alias("total_revenue"),
            pl.len().alias("orders"),
        )
        .sort("total_revenue", descending=True)
    )


df = pl.DataFrame(
    {
        "order_id": list(range(1, 501)),
        "region": (
            ["North America", "Europe", "Asia", "Europe", "North America"] * 100
        ),
        "status": (
            ["completed", "completed", "completed", "cancelled", "completed"] * 100
        ),
        "price": ([49.99, 29.99, 89.99, 19.99, 149.99] * 100),
        "qty": ([2, 1, 3, 1, 1] * 100),
    }
)

result = analyze_orders(df)

# Export SVG
svg = recording_console.export_svg(title="flowview")

with open("assets/demo.svg", "w") as f:
    f.write(svg)

print("Saved to assets/demo.svg")
