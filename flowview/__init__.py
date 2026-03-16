"""flowview — Visual data pipeline debugger for Polars.

Stop print-debugging your pipelines.

Usage:
    import flowview as fv

    # Method chains (idiomatic Polars):
    @fv.trace
    def process(df):
        return (
            df.filter(pl.col("status") == "active")
              .with_columns(...)
              .group_by("category")
              .agg(...)
        )

    # .pipe() chains (also supported):
    @fv.trace
    def process(df):
        return df.pipe(clean).pipe(transform).pipe(aggregate)

    # With options:
    @fv.trace(sample_rows=3, show_schema=True)
    def process(df):
        ...
"""

from flowview._version import __version__
from flowview.tracer import trace

__all__ = ["__version__", "trace"]
