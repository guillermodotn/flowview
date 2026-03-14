"""flowview — Visual data pipeline debugger for Polars.

Stop print-debugging your pipelines.

Usage:
    import flowview as fv

    @fv.trace
    def process(df):
        return (
            df.pipe(clean)
              .pipe(filter_active)
              .pipe(add_revenue)
        )

    # With options:
    @fv.trace(sample_rows=3, show_schema=True)
    def process(df):
        ...
"""

from flowview._version import __version__
from flowview.tracer import trace

__all__ = ["__version__", "trace"]
