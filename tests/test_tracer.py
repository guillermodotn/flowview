"""Tests for flowview.tracer."""

import contextlib

import polars as pl

import flowview as fv
from flowview.tracer import _original_pipe


def _clean(df: pl.DataFrame) -> pl.DataFrame:
    """Dummy step: rename columns to lowercase."""
    return df.rename({col: col.lower() for col in df.columns})


def _filter_positive(df: pl.DataFrame) -> pl.DataFrame:
    """Dummy step: filter rows where value > 0."""
    return df.filter(pl.col("value") > 0)


def _add_double(df: pl.DataFrame) -> pl.DataFrame:
    """Dummy step: add a 'double' column."""
    return df.with_columns((pl.col("value") * 2).alias("double"))


class TestTraceDecorator:
    def test_basic_trace(self, capsys):
        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.pipe(_filter_positive).pipe(_add_double)

        df = pl.DataFrame({"value": [-1, 0, 1, 2, 3]})
        result = pipeline(df)

        # The pipeline still works correctly
        assert result.shape == (3, 2)
        assert "double" in result.columns
        assert result["double"].to_list() == [2, 4, 6]

    def test_trace_with_options(self, capsys):
        @fv.trace(sample_rows=2, show_schema=True)
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.pipe(_add_double)

        df = pl.DataFrame({"value": [10, 20, 30]})
        result = pipeline(df)

        assert result.shape == (3, 2)

    def test_pipe_restored_after_trace(self):
        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.pipe(_add_double)

        df = pl.DataFrame({"value": [1, 2]})
        pipeline(df)

        # pipe should be restored to the original
        assert pl.DataFrame.pipe is _original_pipe

    def test_pipe_restored_on_error(self):
        def bad_step(df: pl.DataFrame) -> pl.DataFrame:
            raise ValueError("intentional error")

        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.pipe(bad_step)

        df = pl.DataFrame({"value": [1]})
        with contextlib.suppress(ValueError):
            pipeline(df)

        # pipe should still be restored
        assert pl.DataFrame.pipe is _original_pipe

    def test_multi_step_pipeline(self, capsys):
        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.pipe(_clean).pipe(_filter_positive).pipe(_add_double)

        df = pl.DataFrame({"VALUE": [-1, 0, 1, 2, 3]})
        result = pipeline(df)

        assert result.shape == (3, 2)
        assert list(result.columns) == ["value", "double"]

    def test_no_pipe_steps(self, capsys):
        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("value") > 0)

        df = pl.DataFrame({"value": [1, 2, 3]})
        result = pipeline(df)

        # Should still work, just no steps captured
        assert result.shape == (3, 1)

    def test_return_value_unchanged(self):
        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.pipe(_add_double)

        df = pl.DataFrame({"value": [10, 20]})
        result = pipeline(df)

        expected = df.with_columns((pl.col("value") * 2).alias("double"))
        assert result.equals(expected)
