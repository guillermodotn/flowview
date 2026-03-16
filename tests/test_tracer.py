"""Tests for flowview.tracer — @trace decorator with proxy wrapping."""

import polars as pl
import pytest

import flowview as fv


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

    def test_multi_step_pipeline(self, capsys):
        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.pipe(_clean).pipe(_filter_positive).pipe(_add_double)

        df = pl.DataFrame({"VALUE": [-1, 0, 1, 2, 3]})
        result = pipeline(df)

        assert result.shape == (3, 2)
        assert list(result.columns) == ["value", "double"]

    def test_filter_now_captured(self, capsys):
        """After proxy migration, direct method calls like .filter()
        are captured as steps (unlike the old monkey-patching approach)."""

        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("value") > 0)

        df = pl.DataFrame({"value": [1, 2, 3]})
        result = pipeline(df)

        assert result.shape == (3, 1)
        # Verify output includes trace (filter step should appear)
        captured = capsys.readouterr()
        assert "filter" in captured.out

    def test_return_value_unchanged(self):
        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.pipe(_add_double)

        df = pl.DataFrame({"value": [10, 20]})
        result = pipeline(df)

        expected = df.with_columns((pl.col("value") * 2).alias("double"))
        assert result.equals(expected)

    def test_result_is_real_dataframe(self):
        """The decorator should always return a real pl.DataFrame,
        never a TracedDataFrame proxy."""

        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("value") > 0)

        df = pl.DataFrame({"value": [1, 2, 3]})
        result = pipeline(df)

        assert type(result) is pl.DataFrame

    def test_method_chain_tracing(self, capsys):
        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return (
                df.filter(pl.col("value") > 0)
                .with_columns((pl.col("value") * 2).alias("double"))
                .sort("double", descending=True)
            )

        df = pl.DataFrame({"value": [-1, 0, 1, 2, 3]})
        result = pipeline(df)

        assert result.shape == (3, 2)
        assert result["double"].to_list() == [6, 4, 2]

        captured = capsys.readouterr()
        assert "filter" in captured.out
        assert "with_columns" in captured.out
        assert "sort" in captured.out

    def test_mixed_pipe_and_method_chain(self, capsys):
        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("value") > 0).pipe(_add_double).sort("double")

        df = pl.DataFrame({"value": [-1, 0, 1, 2, 3]})
        result = pipeline(df)

        assert result.shape == (3, 2)

        captured = capsys.readouterr()
        assert "filter" in captured.out
        assert "_add_double" in captured.out
        assert "sort" in captured.out

    def test_error_propagates(self):
        @fv.trace
        def pipeline(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("nonexistent") > 0)

        df = pl.DataFrame({"value": [1, 2, 3]})
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            pipeline(df)

    def test_df_as_keyword_arg(self, capsys):
        @fv.trace
        def pipeline(*, data: pl.DataFrame) -> pl.DataFrame:
            return data.filter(pl.col("value") > 0)

        df = pl.DataFrame({"value": [1, 2, 3]})
        result = pipeline(data=df)

        assert type(result) is pl.DataFrame
        assert result.shape == (3, 1)

    def test_multiple_dataframes_only_first_traced(self):
        """Only the first DataFrame argument should be wrapped in a proxy."""

        @fv.trace
        def pipeline(df: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
            return df.join(other, on="id", how="left")

        df = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        other = pl.DataFrame({"id": [1, 2], "label": ["a", "b"]})
        result = pipeline(df, other)

        assert type(result) is pl.DataFrame
        assert result.shape[0] == 3
        assert "label" in result.columns
