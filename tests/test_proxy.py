"""Tests for flowview.proxy — TracedDataFrame proxy wrapper."""

import polars as pl
import pytest

from flowview.models import PipelineTrace
from flowview.proxy import TracedDataFrame, unwrap

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "status": ["active", "inactive", "active", "active", "cancelled"],
            "category": ["A", "B", "A", "B", "A"],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "quantity": [2, 1, 3, 4, 5],
            "id": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture()
def traced(sample_df: pl.DataFrame) -> TracedDataFrame:
    trace = PipelineTrace(function_name="test")
    return TracedDataFrame(sample_df, trace, sample_rows=3)


def _get_trace(proxy: TracedDataFrame) -> PipelineTrace:
    return object.__getattribute__(proxy, "_trace")


# ------------------------------------------------------------------
# Basic delegation
# ------------------------------------------------------------------


class TestDelegation:
    def test_shape_delegates(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        assert traced.shape == sample_df.shape

    def test_columns_delegates(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        assert traced.columns == sample_df.columns

    def test_dtypes_delegates(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        assert traced.dtypes == sample_df.dtypes

    def test_schema_delegates(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        assert traced.schema == sample_df.schema

    def test_height_delegates(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        assert traced.height == sample_df.height

    def test_width_delegates(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        assert traced.width == sample_df.width


# ------------------------------------------------------------------
# Type transparency
# ------------------------------------------------------------------


class TestTypeTransparency:
    def test_isinstance_returns_true(self, traced: TracedDataFrame):
        assert isinstance(traced, pl.DataFrame)

    def test_type_returns_traced_dataframe(self, traced: TracedDataFrame):
        assert type(traced) is TracedDataFrame

    def test_class_property(self, traced: TracedDataFrame):
        assert traced.__class__ is pl.DataFrame


# ------------------------------------------------------------------
# Dunder methods
# ------------------------------------------------------------------


class TestDunderMethods:
    def test_repr(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        assert repr(traced) == repr(sample_df)

    def test_str(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        assert str(traced) == str(sample_df)

    def test_len(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        assert len(traced) == len(sample_df)

    def test_contains(self, traced: TracedDataFrame):
        assert "status" in traced
        assert "nonexistent" not in traced

    def test_getitem_column(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        result = traced["status"]
        expected = sample_df["status"]
        assert result.to_list() == expected.to_list()

    def test_getitem_slice(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        result = traced[:2]
        expected = sample_df[:2]
        assert result.shape == expected.shape


# ------------------------------------------------------------------
# unwrap()
# ------------------------------------------------------------------


class TestUnwrap:
    def test_unwrap_proxy(self, traced: TracedDataFrame, sample_df: pl.DataFrame):
        result = unwrap(traced)
        assert type(result) is pl.DataFrame
        assert result.shape == sample_df.shape

    def test_unwrap_real_dataframe(self, sample_df: pl.DataFrame):
        result = unwrap(sample_df)
        assert result is sample_df

    def test_unwrap_non_dataframe(self):
        assert unwrap(42) == 42
        assert unwrap("hello") == "hello"
        assert unwrap(None) is None


# ------------------------------------------------------------------
# Method tracing — filter
# ------------------------------------------------------------------


class TestTraceFilter:
    def test_filter_captures_step(self, traced: TracedDataFrame):
        result = traced.filter(pl.col("status") == "active")
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.step_name == "filter"
        assert step.row_count == 3
        assert step.execution_time_ms >= 0

    def test_filter_row_diff(self, traced: TracedDataFrame):
        trace = _get_trace(traced)
        trace.input_snapshot = type(trace.input_snapshot)  # clear
        # Set up input snapshot for diffing
        from flowview.collector import capture_snapshot

        trace.input_snapshot = capture_snapshot(
            object.__getattribute__(traced, "_df"),
            step_name="input",
            sample_rows=3,
        )

        result = traced.filter(pl.col("status") == "active")
        step = _get_trace(result).steps[0]
        assert step.row_diff == -2  # 5 -> 3

    def test_filter_returns_proxy(self, traced: TracedDataFrame):
        result = traced.filter(pl.col("status") == "active")
        assert type(result) is TracedDataFrame
        assert isinstance(result, pl.DataFrame)


# ------------------------------------------------------------------
# Method tracing — with_columns
# ------------------------------------------------------------------


class TestTraceWithColumns:
    def test_with_columns_captures_step(self, traced: TracedDataFrame):
        result = traced.with_columns(
            (pl.col("price") * pl.col("quantity")).alias("revenue")
        )
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.step_name == "with_columns"
        assert step.col_count == 6  # original 5 + revenue

    def test_with_columns_schema_diff(self, traced: TracedDataFrame):
        trace = _get_trace(traced)
        from flowview.collector import capture_snapshot

        trace.input_snapshot = capture_snapshot(
            object.__getattribute__(traced, "_df"),
            step_name="input",
            sample_rows=3,
        )

        result = traced.with_columns(
            (pl.col("price") * pl.col("quantity")).alias("revenue")
        )
        step = _get_trace(result).steps[0]
        assert step.schema_diff is not None
        assert "revenue" in step.schema_diff.added

    def test_with_columns_named_exprs(self, traced: TracedDataFrame):
        result = traced.with_columns(doubled=pl.col("price") * 2)
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert "doubled" in step.schema

    def test_with_columns_multiple_exprs(self, traced: TracedDataFrame):
        result = traced.with_columns(
            (pl.col("price") * pl.col("quantity")).alias("revenue"),
            pl.col("price").cast(pl.Int64).alias("price_int"),
        )
        trace = _get_trace(result)
        step = trace.steps[0]
        assert step.col_count == 7  # 5 + revenue + price_int


# ------------------------------------------------------------------
# Method tracing — select
# ------------------------------------------------------------------


class TestTraceSelect:
    def test_select_captures_step(self, traced: TracedDataFrame):
        result = traced.select("status", "price")
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.step_name == "select"
        assert step.col_count == 2

    def test_select_with_expr(self, traced: TracedDataFrame):
        result = traced.select(pl.col("price") * 2)
        trace = _get_trace(result)
        assert trace.steps[0].col_count == 1


# ------------------------------------------------------------------
# Method tracing — drop
# ------------------------------------------------------------------


class TestTraceDrop:
    def test_drop_captures_step(self, traced: TracedDataFrame):
        result = traced.drop("status", "category")
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.step_name == "drop"
        assert step.col_count == 3  # 5 - 2


# ------------------------------------------------------------------
# Method tracing — rename
# ------------------------------------------------------------------


class TestTraceRename:
    def test_rename_captures_step(self, traced: TracedDataFrame):
        result = traced.rename({"price": "unit_price"})
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.step_name == "rename"
        assert "unit_price" in step.schema
        assert "price" not in step.schema


# ------------------------------------------------------------------
# Method tracing — sort
# ------------------------------------------------------------------


class TestTraceSort:
    def test_sort_captures_step(self, traced: TracedDataFrame):
        result = traced.sort("price", descending=True)
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.step_name == "sort"
        assert step.row_count == 5  # same row count
        assert step.col_count == 5  # same col count


# ------------------------------------------------------------------
# Method tracing — head / tail
# ------------------------------------------------------------------


class TestTraceHeadTail:
    def test_head_captures_step(self, traced: TracedDataFrame):
        result = traced.head(3)
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.step_name == "head"
        assert step.row_count == 3

    def test_tail_captures_step(self, traced: TracedDataFrame):
        result = traced.tail(2)
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        assert trace.steps[0].row_count == 2


# ------------------------------------------------------------------
# Method tracing — unique
# ------------------------------------------------------------------


class TestTraceUnique:
    def test_unique_captures_step(self, traced: TracedDataFrame):
        result = traced.unique(subset=["category"])
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.step_name == "unique"
        assert step.row_count == 2  # "A" and "B"


# ------------------------------------------------------------------
# Method tracing — chaining
# ------------------------------------------------------------------


class TestChaining:
    def test_multi_step_chain(self, traced: TracedDataFrame):
        result = (
            traced.filter(pl.col("status") == "active")
            .with_columns((pl.col("price") * pl.col("quantity")).alias("revenue"))
            .sort("revenue", descending=True)
        )
        trace = _get_trace(result)

        assert len(trace.steps) == 3
        assert trace.steps[0].step_name == "filter"
        assert trace.steps[1].step_name == "with_columns"
        assert trace.steps[2].step_name == "sort"

    def test_chain_preserves_data(self, traced: TracedDataFrame):
        result = traced.filter(pl.col("status") == "active").select("status", "price")

        # Unwrap and verify data
        real_df = unwrap(result)
        assert type(real_df) is pl.DataFrame
        assert real_df.shape == (3, 2)
        assert real_df.columns == ["status", "price"]

    def test_chain_row_diffs_accumulate(self, traced: TracedDataFrame):
        trace = _get_trace(traced)
        from flowview.collector import capture_snapshot

        trace.input_snapshot = capture_snapshot(
            object.__getattribute__(traced, "_df"),
            step_name="input",
            sample_rows=3,
        )

        result = traced.filter(pl.col("status") == "active").head(2)
        trace = _get_trace(result)

        assert trace.steps[0].row_diff == -2  # 5 -> 3
        assert trace.steps[1].row_diff == -1  # 3 -> 2


# ------------------------------------------------------------------
# Non-DataFrame returns (no tracing)
# ------------------------------------------------------------------


class TestNonDataFrameReturns:
    def test_shape_not_traced(self, traced: TracedDataFrame):
        _ = traced.shape
        trace = _get_trace(traced)
        assert len(trace.steps) == 0

    def test_columns_not_traced(self, traced: TracedDataFrame):
        _ = traced.columns
        trace = _get_trace(traced)
        assert len(trace.steps) == 0

    def test_to_dict_not_traced(self, traced: TracedDataFrame):
        _ = traced.to_dict()
        trace = _get_trace(traced)
        assert len(trace.steps) == 0

    def test_non_df_method_returns_raw_value(self, traced: TracedDataFrame):
        result = traced.to_dict()
        assert isinstance(result, dict)
        assert type(result) is not TracedDataFrame


# ------------------------------------------------------------------
# Arg unwrapping
# ------------------------------------------------------------------


class TestArgUnwrapping:
    def test_join_with_real_df(self, traced: TracedDataFrame):
        other = pl.DataFrame({"id": [1, 2], "label": ["x", "y"]})
        result = traced.join(other, on="id", how="left")
        trace = _get_trace(result)

        assert len(trace.steps) == 1
        assert trace.steps[0].step_name == "join"

    def test_join_with_proxy_as_other(self, sample_df: pl.DataFrame):
        """When a proxied DataFrame is passed as the 'other' arg to join,
        it should be unwrapped automatically."""
        trace1 = PipelineTrace(function_name="t1")
        trace2 = PipelineTrace(function_name="t2")

        other_df = pl.DataFrame({"id": [1, 2], "label": ["x", "y"]})
        other_proxy = TracedDataFrame(other_df, trace2, sample_rows=3)

        main_proxy = TracedDataFrame(sample_df, trace1, sample_rows=3)

        # This should work — the proxy unwraps other_proxy before calling join
        result = main_proxy.join(other_proxy, on="id", how="left")

        trace = _get_trace(result)
        assert len(trace.steps) == 1
        assert trace.steps[0].step_name == "join"


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestErrorHandling:
    def test_error_propagates(self, traced: TracedDataFrame):
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            traced.filter(pl.col("nonexistent") > 0)

    def test_trace_state_after_error(self, traced: TracedDataFrame):
        """Steps before an error should still be recorded."""
        result = traced.filter(pl.col("status") == "active")

        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            result.filter(pl.col("nonexistent") > 0)

        trace = _get_trace(result)
        assert len(trace.steps) == 1  # the first filter was recorded


# ------------------------------------------------------------------
# Sample data capture
# ------------------------------------------------------------------


class TestSampleCapture:
    def test_sample_rows_respected(self, sample_df: pl.DataFrame):
        trace = PipelineTrace(function_name="test")
        proxy = TracedDataFrame(sample_df, trace, sample_rows=2)

        result = proxy.filter(pl.col("status") == "active")
        step = _get_trace(result).steps[0]
        assert step.sample.shape[0] <= 2

    def test_sample_captures_data(self, traced: TracedDataFrame):
        result = traced.filter(pl.col("status") == "active")
        step = _get_trace(result).steps[0]
        assert step.sample.shape[0] > 0
        assert step.sample.columns == ["status", "category", "price", "quantity", "id"]
