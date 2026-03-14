"""Tests for flowview.collector."""

import polars as pl
import pytest

from flowview.collector import capture_snapshot, compute_schema_diff, timed_call


class TestCaptureSnapshot:
    def test_basic_capture(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        snap = capture_snapshot(df, step_name="test_step", sample_rows=2)

        assert snap.step_name == "test_step"
        assert snap.row_count == 3
        assert snap.col_count == 2
        assert snap.schema == {"a": "Int64", "b": "String"}
        assert snap.sample.shape == (2, 2)
        assert snap.row_diff is None
        assert snap.schema_diff is None

    def test_capture_with_previous(self):
        df1 = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        snap1 = capture_snapshot(df1, step_name="step1")

        df2 = pl.DataFrame(
            {"a": [1, 2, 3], "b": [10, 20, 30], "c": [True, False, True]}
        )
        snap2 = capture_snapshot(df2, step_name="step2", previous=snap1)

        assert snap2.row_diff == -2
        assert snap2.schema_diff is not None
        assert snap2.schema_diff.added == ["c"]
        assert snap2.schema_diff.removed == []

    def test_capture_empty_df(self):
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
        snap = capture_snapshot(df, step_name="empty")

        assert snap.row_count == 0
        assert snap.col_count == 1
        assert snap.sample.shape == (0, 1)

    def test_sample_rows_limit(self):
        df = pl.DataFrame({"a": list(range(100))})
        snap = capture_snapshot(df, step_name="big", sample_rows=3)

        assert snap.sample.shape[0] == 3
        assert snap.row_count == 100


class TestComputeSchemaDiff:
    def test_no_changes(self):
        schema = {"a": "Int64", "b": "String"}
        diff = compute_schema_diff(schema, schema)

        assert not diff.has_changes
        assert diff.added == []
        assert diff.removed == []
        assert diff.type_changed == {}

    def test_added_columns(self):
        before = {"a": "Int64"}
        after = {"a": "Int64", "b": "String", "c": "Float64"}
        diff = compute_schema_diff(before, after)

        assert diff.has_changes
        assert diff.added == ["b", "c"]
        assert diff.removed == []

    def test_removed_columns(self):
        before = {"a": "Int64", "b": "String", "c": "Float64"}
        after = {"a": "Int64"}
        diff = compute_schema_diff(before, after)

        assert diff.has_changes
        assert diff.removed == ["b", "c"]
        assert diff.added == []

    def test_type_changed(self):
        before = {"a": "Int64", "b": "String"}
        after = {"a": "Float64", "b": "String"}
        diff = compute_schema_diff(before, after)

        assert diff.has_changes
        assert diff.type_changed == {"a": ("Int64", "Float64")}

    def test_mixed_changes(self):
        before = {"a": "Int64", "b": "String"}
        after = {"a": "Float64", "c": "Boolean"}
        diff = compute_schema_diff(before, after)

        assert diff.added == ["c"]
        assert diff.removed == ["b"]
        assert diff.type_changed == {"a": ("Int64", "Float64")}


class TestTimedCall:
    def test_returns_result_and_time(self):
        def add_col(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.lit(1).alias("new"))

        df = pl.DataFrame({"a": [1, 2, 3]})
        result, elapsed = timed_call(add_col, df)

        assert isinstance(result, pl.DataFrame)
        assert "new" in result.columns
        assert elapsed >= 0

    def test_raises_on_non_dataframe_return(self):
        def bad_step(df: pl.DataFrame) -> int:
            return 42  # type: ignore[return-value]

        df = pl.DataFrame({"a": [1]})
        with pytest.raises(TypeError, match=r"expected polars\.DataFrame"):
            timed_call(bad_step, df)
