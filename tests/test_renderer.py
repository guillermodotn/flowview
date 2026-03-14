"""Tests for flowview.renderer."""

import polars as pl

from flowview.models import PipelineTrace, SchemaDiff, StepSnapshot
from flowview.renderer import (
    _format_cell,
    _format_schema_diff,
    _format_shape,
    _format_time,
    render_trace,
)


class TestFormatTime:
    def test_microseconds(self):
        assert _format_time(0.5) == "500us"

    def test_milliseconds(self):
        assert _format_time(42.3) == "42.3ms"

    def test_seconds(self):
        assert _format_time(1500.0) == "1.50s"


class TestFormatShape:
    def test_basic(self):
        result = _format_shape(1000, 5)
        assert "1,000" in result
        assert "5" in result


class TestFormatCell:
    def test_none(self):
        assert "null" in _format_cell(None)

    def test_float(self):
        assert _format_cell(3.14159) == "3.14"

    def test_int(self):
        assert _format_cell(1000) == "1,000"

    def test_string(self):
        assert _format_cell("hello") == "hello"


class TestFormatSchemaDiff:
    def test_added(self):
        diff = SchemaDiff(added=["col_a", "col_b"])
        result = _format_schema_diff(diff)
        assert "col_a" in result
        assert "col_b" in result

    def test_removed(self):
        diff = SchemaDiff(removed=["old_col"])
        result = _format_schema_diff(diff)
        assert "old_col" in result

    def test_type_changed(self):
        diff = SchemaDiff(type_changed={"x": ("Int64", "Float64")})
        result = _format_schema_diff(diff)
        assert "Int64" in result
        assert "Float64" in result


class TestRenderTrace:
    def test_renders_without_error(self):
        """Smoke test: rendering doesn't crash."""
        trace = PipelineTrace(
            function_name="test_pipeline",
            input_snapshot=StepSnapshot(
                step_name="input",
                row_count=100,
                col_count=3,
                schema={"a": "Int64", "b": "String", "c": "Float64"},
                sample=pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]}),
                execution_time_ms=0,
            ),
            steps=[
                StepSnapshot(
                    step_name="filter_active",
                    row_count=80,
                    col_count=3,
                    schema={"a": "Int64", "b": "String", "c": "Float64"},
                    sample=pl.DataFrame(
                        {"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]}
                    ),
                    execution_time_ms=12.5,
                    row_diff=-20,
                    schema_diff=SchemaDiff(),
                ),
                StepSnapshot(
                    step_name="add_revenue",
                    row_count=80,
                    col_count=4,
                    schema={
                        "a": "Int64",
                        "b": "String",
                        "c": "Float64",
                        "revenue": "Float64",
                    },
                    sample=pl.DataFrame(
                        {
                            "a": [1, 2],
                            "b": ["x", "y"],
                            "c": [1.0, 2.0],
                            "revenue": [10.0, 20.0],
                        }
                    ),
                    execution_time_ms=3.2,
                    row_diff=0,
                    schema_diff=SchemaDiff(added=["revenue"]),
                ),
            ],
            total_time_ms=15.7,
        )

        config = {"show_sample": True, "show_schema": False}

        # Should not raise
        render_trace(trace, config)
