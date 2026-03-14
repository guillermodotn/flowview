"""Capture DataFrame state and compute diffs between steps."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import polars as pl

from flowview.models import SchemaDiff, StepSnapshot


def capture_snapshot(
    df: pl.DataFrame,
    step_name: str,
    execution_time_ms: float = 0.0,
    sample_rows: int = 5,
    previous: StepSnapshot | None = None,
) -> StepSnapshot:
    """Capture the current state of a DataFrame as a StepSnapshot."""
    schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes, strict=True)}
    sample = df.head(sample_rows)
    row_count = df.shape[0]
    col_count = df.shape[1]

    row_diff = None
    schema_diff = None
    if previous is not None:
        row_diff = row_count - previous.row_count
        schema_diff = compute_schema_diff(previous.schema, schema)

    return StepSnapshot(
        step_name=step_name,
        row_count=row_count,
        col_count=col_count,
        schema=schema,
        sample=sample,
        execution_time_ms=execution_time_ms,
        row_diff=row_diff,
        schema_diff=schema_diff,
    )


def compute_schema_diff(
    before: dict[str, str],
    after: dict[str, str],
) -> SchemaDiff:
    """Compute the difference between two schemas."""
    before_cols = set(before.keys())
    after_cols = set(after.keys())

    added = sorted(after_cols - before_cols)
    removed = sorted(before_cols - after_cols)

    type_changed: dict[str, tuple[str, str]] = {}
    for col in before_cols & after_cols:
        if before[col] != after[col]:
            type_changed[col] = (before[col], after[col])

    return SchemaDiff(added=added, removed=removed, type_changed=type_changed)


def timed_call(
    fn: Callable[..., Any],
    df: pl.DataFrame,
    *args: Any,
    **kwargs: Any,
) -> tuple[pl.DataFrame, float]:
    """Call a function with timing, returning (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn(df, *args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if not isinstance(result, pl.DataFrame):
        raise TypeError(
            f"Pipeline step '{fn.__name__}' returned {type(result).__name__}, "
            f"expected polars.DataFrame"
        )

    return result, elapsed_ms
