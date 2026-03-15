"""Proxy wrapper for transparent DataFrame method chain tracing."""

from __future__ import annotations

import time
from typing import Any

import polars as pl

from flowview.collector import capture_snapshot
from flowview.models import PipelineTrace, StepSnapshot


class TracedDataFrame:
    """Proxy that intercepts DataFrame method calls for tracing.

    Wraps a real ``pl.DataFrame`` and records a :class:`StepSnapshot`
    each time a method returns a new DataFrame.  Non-DataFrame results
    and attribute accesses are delegated transparently to the underlying
    DataFrame.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        trace: PipelineTrace,
        sample_rows: int = 5,
    ) -> None:
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_trace", trace)
        object.__setattr__(self, "_sample_rows", sample_rows)

    # ------------------------------------------------------------------
    # Type transparency
    # ------------------------------------------------------------------

    @property
    def __class__(self) -> type:
        """Make ``isinstance(proxy, pl.DataFrame)`` return ``True``."""
        return pl.DataFrame  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Dunder methods (not intercepted by __getattr__)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return repr(self._df)

    def __str__(self) -> str:
        return str(self._df)

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Any:
        return iter(self._df)

    def __contains__(self, item: Any) -> bool:
        return item in self._df

    def __getitem__(self, key: Any) -> Any:
        return self._df[key]

    # ------------------------------------------------------------------
    # Core interception
    # ------------------------------------------------------------------

    _GROUPBY_METHODS = frozenset({"group_by", "group_by_dynamic", "rolling"})

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._df, name)

        if not callable(attr):
            return attr

        if name == "pipe":
            return self._make_pipe_wrapper()

        if name in self._GROUPBY_METHODS:
            return self._make_groupby_wrapper(name, attr)

        return self._make_traced_wrapper(name, attr)

    # ------------------------------------------------------------------
    # Generic method wrapper
    # ------------------------------------------------------------------

    def _make_traced_wrapper(self, method_name: str, method: Any) -> Any:
        """Create a wrapper that traces a method call if it returns a DataFrame."""
        proxy = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Unwrap any proxy args so Polars receives real DataFrames
            clean_args = tuple(unwrap(a) for a in args)
            clean_kwargs = {k: unwrap(v) for k, v in kwargs.items()}

            start = time.perf_counter()
            result = method(*clean_args, **clean_kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if isinstance(result, pl.DataFrame):
                previous = _get_previous(proxy._trace, proxy._df)
                snapshot = capture_snapshot(
                    result,
                    step_name=method_name,
                    execution_time_ms=elapsed_ms,
                    sample_rows=proxy._sample_rows,
                    previous=previous,
                )
                proxy._trace.steps.append(snapshot)
                return TracedDataFrame(result, proxy._trace, proxy._sample_rows)

            return result

        return wrapper

    # ------------------------------------------------------------------
    # Pipe wrapper
    # ------------------------------------------------------------------

    def _make_pipe_wrapper(self) -> Any:
        """Wrap ``.pipe()`` using the function name as step label."""
        proxy = self

        def wrapper(function: Any, *args: Any, **kwargs: Any) -> Any:
            step_name = getattr(function, "__name__", str(function))

            start = time.perf_counter()
            result = function(proxy._df, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if isinstance(result, pl.DataFrame):
                previous = _get_previous(proxy._trace, proxy._df)
                snapshot = capture_snapshot(
                    result,
                    step_name=step_name,
                    execution_time_ms=elapsed_ms,
                    sample_rows=proxy._sample_rows,
                    previous=previous,
                )
                proxy._trace.steps.append(snapshot)
                return TracedDataFrame(result, proxy._trace, proxy._sample_rows)

            return result

        return wrapper

    # ------------------------------------------------------------------
    # GroupBy wrapper
    # ------------------------------------------------------------------

    def _make_groupby_wrapper(self, method_name: str, method: Any) -> Any:
        """Create a wrapper for group_by methods that traces the combined step."""
        proxy = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            gb_result = method(*args, **kwargs)

            # Patch .agg() on the returned GroupBy object
            original_agg = gb_result.agg

            def traced_agg(*agg_args: Any, **agg_kwargs: Any) -> Any:
                result = original_agg(*agg_args, **agg_kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000

                if isinstance(result, pl.DataFrame):
                    # Build combined step name: group_by(col1, col2).agg(...)
                    by_cols = _format_groupby_cols(gb_result)
                    step_name = f"{method_name}({by_cols}).agg"

                    previous = _get_previous(proxy._trace, proxy._df)
                    snapshot = capture_snapshot(
                        result,
                        step_name=step_name,
                        execution_time_ms=elapsed_ms,
                        sample_rows=proxy._sample_rows,
                        previous=previous,
                    )
                    proxy._trace.steps.append(snapshot)
                    return TracedDataFrame(result, proxy._trace, proxy._sample_rows)

                return result

            gb_result.agg = traced_agg
            return gb_result

        return wrapper


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def unwrap(obj: Any) -> Any:
    """Extract the real DataFrame from a TracedDataFrame, or return as-is."""
    if type(obj) is TracedDataFrame:
        return object.__getattribute__(obj, "_df")
    return obj


def _get_previous(
    trace: PipelineTrace,
    current_df: pl.DataFrame,
) -> StepSnapshot | None:
    """Return the most recent snapshot for diffing against."""
    if trace.steps:
        return trace.steps[-1]
    if trace.input_snapshot:
        return trace.input_snapshot
    return None


def _format_groupby_cols(gb: Any) -> str:
    """Extract grouping column names from a GroupBy-like object."""
    try:
        by = gb.by  # type: ignore[attr-defined]
        if isinstance(by, (list, tuple)):
            return ", ".join(str(c) for c in by)
        return str(by)
    except Exception:
        return "..."
