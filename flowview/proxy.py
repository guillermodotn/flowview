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

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._df, name)

        if not callable(attr):
            return attr

        return self._make_traced_wrapper(name, attr)

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
