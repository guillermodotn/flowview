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
                step_name = _summarize_step(method_name, clean_args, clean_kwargs)
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
    # Pipe wrapper
    # ------------------------------------------------------------------

    def _make_pipe_wrapper(self) -> Any:
        """Wrap ``.pipe()`` using the function name as step label."""
        proxy = self

        def wrapper(function: Any, *args: Any, **kwargs: Any) -> Any:
            step_name = getattr(function, "__name__", str(function))

            # Unwrap any proxy args so piped functions receive real DataFrames
            clean_args = tuple(unwrap(a) for a in args)
            clean_kwargs = {k: unwrap(v) for k, v in kwargs.items()}

            start = time.perf_counter()
            result = function(proxy._df, *clean_args, **clean_kwargs)
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
                    step_name = _summarize_groupby(
                        method_name,
                        gb_result,
                        agg_args,
                        agg_kwargs,
                    )

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


# ------------------------------------------------------------------
# Expression summarizer
# ------------------------------------------------------------------

_MAX_STEP_NAME_LEN = 60


def _truncate(s: str, max_len: int = _MAX_STEP_NAME_LEN) -> str:
    """Truncate a string with ``...`` suffix if it exceeds *max_len*."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _safe_repr(obj: Any, max_len: int = 45) -> str:
    """Return ``repr(obj)`` truncated, never raising."""
    try:
        return _truncate(repr(obj), max_len)
    except Exception:
        return "?"


def _expr_output_names(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    """Collect output column names from expression args and kwargs.

    Shared by ``with_columns``, ``select``, and ``group_by().agg()``.
    """
    names: list[str] = []
    for arg in args:
        if isinstance(arg, str):
            names.append(arg)
        elif isinstance(arg, pl.Expr):
            try:
                names.append(arg.meta.output_name())
            except Exception:
                names.append(str(arg))
        elif isinstance(arg, (list, tuple)):
            # with_columns([expr1, expr2]) passes a list as first arg
            for item in arg:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, pl.Expr):
                    try:
                        names.append(item.meta.output_name())
                    except Exception:
                        names.append(str(item))
    for name in kwargs:
        names.append(name)  # kwargs key IS the output name
    return names


# -- Method-specific summarizers -----------------------------------------


def _summarize_filter(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """``filter(col("status") == "active")``."""
    if not args:
        return "filter"
    text = str(args[0])
    # str(expr) wraps in outer [] — strip them for readability
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return f"filter({_truncate(text, 45)})"


def _summarize_with_columns(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    names = _expr_output_names(args, kwargs)
    if not names:
        return "with_columns"
    return f"with_columns({', '.join(names)})"


def _summarize_select(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    names = _expr_output_names(args, kwargs)
    if not names:
        return "select"
    return f"select({', '.join(names)})"


def _summarize_drop(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    names = [str(a) for a in args if isinstance(a, str)]
    if not names:
        return "drop"
    return f"drop({', '.join(names)})"


def _summarize_rename(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if args and isinstance(args[0], dict):
        pairs = [f"{old}->{new}" for old, new in args[0].items()]
        return f"rename({', '.join(pairs)})"
    return "rename"


def _summarize_sort(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    names: list[str] = []
    for a in args:
        if isinstance(a, str):
            names.append(a)
        elif isinstance(a, pl.Expr):
            try:
                names.append(a.meta.output_name())
            except Exception:
                names.append(str(a))
    if not names:
        return "sort"
    return f"sort({', '.join(names)})"


def _summarize_numeric(
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    """``head(5)``, ``tail(10)``, ``limit(100)``, ``slice(0, 10)``."""
    nums = [str(a) for a in args if isinstance(a, (int, float))]
    if nums:
        return f"{method_name}({', '.join(nums)})"
    return method_name


def _summarize_unique(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    subset = kwargs.get("subset") or (args[0] if args else None)
    if subset is None:
        return "unique"
    if isinstance(subset, str):
        return f"unique({subset})"
    if isinstance(subset, (list, tuple)):
        return f"unique({', '.join(str(s) for s in subset)})"
    return "unique"


def _summarize_join(
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    parts: list[str] = []
    on = kwargs.get("on")
    if on is not None:
        if isinstance(on, (list, tuple)):
            parts.append(f"on={', '.join(str(c) for c in on)}")
        else:
            parts.append(f"on={on}")
    else:
        left_on = kwargs.get("left_on")
        if left_on is not None:
            if isinstance(left_on, (list, tuple)):
                parts.append(f"left_on={', '.join(str(c) for c in left_on)}")
            else:
                parts.append(f"left_on={left_on}")
        right_on = kwargs.get("right_on")
        if right_on is not None:
            if isinstance(right_on, (list, tuple)):
                parts.append(f"right_on={', '.join(str(c) for c in right_on)}")
            else:
                parts.append(f"right_on={right_on}")
    how = kwargs.get("how")
    if how is not None:
        parts.append(f"how={how}")
    if parts:
        return f"{method_name}({', '.join(parts)})"
    return method_name


# -- Dispatch table -------------------------------------------------------

_SUMMARIZERS: dict[
    str,
    Any,  # callable
] = {
    "filter": lambda a, k: _summarize_filter(a, k),
    "with_columns": lambda a, k: _summarize_with_columns(a, k),
    "select": lambda a, k: _summarize_select(a, k),
    "drop": lambda a, k: _summarize_drop(a, k),
    "rename": lambda a, k: _summarize_rename(a, k),
    "sort": lambda a, k: _summarize_sort(a, k),
    "head": lambda a, k: _summarize_numeric("head", a, k),
    "tail": lambda a, k: _summarize_numeric("tail", a, k),
    "limit": lambda a, k: _summarize_numeric("limit", a, k),
    "slice": lambda a, k: _summarize_numeric("slice", a, k),
    "sample": lambda a, k: _summarize_numeric("sample", a, k),
    "unique": lambda a, k: _summarize_unique(a, k),
    "join": lambda a, k: _summarize_join("join", a, k),
    "join_asof": lambda a, k: _summarize_join("join_asof", a, k),
}


def _summarize_step(
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    """Generate a readable step name from a method call.

    Fallback chain:
    1. Method-specific summarizer
    2. ``method_name(str(args[0]))`` truncated
    3. Bare method name
    """
    try:
        summarizer = _SUMMARIZERS.get(method_name)
        if summarizer is not None:
            return _truncate(summarizer(args, kwargs))
        # Fallback: show first arg if available
        if args:
            first = _safe_repr(args[0], 45)
            return _truncate(f"{method_name}({first})")
        return method_name
    except Exception:
        return method_name


def _summarize_groupby(
    method_name: str,
    gb_result: Any,
    agg_args: tuple[Any, ...],
    agg_kwargs: dict[str, Any],
) -> str:
    """``group_by(category).agg(total_revenue, count)``."""
    try:
        by_cols = _format_groupby_cols(gb_result)
        agg_names = _expr_output_names(agg_args, agg_kwargs)
        if agg_names:
            agg_part = ", ".join(agg_names)
            return _truncate(f"{method_name}({by_cols}).agg({agg_part})")
        return _truncate(f"{method_name}({by_cols}).agg")
    except Exception:
        return f"{method_name}(...).agg"
