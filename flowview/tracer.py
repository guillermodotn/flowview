"""Core @trace decorator that wraps DataFrames in a tracing proxy."""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, overload

import polars as pl

from flowview.collector import capture_snapshot
from flowview.models import PipelineTrace
from flowview.proxy import TracedDataFrame, unwrap
from flowview.renderer import render_trace

F = TypeVar("F", bound=Callable[..., Any])


@overload
def trace(fn: F) -> F: ...


@overload
def trace(
    *,
    sample_rows: int = 5,
    show_sample: bool = True,
    show_schema: bool = False,
) -> Callable[[F], F]: ...


def trace(
    fn: F | None = None,
    *,
    sample_rows: int = 5,
    show_sample: bool = True,
    show_schema: bool = False,
) -> F | Callable[[F], F]:
    """Decorator to trace a Polars pipeline.

    Can be used with or without arguments:

        @fv.trace
        def process(df): ...

        @fv.trace(sample_rows=3, show_schema=True)
        def process(df): ...

    Args:
        sample_rows: Number of sample rows to capture at each step.
        show_sample: Whether to display sample data in the output.
        show_schema: Whether to display full schema at each step.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = {
                "sample_rows": sample_rows,
                "show_sample": show_sample,
                "show_schema": show_schema,
            }

            pipeline_trace = PipelineTrace(function_name=func.__name__)

            # 1. Find input DataFrame, capture input snapshot
            input_df = _find_dataframe_arg(args, kwargs)
            if input_df is not None:
                pipeline_trace.input_snapshot = capture_snapshot(
                    input_df,
                    step_name="input",
                    sample_rows=sample_rows,
                )

            # 2. Replace DataFrame arg with proxy
            new_args, new_kwargs = _replace_dataframe_arg(
                args, kwargs, pipeline_trace, sample_rows
            )

            # 3. Run the function (proxy traces every method call)
            start = time.perf_counter()
            result = func(*new_args, **new_kwargs)
            pipeline_trace.total_time_ms = (time.perf_counter() - start) * 1000

            # 4. Unwrap result — always return real pl.DataFrame
            result = unwrap(result)

            # 5. Render trace
            try:
                render_trace(pipeline_trace, config)
            except Exception as e:
                warnings.warn(
                    f"flowview: failed to render trace: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            return result

        return wrapper  # type: ignore[return-value]

    if fn is not None:
        # Called as @trace without parentheses
        return decorator(fn)

    # Called as @trace(...) with arguments
    return decorator


def _find_dataframe_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> pl.DataFrame | None:
    """Find the first pl.DataFrame in the arguments."""
    for arg in args:
        if isinstance(arg, pl.DataFrame):
            return arg
    for val in kwargs.values():
        if isinstance(val, pl.DataFrame):
            return val
    return None


def _replace_dataframe_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    trace: PipelineTrace,
    sample_rows: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Replace the first DataFrame in args/kwargs with a proxy."""
    new_args = list(args)
    for i, arg in enumerate(new_args):
        if isinstance(arg, pl.DataFrame):
            new_args[i] = TracedDataFrame(arg, trace, sample_rows)
            return tuple(new_args), kwargs

    new_kwargs = dict(kwargs)
    for key, val in new_kwargs.items():
        if isinstance(val, pl.DataFrame):
            new_kwargs[key] = TracedDataFrame(val, trace, sample_rows)
            return args, new_kwargs

    return args, kwargs
