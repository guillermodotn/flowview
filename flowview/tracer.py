"""Core @trace decorator that wraps DataFrames in a tracing proxy."""

from __future__ import annotations

import asyncio
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
        def _setup(
            args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[
            dict[str, Any],
            PipelineTrace,
            tuple[Any, ...],
            dict[str, Any],
        ]:
            config = {
                "sample_rows": sample_rows,
                "show_sample": show_sample,
                "show_schema": show_schema,
            }

            pipeline_trace = PipelineTrace(function_name=func.__name__)

            input_df = _find_dataframe_arg(args, kwargs)
            if input_df is not None:
                pipeline_trace.input_snapshot = capture_snapshot(
                    input_df,
                    step_name="input",
                    sample_rows=sample_rows,
                )

            new_args, new_kwargs = _replace_dataframe_arg(
                args, kwargs, pipeline_trace, sample_rows
            )

            return config, pipeline_trace, new_args, new_kwargs

        def _finish(
            result: Any,
            pipeline_trace: PipelineTrace,
            config: dict[str, Any],
            elapsed_ms: float,
        ) -> Any:
            pipeline_trace.total_time_ms = elapsed_ms
            result = unwrap(result)

            try:
                render_trace(pipeline_trace, config)
            except Exception as e:
                warnings.warn(
                    f"flowview: failed to render trace: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            return result

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                config, pipeline_trace, new_args, new_kwargs = _setup(args, kwargs)
                start = time.perf_counter()
                result = await func(*new_args, **new_kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                return _finish(result, pipeline_trace, config, elapsed_ms)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config, pipeline_trace, new_args, new_kwargs = _setup(args, kwargs)
            start = time.perf_counter()
            result = func(*new_args, **new_kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return _finish(result, pipeline_trace, config, elapsed_ms)

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
            return unwrap(arg)
    for val in kwargs.values():
        if isinstance(val, pl.DataFrame):
            return unwrap(val)
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
            # Unwrap any existing proxy to avoid double-wrapping
            real_df = unwrap(arg)
            new_args[i] = TracedDataFrame(real_df, trace, sample_rows)
            return tuple(new_args), kwargs

    new_kwargs = dict(kwargs)
    for key, val in new_kwargs.items():
        if isinstance(val, pl.DataFrame):
            real_df = unwrap(val)
            new_kwargs[key] = TracedDataFrame(real_df, trace, sample_rows)
            return args, new_kwargs

    return args, kwargs
