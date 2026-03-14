"""Core @trace decorator that intercepts .pipe() calls."""

from __future__ import annotations

import contextvars
import functools
from collections.abc import Callable
from typing import Any, TypeVar, overload

import polars as pl

from flowview.collector import capture_snapshot, timed_call
from flowview.models import PipelineTrace
from flowview.renderer import render_trace

F = TypeVar("F", bound=Callable[..., Any])

# Context variable to track the active trace (supports nesting)
_active_trace: contextvars.ContextVar[PipelineTrace | None] = contextvars.ContextVar(
    "_active_trace", default=None
)
_trace_config: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_trace_config", default=None
)

# Store the original pipe method
_original_pipe = pl.DataFrame.pipe


def _traced_pipe(
    self: pl.DataFrame,
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Replacement for pl.DataFrame.pipe that captures step data."""
    current_trace = _active_trace.get(None)

    if current_trace is None:
        # No active trace — fall through to original
        return _original_pipe(self, function, *args, **kwargs)

    config = _trace_config.get(None) or {}
    sample_rows = config.get("sample_rows", 5)

    # Get the previous snapshot for diffing
    previous = None
    if current_trace.steps:
        previous = current_trace.steps[-1]
    elif current_trace.input_snapshot:
        previous = current_trace.input_snapshot

    # Execute with timing
    result, elapsed_ms = timed_call(function, self, *args, **kwargs)

    # Determine step name
    step_name = getattr(function, "__name__", str(function))

    # Capture snapshot
    snapshot = capture_snapshot(
        result,
        step_name=step_name,
        execution_time_ms=elapsed_ms,
        sample_rows=sample_rows,
        previous=previous,
    )

    current_trace.steps.append(snapshot)
    return result


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

        @fv.trace(sample_rows=3)
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

            # Capture input snapshot from the first DataFrame argument
            input_df = _find_dataframe_arg(args, kwargs)
            if input_df is not None:
                pipeline_trace.input_snapshot = capture_snapshot(
                    input_df,
                    step_name="input",
                    sample_rows=sample_rows,
                )

            # Install the traced pipe and set context
            token_trace = _active_trace.set(pipeline_trace)
            token_config = _trace_config.set(config)

            # Monkey-patch .pipe()
            pl.DataFrame.pipe = _traced_pipe  # type: ignore[assignment]

            try:
                import time

                start = time.perf_counter()
                result = func(*args, **kwargs)
                pipeline_trace.total_time_ms = (time.perf_counter() - start) * 1000
            finally:
                # Always restore
                pl.DataFrame.pipe = _original_pipe  # type: ignore[assignment]
                _active_trace.reset(token_trace)
                _trace_config.reset(token_config)

            # Render the trace
            render_trace(pipeline_trace, config)

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
