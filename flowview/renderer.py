"""Rich terminal rendering for pipeline traces."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from flowview.models import PipelineTrace, SchemaDiff, StepSnapshot

console = Console()


def render_trace(trace: PipelineTrace, config: dict[str, Any]) -> None:
    """Render a complete pipeline trace to the terminal."""
    show_sample = config.get("show_sample", True)
    show_schema = config.get("show_schema", False)

    console.print()

    # Header
    header = Text(f" flowview: {trace.function_name} ", style="bold white on blue")
    console.print(header, justify="center")
    console.print()

    # Input step
    if trace.input_snapshot:
        _render_input(trace.input_snapshot)

    # Pipeline steps
    for step in trace.steps:
        _render_arrow()
        _render_step(step, show_sample=show_sample, show_schema=show_schema)

    # Footer
    console.print()
    total = f"Total: {_format_time(trace.total_time_ms)}"
    steps_count = f"{len(trace.steps)} steps"
    footer = Text(f" {steps_count} | {total} ", style="bold white on blue")
    console.print(footer, justify="center")
    console.print()


def _render_input(snapshot: StepSnapshot) -> None:
    """Render the input DataFrame info."""
    shape_text = _format_shape(snapshot.row_count, snapshot.col_count)
    panel = Panel(
        shape_text,
        title="[bold]Input[/bold]",
        title_align="left",
        border_style="dim",
        padding=(0, 1),
    )
    console.print(panel)


def _render_arrow() -> None:
    """Render a downward arrow between steps."""
    console.print("  [dim]|[/dim]")
    console.print("  [dim]v[/dim]")


def _render_step(
    step: StepSnapshot,
    show_sample: bool = True,
    show_schema: bool = False,
) -> None:
    """Render a single pipeline step."""
    # Build the content
    parts: list[str] = []

    # Shape line with diff
    shape_line = _format_shape(step.row_count, step.col_count)
    if step.row_diff is not None and step.row_diff != 0:
        diff_style = "green" if step.row_diff > 0 else "red"
        sign = "+" if step.row_diff > 0 else ""
        shape_line += f"  [{diff_style}]({sign}{step.row_diff:,} rows)[/{diff_style}]"

    parts.append(shape_line)

    # Schema diff
    if step.schema_diff and step.schema_diff.has_changes:
        parts.append(_format_schema_diff(step.schema_diff))

    # Timing
    parts.append(f"[dim]{_format_time(step.execution_time_ms)}[/dim]")

    content = "\n".join(parts)

    # Build the panel title
    title = f"[bold]{step.step_name}[/bold]"

    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(panel)

    # Sample table (outside the panel for better readability)
    if show_sample and step.sample.shape[0] > 0:
        _render_sample_table(step.sample)

    # Full schema
    if show_schema:
        _render_schema(step.schema)


def _render_sample_table(df: Any) -> None:
    """Render a sample DataFrame as a Rich table."""
    table = Table(
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        padding=(0, 1),
        show_edge=False,
    )

    # Add columns
    for col_name in df.columns:
        table.add_column(col_name, style="dim" if col_name.startswith("_") else "")

    # Add rows
    for row in df.iter_rows():
        table.add_row(*[_format_cell(v) for v in row])

    console.print(table)


def _render_schema(schema: dict[str, str]) -> None:
    """Render a full schema listing."""
    table = Table(
        title="Schema",
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Column", style="bold")
    table.add_column("Type", style="cyan")

    for col, dtype in schema.items():
        table.add_row(col, dtype)

    console.print(table)


def _format_shape(rows: int, cols: int) -> str:
    """Format a shape as 'N rows x M cols'."""
    return f"[bold]{rows:,}[/bold] rows x [bold]{cols}[/bold] cols"


def _format_time(ms: float) -> str:
    """Format milliseconds into a readable string."""
    if ms < 1:
        return f"{ms * 1000:.0f}µs"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


def _format_schema_diff(diff: SchemaDiff) -> str:
    """Format a schema diff into a readable string."""
    parts: list[str] = []

    if diff.added:
        cols = ", ".join(diff.added)
        parts.append(f"[green]+cols: {cols}[/green]")

    if diff.removed:
        cols = ", ".join(diff.removed)
        parts.append(f"[red]-cols: {cols}[/red]")

    if diff.type_changed:
        changes = ", ".join(
            f"{col}: {old}->{new}" for col, (old, new) in diff.type_changed.items()
        )
        parts.append(f"[yellow]~types: {changes}[/yellow]")

    return "  ".join(parts)


def _format_cell(value: Any) -> str:
    """Format a cell value for display."""
    if value is None:
        return "[dim]null[/dim]"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
