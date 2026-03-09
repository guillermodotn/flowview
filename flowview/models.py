"""Data models for pipeline tracing."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass(frozen=True)
class SchemaDiff:
    """Difference in schema between two consecutive pipeline steps."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    type_changed: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.type_changed)


@dataclass
class StepSnapshot:
    """Captured state of a DataFrame at one pipeline step."""

    step_name: str
    row_count: int
    col_count: int
    schema: dict[str, str]  # col_name -> dtype string
    sample: pl.DataFrame  # first N rows
    execution_time_ms: float
    row_diff: int | None = None  # vs previous step (negative = rows removed)
    schema_diff: SchemaDiff | None = None


@dataclass
class PipelineTrace:
    """Complete trace of a pipeline execution."""

    function_name: str
    steps: list[StepSnapshot] = field(default_factory=list)
    input_snapshot: StepSnapshot | None = None
    total_time_ms: float = 0.0
