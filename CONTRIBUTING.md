# Contributing to flowview

## Setup

Prerequisites: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/guillermodotn/flowview.git
cd flowview
uv sync --group dev
```

## Development

Run tests:

```bash
uv run pytest tests/ -v
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format .
```

Run an example:

```bash
uv run python examples/method_chain_pipeline.py
```

## Architecture

```
flowview/
  __init__.py      Public API — exports `trace` and `__version__`
  tracer.py        @trace decorator — wraps DataFrame args in proxy, renders output
  proxy.py         TracedDataFrame — intercepts method calls, captures snapshots
  collector.py     capture_snapshot() — records row count, schema, sample data
  models.py        Data classes: StepSnapshot, SchemaDiff, PipelineTrace
  renderer.py      Rich terminal rendering — panels, tables, arrows
```

**How tracing works:**

1. `@fv.trace` wraps the first `pl.DataFrame` argument in a `TracedDataFrame` proxy.
2. Every method call on the proxy (`.filter()`, `.with_columns()`, etc.) is intercepted via `__getattr__`.
3. The real Polars method runs on the underlying DataFrame. If it returns a new DataFrame, a `StepSnapshot` is captured and the result is re-wrapped in a fresh proxy.
4. `.pipe(fn)` passes the real DataFrame to `fn` and records the result as a single step.
5. `.group_by().agg()` is recorded as a combined step.
6. When the decorated function returns, the proxy is unwrapped and a real `pl.DataFrame` is returned.
7. The collected trace is rendered to the terminal using Rich.

No monkey-patching. No global state. Each call is isolated.

## Commit Style

- Lowercase imperative mood: `add feature`, `fix bug`, `remove unused code`
- Keep commits incremental and independently testable

## Versioning and Publishing

- Version is derived from git tags via [hatch-vcs](https://github.com/ofek/hatch-vcs) — no manual version bumps
- CI runs on every push: lint + format + tests across Python 3.10-3.13
- CD publishes to PyPI automatically on GitHub release (trusted publishing via OIDC)
