# flowview

[![CI](https://github.com/guillermodotn/flowview/actions/workflows/ci.yml/badge.svg)](https://github.com/guillermodotn/flowview/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/flowview)](https://pypi.org/project/flowview/)
[![Python](https://img.shields.io/pypi/pyversions/flowview)](https://pypi.org/project/flowview/)
[![License](https://img.shields.io/github/license/guillermodotn/flowview)](https://github.com/guillermodotn/flowview/blob/main/LICENSE)

Visual data pipeline debugger for Polars. Stop print-debugging your pipelines.

<p align="center">
  <img src="assets/demo.svg" alt="flowview demo" width="800">
</p>

## Install

```bash
pip install flowview
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add flowview
```

## Quick Start

Add `@fv.trace` to any function that transforms a Polars DataFrame. flowview traces every method call and renders a visual flow in your terminal.

```python
import polars as pl
import flowview as fv

@fv.trace
def process(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("status") == "active")
          .with_columns((pl.col("price") * pl.col("quantity")).alias("revenue"))
          .group_by("category")
          .agg(pl.col("revenue").sum().alias("total_revenue"))
          .sort("total_revenue", descending=True)
    )

df = pl.DataFrame({
    "status": ["active", "inactive", "active"],
    "category": ["Books", "Books", "Electronics"],
    "price": [14.99, 599.99, 299.99],
    "quantity": [5, 1, 2],
})

result = process(df)
```

`.pipe()` chains work too:

```python
@fv.trace
def process(df: pl.DataFrame) -> pl.DataFrame:
    return df.pipe(clean).pipe(filter_active).pipe(add_revenue)
```

## What You See

Each step in your pipeline is displayed as a box showing:

- **Row count** with diff from the previous step (e.g., `700 rows x 4 cols (-300 rows)`)
- **Schema changes** — columns added or removed (e.g., `+cols: revenue  -cols: status`)
- **Sample data** — first N rows at each transformation
- **Execution time** per step

Steps are connected with arrows to show the flow. A summary footer shows the total step count and wall-clock time.

## Supported Operations

flowview traces any DataFrame method that returns a new DataFrame. These methods get human-readable step names:

| Method | Step name example |
|---|---|
| `filter(expr)` | `filter((col("status")) == ("active"))` |
| `with_columns(exprs)` | `with_columns(revenue, tax)` |
| `select(cols)` | `select(status, price)` |
| `drop(cols)` | `drop(status, category)` |
| `rename(mapping)` | `rename(price->unit_price)` |
| `sort(cols)` | `sort(price, quantity)` |
| `head(n)` / `tail(n)` | `head(10)` / `tail(5)` |
| `unique(subset)` | `unique(id)` |
| `join(other, ...)` | `join(on=id, how=left)` |
| `group_by(cols).agg(exprs)` | `group_by(category).agg(total_revenue)` |
| `pipe(fn)` | uses the function name, e.g. `clean_data` |

Other methods (e.g., `explode`, `melt`, `unpivot`) are traced with a fallback name like `explode('tags')`.

## Options

```python
@fv.trace(sample_rows=3, show_sample=True, show_schema=True)
def process(df):
    ...
```

| Option | Type | Default | Description |
|---|---|---|---|
| `sample_rows` | `int` | `5` | Number of sample rows to capture at each step |
| `show_sample` | `bool` | `True` | Display sample data tables in the output |
| `show_schema` | `bool` | `False` | Display the full schema at each step |

## How It Works

The `@fv.trace` decorator wraps the first DataFrame argument in a lightweight proxy before calling your function. The proxy intercepts every method call, captures a snapshot of the result (row count, schema, sample rows, timing), and delegates to the real Polars DataFrame underneath. When your function returns, the proxy is unwrapped and you get back a regular `pl.DataFrame`.

There is no monkey-patching and no global state. Each decorated call is fully isolated.

## Limitations

- **LazyFrame** is not supported — `df.lazy()` exits the proxy. Only eager DataFrames are traced.
- **GroupBy shortcuts** like `.count()`, `.sum()`, `.first()` on a GroupBy object are not traced — use `.agg()` instead.
- **Pipe internals** are not individually traced — `df.pipe(fn)` produces a single step named after `fn`, not one step per operation inside `fn`.
- **IDE autocomplete** may not show DataFrame methods inside the decorated function body.
- **`type(df)`** returns `TracedDataFrame` inside the decorated function. `isinstance(df, pl.DataFrame)` works correctly.
- **Only the first DataFrame argument** is wrapped when a function takes multiple DataFrames.

## License

[MIT](LICENSE)
