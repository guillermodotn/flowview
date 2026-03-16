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

## Usage

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
```

`.pipe()` chains are also supported:

```python
@fv.trace
def process(df: pl.DataFrame) -> pl.DataFrame:
    return df.pipe(clean).pipe(filter_active).pipe(add_revenue)
```

Each step shows:

- Row counts with diffs
- Schema changes (added/removed columns)
- Sample data at each transformation
- Execution time per step

## Options

```python
@fv.trace(sample_rows=3, show_sample=True, show_schema=True)
def process(df):
    ...
```
