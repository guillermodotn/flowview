"""Tests for expression summarizer functions in flowview.proxy."""

import polars as pl

from flowview.proxy import (
    _expr_output_names,
    _safe_repr,
    _summarize_groupby,
    _summarize_step,
    _truncate,
)

# ------------------------------------------------------------------
# _truncate
# ------------------------------------------------------------------


class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate("hello", 60) == "hello"

    def test_exact_length_unchanged(self):
        s = "a" * 60
        assert _truncate(s, 60) == s

    def test_long_string_truncated(self):
        s = "a" * 80
        result = _truncate(s, 60)
        assert len(result) == 60
        assert result.endswith("...")

    def test_custom_max_len(self):
        result = _truncate("abcdefghij", 7)
        assert result == "abcd..."


# ------------------------------------------------------------------
# _safe_repr
# ------------------------------------------------------------------


class TestSafeRepr:
    def test_string(self):
        assert _safe_repr("hello") == "'hello'"

    def test_int(self):
        assert _safe_repr(42) == "42"

    def test_long_repr_truncated(self):
        result = _safe_repr("x" * 100, 20)
        assert len(result) <= 20
        assert result.endswith("...")


# ------------------------------------------------------------------
# _expr_output_names
# ------------------------------------------------------------------


class TestExprOutputNames:
    def test_string_args(self):
        names = _expr_output_names(("status", "price"), {})
        assert names == ["status", "price"]

    def test_expr_args(self):
        e = (pl.col("price") * pl.col("quantity")).alias("revenue")
        names = _expr_output_names((e,), {})
        assert names == ["revenue"]

    def test_kwargs_keys(self):
        names = _expr_output_names((), {"doubled": pl.col("x") * 2})
        assert names == ["doubled"]

    def test_mixed_args_and_kwargs(self):
        e = pl.col("price").alias("p")
        names = _expr_output_names((e,), {"extra": pl.col("x")})
        assert names == ["p", "extra"]

    def test_list_arg(self):
        """with_columns([expr1, expr2]) passes a list as first arg."""
        e1 = pl.col("a").alias("aa")
        e2 = pl.col("b").alias("bb")
        names = _expr_output_names(([e1, e2],), {})
        assert names == ["aa", "bb"]

    def test_string_in_list(self):
        names = _expr_output_names((["status", "price"],), {})
        assert names == ["status", "price"]


# ------------------------------------------------------------------
# _summarize_step — method-specific
# ------------------------------------------------------------------


class TestSummarizeFilter:
    def test_simple_equality(self):
        expr = pl.col("status") == "active"
        result = _summarize_step("filter", (expr,), {})
        assert result == 'filter((col("status")) == ("active"))'

    def test_strips_outer_brackets(self):
        expr = pl.col("price") > 100
        result = _summarize_step("filter", (expr,), {})
        # str(expr) is [(col("price")) > (dyn int: 100)]
        # outer [] should be stripped
        assert result.startswith("filter(")
        assert not result.startswith("filter([")

    def test_no_args_fallback(self):
        result = _summarize_step("filter", (), {})
        assert result == "filter"


class TestSummarizeWithColumns:
    def test_single_expr(self):
        e = (pl.col("price") * pl.col("quantity")).alias("revenue")
        result = _summarize_step("with_columns", (e,), {})
        assert result == "with_columns(revenue)"

    def test_named_kwargs(self):
        result = _summarize_step("with_columns", (), {"doubled": pl.col("x") * 2})
        assert result == "with_columns(doubled)"

    def test_multiple_exprs(self):
        e1 = pl.col("a").alias("aa")
        e2 = pl.col("b").alias("bb")
        result = _summarize_step("with_columns", (e1, e2), {})
        assert result == "with_columns(aa, bb)"


class TestSummarizeSelect:
    def test_string_args(self):
        result = _summarize_step("select", ("status", "price"), {})
        assert result == "select(status, price)"

    def test_expr_args(self):
        result = _summarize_step("select", (pl.col("price") * 2,), {})
        assert result == "select(price)"


class TestSummarizeDrop:
    def test_string_args(self):
        result = _summarize_step("drop", ("status", "category"), {})
        assert result == "drop(status, category)"


class TestSummarizeRename:
    def test_dict_mapping(self):
        result = _summarize_step("rename", ({"price": "unit_price"},), {})
        assert result == "rename(price->unit_price)"

    def test_multiple_renames(self):
        result = _summarize_step("rename", ({"a": "x", "b": "y"},), {})
        assert result == "rename(a->x, b->y)"


class TestSummarizeSort:
    def test_single_column(self):
        result = _summarize_step("sort", ("price",), {"descending": True})
        assert result == "sort(price)"

    def test_multiple_columns(self):
        result = _summarize_step("sort", ("price", "quantity"), {})
        assert result == "sort(price, quantity)"


class TestSummarizeNumeric:
    def test_head(self):
        assert _summarize_step("head", (10,), {}) == "head(10)"

    def test_tail(self):
        assert _summarize_step("tail", (5,), {}) == "tail(5)"

    def test_slice(self):
        assert _summarize_step("slice", (0, 10), {}) == "slice(0, 10)"

    def test_no_args(self):
        assert _summarize_step("head", (), {}) == "head"


class TestSummarizeUnique:
    def test_subset_kwarg_list(self):
        result = _summarize_step("unique", (), {"subset": ["id"]})
        assert result == "unique(id)"

    def test_subset_kwarg_string(self):
        result = _summarize_step("unique", (), {"subset": "id"})
        assert result == "unique(id)"

    def test_subset_positional(self):
        result = _summarize_step("unique", (["id", "name"],), {})
        assert result == "unique(id, name)"

    def test_no_subset(self):
        result = _summarize_step("unique", (), {})
        assert result == "unique"


class TestSummarizeJoin:
    def test_on_and_how(self):
        other = pl.DataFrame({"id": [1]})
        result = _summarize_step("join", (other,), {"on": "id", "how": "left"})
        assert result == "join(on=id, how=left)"

    def test_on_list(self):
        other = pl.DataFrame({"a": [1], "b": [2]})
        result = _summarize_step("join", (other,), {"on": ["a", "b"], "how": "inner"})
        assert result == "join(on=a, b, how=inner)"

    def test_left_on(self):
        other = pl.DataFrame({"id": [1]})
        result = _summarize_step("join", (other,), {"left_on": "id", "how": "left"})
        assert result == "join(left_on=id, how=left)"

    def test_left_on_right_on(self):
        other = pl.DataFrame({"other_id": [1]})
        result = _summarize_step(
            "join",
            (other,),
            {"left_on": "id", "right_on": "other_id", "how": "left"},
        )
        assert result == "join(left_on=id, right_on=other_id, how=left)"

    def test_right_on_list(self):
        other = pl.DataFrame({"a": [1], "b": [2]})
        result = _summarize_step(
            "join",
            (other,),
            {"left_on": ["x", "y"], "right_on": ["a", "b"], "how": "inner"},
        )
        assert result == "join(left_on=x, y, right_on=a, b, how=inner)"

    def test_join_asof(self):
        other = pl.DataFrame({"id": [1]})
        result = _summarize_step("join_asof", (other,), {"on": "ts"})
        assert result == "join_asof(on=ts)"


# ------------------------------------------------------------------
# _summarize_step — fallback and truncation
# ------------------------------------------------------------------


class TestSummarizeStepFallback:
    def test_unknown_method_with_args(self):
        """Unknown method shows first arg repr."""
        result = _summarize_step("explode", ("tags",), {})
        assert result == "explode('tags')"

    def test_unknown_method_no_args(self):
        result = _summarize_step("clear", (), {})
        assert result == "clear"

    def test_truncation_at_max_len(self):
        """Very long step name is truncated to 60 chars."""
        result = _summarize_step("select", tuple(f"col{i}" for i in range(50)), {})
        assert len(result) <= 60
        assert result.endswith("...")

    def test_graceful_error_handling(self):
        """Bad arg types don't crash — fall back to method name."""

        class BadRepr:
            def __repr__(self):
                raise RuntimeError("boom")

        result = _summarize_step("unknown", (BadRepr(),), {})
        # Should not raise — falls back gracefully
        assert isinstance(result, str)


# ------------------------------------------------------------------
# _summarize_groupby
# ------------------------------------------------------------------


class TestSummarizeGroupby:
    def test_single_group_col(self):
        df = pl.DataFrame({"cat": ["A", "B"], "val": [1, 2]})
        gb = df.group_by("cat")
        e = pl.col("val").sum().alias("total")
        result = _summarize_groupby("group_by", gb, (e,), {})
        assert result == "group_by(cat).agg(total)"

    def test_multiple_group_cols(self):
        df = pl.DataFrame({"a": ["x", "y"], "b": ["m", "n"], "val": [1, 2]})
        gb = df.group_by("a", "b")
        e = pl.col("val").mean().alias("avg")
        result = _summarize_groupby("group_by", gb, (e,), {})
        assert result == "group_by(a, b).agg(avg)"

    def test_multiple_agg_exprs(self):
        df = pl.DataFrame({"cat": ["A", "B"], "val": [1, 2]})
        gb = df.group_by("cat")
        e1 = pl.col("val").sum().alias("total")
        e2 = pl.col("val").count().alias("cnt")
        result = _summarize_groupby("group_by", gb, (e1, e2), {})
        assert result == "group_by(cat).agg(total, cnt)"

    def test_no_agg_exprs(self):
        df = pl.DataFrame({"cat": ["A", "B"], "val": [1, 2]})
        gb = df.group_by("cat")
        result = _summarize_groupby("group_by", gb, (), {})
        assert result == "group_by(cat).agg"

    def test_truncation(self):
        """Very long groupby step name is truncated."""
        df = pl.DataFrame({f"col{i}": [1, 2] for i in range(20)} | {"v": [1, 2]})
        cols = [f"col{i}" for i in range(20)]
        gb = df.group_by(*cols)
        e = pl.col("v").sum().alias("total")
        result = _summarize_groupby("group_by", gb, (e,), {})
        assert len(result) <= 60
        assert result.endswith("...")
