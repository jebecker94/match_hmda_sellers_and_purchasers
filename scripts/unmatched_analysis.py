"""Profiling utilities for HMDA seller and purchaser unmatched records."""

# pip install polars pyarrow
import polars as pl


ORIGINATION_INDEX_COL = "HMDAIndex"
MATCH_ORIGINATION_COL = "HMDAIndex_s"
MATCH_PURCHASE_COL = "HMDAIndex_p"


def _has(df: pl.DataFrame, col: str) -> bool:
    """Return ``True`` when ``col`` is present in ``df``."""

    return col in df.columns


def _safe_cols(df: pl.DataFrame, cols: list[str]) -> list[str]:
    """Filter ``cols`` to those present in ``df``."""

    return [c for c in cols if c in df.columns]


def prepare_unique_matches(
    matches_df: pl.DataFrame,
    seller_index_col: str = MATCH_ORIGINATION_COL,
    purchaser_index_col: str = MATCH_PURCHASE_COL,
    score_col: str | None = "match_score",
) -> pl.DataFrame:
    """Deduplicate matches to one purchaser per seller loan when scores exist."""

    subset_cols = [seller_index_col]
    if purchaser_index_col in matches_df.columns:
        subset_cols.append(purchaser_index_col)

    if score_col and score_col in matches_df.columns:
        return (
            matches_df
            .with_row_count("_ridx")
            .sort([seller_index_col, pl.col(score_col).desc(), "_ridx"])
            .unique(subset=[seller_index_col], keep="first")
            .drop("_ridx")
        )
    return matches_df.unique(subset=subset_cols)


def classify_product_from_loan_type(df: pl.DataFrame) -> pl.DataFrame:
    """Append a human-readable product label derived from ``loan_type`` codes."""

    if not _has(df, "loan_type"):
        return df.with_columns(pl.lit(None).alias("product"))
    return df.with_columns(
        pl.when(pl.col("loan_type") == 1)
        .then(pl.lit("Conventional"))
        .when(pl.col("loan_type") == 2)
        .then(pl.lit("FHA"))
        .when(pl.col("loan_type") == 3)
        .then(pl.lit("VA"))
        .when(pl.col("loan_type") == 4)
        .then(pl.lit("USDA"))
        .otherwise(pl.lit("Other/Unknown"))
        .alias("product")
    )


def add_rural_flag(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize the HMDA rural indicator to a boolean column."""

    if _has(df, "is_rural_county"):
        source_col = "is_rural_county"
    elif _has(df, "rural_indicator"):
        source_col = "rural_indicator"
    else:
        return df.with_columns(pl.lit(None).alias("is_rural_county"))

    true_values = [1, True, "1", "Y", "y", "yes", "YES", "true", "TRUE"]
    false_values = [0, False, "0", "N", "n", "no", "NO", "false", "FALSE"]
    source_expr = pl.col(source_col)

    standardized = (
        pl.when(source_expr.is_null())
        .then(None)
        .when(source_expr.is_in(true_values))
        .then(pl.lit(True))
        .when(source_expr.is_in(false_values))
        .then(pl.lit(False))
        .otherwise(source_expr.cast(pl.Boolean, strict=False))
        .alias("is_rural_county")
    )
    return df.with_columns(standardized)


def std_mean_diff(a: pl.Series, b: pl.Series) -> float | None:
    """Compute the standardized mean difference between two numeric series."""

    try:
        a_clean = a.cast(pl.Float64, strict=False).drop_nans().drop_nulls()
        b_clean = b.cast(pl.Float64, strict=False).drop_nans().drop_nulls()
        if a_clean.len() == 0 or b_clean.len() == 0:
            return None
        ma, mb = a_clean.mean(), b_clean.mean()
        va, vb = a_clean.var(), b_clean.var()
        pooled = ((va + vb) / 2.0) ** 0.5 if va is not None and vb is not None else None
        return (ma - mb) / pooled if pooled and pooled != 0 else None
    except Exception:
        return None


def profile_unmatched_originations(
    originations_df: pl.DataFrame,
    matches_df: pl.DataFrame,
    numeric_cols: list[str] | None = None,
) -> dict[str, pl.DataFrame | None]:
    """Summarize originations that are not linked in the match crosswalk."""

    numeric_cols = numeric_cols or ["loan_amount"]
    m = prepare_unique_matches(matches_df)

    matched_orig_ids = (
        m.select(pl.col(MATCH_ORIGINATION_COL).alias(ORIGINATION_INDEX_COL)).unique()
    )
    unmatched_orig = originations_df.join(
        matched_orig_ids, on=ORIGINATION_INDEX_COL, how="anti"
    )

    orig_enriched = add_rural_flag(classify_product_from_loan_type(originations_df))
    unmatched_enriched = add_rural_flag(classify_product_from_loan_type(unmatched_orig))

    by_year_base = (
        unmatched_enriched
        .group_by("activity_year")
        .agg([
            pl.len().alias("N_unmatched"),
            *[
                pl.col(c).cast(pl.Float64, strict=False).mean().alias(f"{c}_mean")
                for c in _safe_cols(unmatched_enriched, numeric_cols)
            ],
            *[
                pl.col(c).cast(pl.Float64, strict=False).median().alias(f"{c}_median")
                for c in _safe_cols(unmatched_enriched, numeric_cols)
            ],
        ])
        .sort("activity_year")
    )

    overall_metrics = ["N_unmatched"]
    overall_values = [unmatched_enriched.height]
    for column in numeric_cols:
        if _has(unmatched_enriched, column):
            overall_metrics.extend([f"{column}_mean", f"{column}_median"])
            overall_values.append(
                unmatched_enriched.select(
                    pl.col(column).cast(pl.Float64, strict=False).mean()
                ).item()
            )
            overall_values.append(
                unmatched_enriched.select(
                    pl.col(column).cast(pl.Float64, strict=False).median()
                ).item()
            )

    overall_base = pl.DataFrame({
        "metric": overall_metrics,
        "value": overall_values,
    })

    dist_product = (
        unmatched_enriched
        .group_by("product")
        .agg(pl.len().alias("N"))
        .with_columns((pl.col("N") / pl.col("N").sum()).alias("share"))
        .sort("N", descending=True)
    ) if _has(unmatched_enriched, "product") else None

    dist_rural = (
        unmatched_enriched
        .group_by("is_rural_county")
        .agg(pl.len().alias("N"))
        .with_columns((pl.col("N") / pl.col("N").sum()).alias("share"))
        .sort("N", descending=True)
    ) if _has(unmatched_enriched, "is_rural_county") else None

    matched_orig = originations_df.join(
        matched_orig_ids, on=ORIGINATION_INDEX_COL, how="inner"
    )

    comp_rows: list[dict[str, float | str | None]] = []
    for column in numeric_cols:
        if _has(originations_df, column) and _has(unmatched_enriched, column):
            a = matched_orig.get_column(column)
            b = unmatched_enriched.get_column(column)
            smd = std_mean_diff(a, b)
            comp_rows.append(
                {
                    "variable": column,
                    "matched_mean": matched_orig.select(
                        pl.col(column).cast(pl.Float64, strict=False).mean()
                    ).item(),
                    "unmatched_mean": unmatched_enriched.select(
                        pl.col(column).cast(pl.Float64, strict=False).mean()
                    ).item(),
                    "matched_median": matched_orig.select(
                        pl.col(column).cast(pl.Float64, strict=False).median()
                    ).item(),
                    "unmatched_median": unmatched_enriched.select(
                        pl.col(column).cast(pl.Float64, strict=False).median()
                    ).item(),
                    "std_mean_diff": smd,
                }
            )

    compare_numeric = pl.DataFrame(comp_rows) if comp_rows else None

    def _mix(df: pl.DataFrame, col: str, label: str) -> pl.DataFrame | None:
        if not _has(df, col):
            return None
        return (
            df.group_by(col)
            .agg(pl.len().alias(f"N_{label}"))
            .with_columns(
                (
                    pl.col(f"N_{label}") / pl.col(f"N_{label}").sum()
                ).alias(f"share_{label}")
            )
            .sort(f"N_{label}", descending=True)
        )

    matched_enriched = add_rural_flag(classify_product_from_loan_type(matched_orig))

    mix_product = None
    if _has(matched_enriched, "product") and _has(unmatched_enriched, "product"):
        mix_product = (
            _mix(matched_enriched, "product", "matched")
            .join(
                _mix(unmatched_enriched, "product", "unmatched"),
                on="product",
                how="outer",
            )
            .fill_null(0.0)
            .sort("share_unmatched", descending=True)
        )

    mix_rural = None
    if _has(matched_enriched, "is_rural_county") and _has(unmatched_enriched, "is_rural_county"):
        mix_rural = (
            _mix(matched_enriched, "is_rural_county", "matched")
            .join(
                _mix(unmatched_enriched, "is_rural_county", "unmatched"),
                on="is_rural_county",
                how="outer",
            )
            .fill_null(0.0)
            .sort("share_unmatched", descending=True)
        )

    return {
        "summary_unmatched_overall": overall_base,
        "summary_unmatched_by_year": by_year_base,
        "dist_product_unmatched": dist_product,
        "dist_rural_unmatched": dist_rural,
        "compare_numeric_matched_vs_unmatched": compare_numeric,
        "compare_mix_product": mix_product,
        "compare_mix_rural": mix_rural,
        "unmatched_originations": unmatched_enriched.select(
            _safe_cols(
                unmatched_enriched,
                [
                    ORIGINATION_INDEX_COL,
                    "activity_year",
                    "lei",
                    "loan_type",
                    "product",
                    "loan_amount",
                    "state_code",
                    "census_tract",
                    "is_rural_county",
                ],
            )
        ),
    }


def profile_unmatched_purchases(
    purchases_df: pl.DataFrame,
    matches_df: pl.DataFrame,
    top_n: int = 25,
) -> dict[str, pl.DataFrame | None]:
    """Summarize unmatched purchase records and their concentration."""

    m = prepare_unique_matches(matches_df)
    matched_purchase_ids = (
        m.select(pl.col(MATCH_PURCHASE_COL).alias(ORIGINATION_INDEX_COL)).unique()
    )
    unmatched_pur = purchases_df.join(
        matched_purchase_ids, on=ORIGINATION_INDEX_COL, how="anti"
    )

    total_unmatched = unmatched_pur.height
    by_year = (
        unmatched_pur.group_by("activity_year")
        .agg(pl.len().alias("N_unmatched"))
        .sort("activity_year")
    )

    by_inst = None
    if _has(unmatched_pur, "respondent_name"):
        by_inst = (
            unmatched_pur
            .group_by("respondent_name")
            .agg(pl.len().alias("N_unmatched"))
            .with_columns((pl.col("N_unmatched") / total_unmatched).alias("share"))
            .sort(["N_unmatched", "respondent_name"], descending=[True, False])
            .head(top_n)
        )

    by_type = None
    if _has(unmatched_pur, "purchaser_type"):
        by_type = (
            unmatched_pur
            .group_by("purchaser_type")
            .agg(pl.len().alias("N_unmatched"))
            .with_columns((pl.col("N_unmatched") / total_unmatched).alias("share"))
            .sort("N_unmatched", descending=True)
        )

    return {
        "summary_unmatched_purchases_by_year": by_year,
        "unmatched_purchases_by_inst_top": by_inst,
        "unmatched_purchases_by_type": by_type,
        "unmatched_purchases": unmatched_pur.select(
            _safe_cols(
                unmatched_pur,
                [
                    ORIGINATION_INDEX_COL,
                    "activity_year",
                    "respondent_name",
                    "purchaser_type",
                ],
            )
        ),
    }

