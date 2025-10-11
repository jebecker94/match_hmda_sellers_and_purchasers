# pip install polars pyarrow
import polars as pl

# Expected minimal schemas (extend as available):
# originations_df:
#   ['origination_id','activity_year','lei','loan_type','loan_amount',
#    'state_code','census_tract','loan_purpose','lien_status','rural_indicator?']
#
# purchases_df:
#   ['purchase_id','activity_year','purchaser_type?','purchaser_name?']
#
# matches_df:
#   ['origination_id','purchase_id','match_score?']

# ----------------------------
# Helpers
# ----------------------------
def _has(df: pl.DataFrame, col: str) -> bool:
    return col in df.columns

def _safe_cols(df: pl.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def prepare_unique_matches(matches_df: pl.DataFrame) -> pl.DataFrame:
    # One link per origination (highest score if available)
    if _has(matches_df, "match_score"):
        return (
            matches_df
            .with_row_count("_ridx")
            .sort(["origination_id", pl.col("match_score").desc(), "_ridx"])
            .unique(subset=["origination_id"], keep="first")
            .drop("_ridx")
        )
    return matches_df.unique(subset=["origination_id","purchase_id"])

def classify_product_from_loan_type(df: pl.DataFrame) -> pl.DataFrame:
    # HMDA loan_type: 1=Conv, 2=FHA, 3=VA, 4=RHS/FSA(USDA)
    if not _has(df, "loan_type"):
        return df.with_columns(pl.lit(None).alias("product"))
    return df.with_columns(
        pl.when(pl.col("loan_type")==1).then(pl.lit("Conventional"))
         .when(pl.col("loan_type")==2).then(pl.lit("FHA"))
         .when(pl.col("loan_type")==3).then(pl.lit("VA"))
         .when(pl.col("loan_type")==4).then(pl.lit("USDA"))
         .otherwise(pl.lit("Other/Unknown"))
         .alias("product")
    )

def add_rural_flag(df: pl.DataFrame) -> pl.DataFrame:
    # If 'rural_indicator' exists, keep it; otherwise fill null
    if _has(df, "rural_indicator"):
        # ensure boolean
        return df.with_columns(pl.col("rural_indicator").cast(pl.Boolean, strict=False))
    return df.with_columns(pl.lit(None).alias("rural_indicator"))

def std_mean_diff(a: pl.Series, b: pl.Series) -> float | None:
    # Standardized mean difference for numeric columns (NaNs ignored)
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

# ----------------------------
# Core profiling functions
# ----------------------------
def profile_unmatched_originations(
    originations_df: pl.DataFrame,
    matches_df: pl.DataFrame,
    numeric_cols: list[str] = ["loan_amount"],
) -> dict[str, pl.DataFrame]:
    """
    Returns:
      - summary_unmatched: counts and basic stats overall + by year
      - dist_product: product mix (Conv/FHA/VA/USDA) for unmatched
      - dist_rural: rural vs non-rural for unmatched (if available)
      - compare_matched_unmatched: side-by-side stats and standardized mean diffs
    """
    m = prepare_unique_matches(matches_df)

    # Identify unmatched originations
    matched_orig_ids = m.select("origination_id").unique()
    unmatched_orig = originations_df.join(matched_orig_ids, on="origination_id", how="anti")

    # Prepare enriched frames
    orig_enriched = add_rural_flag(classify_product_from_loan_type(originations_df))
    unmatched_enriched = add_rural_flag(classify_product_from_loan_type(unmatched_orig))

    # Summaries
    by_year_base = (
        unmatched_enriched
        .group_by("activity_year")
        .agg([
            pl.len().alias("N_unmatched"),
            *[pl.col(c).cast(pl.Float64, strict=False).mean().alias(f"{c}_mean") for c in _safe_cols(unmatched_enriched, numeric_cols)],
            *[pl.col(c).cast(pl.Float64, strict=False).median().alias(f"{c}_median") for c in _safe_cols(unmatched_enriched, numeric_cols)],
        ])
        .sort("activity_year")
    )

    overall_base = pl.DataFrame({
        "metric": ["N_unmatched"] + [f"{c}_mean" for c in numeric_cols if _has(unmatched_enriched, c)] + [f"{c}_median" for c in numeric_cols if _has(unmatched_enriched, c)],
        "value": [unmatched_enriched.height] + \
                 [unmatched_enriched.select(pl.col(c).cast(pl.Float64, strict=False).mean()).item() for c in numeric_cols if _has(unmatched_enriched, c)] + \
                 [unmatched_enriched.select(pl.col(c).cast(pl.Float64, strict=False).median()).item() for c in numeric_cols if _has(unmatched_enriched, c)],
    })

    # Distributions
    dist_product = (
        unmatched_enriched
        .group_by("product")
        .agg(pl.len().alias("N"))
        .with_columns((pl.col("N")/pl.col("N").sum()).alias("share"))
        .sort("N", descending=True)
    ) if _has(unmatched_enriched, "product") else None

    dist_rural = (
        unmatched_enriched
        .group_by("rural_indicator")
        .agg(pl.len().alias("N"))
        .with_columns((pl.col("N")/pl.col("N").sum()).alias("share"))
        .sort("N", descending=True)
    ) if _has(unmatched_enriched, "rural_indicator") else None

    # Matched vs Unmatched comparison (originations side)
    matched_orig = originations_df.join(matched_orig_ids, on="origination_id", how="inner")

    # Numeric comparisons with standardized mean differences
    comp_rows = []
    for c in numeric_cols:
        if _has(originations_df, c):
            a = matched_orig.get_column(c)
            b = unmatched_enriched.get_column(c) if _has(unmatched_enriched, c) else None
            if b is not None:
                smd = std_mean_diff(a, b)
                comp_rows.append({
                    "variable": c,
                    "matched_mean": matched_orig.select(pl.col(c).cast(pl.Float64, strict=False).mean()).item(),
                    "unmatched_mean": unmatched_enriched.select(pl.col(c).cast(pl.Float64, strict=False).mean()).item(),
                    "matched_median": matched_orig.select(pl.col(c).cast(pl.Float64, strict=False).median()).item(),
                    "unmatched_median": unmatched_enriched.select(pl.col(c).cast(pl.Float64, strict=False).median()).item(),
                    "std_mean_diff": smd
                })

    compare_numeric = pl.DataFrame(comp_rows) if comp_rows else None

    # Categorical mixes (product, rural)
    def _mix(df: pl.DataFrame, col: str, label: str) -> pl.DataFrame | None:
        if not _has(df, col):
            return None
        return (
            df.group_by(col)
              .agg(pl.len().alias(f"N_{label}"))
              .with_columns((pl.col(f"N_{label}") / pl.col(f"N_{label}").sum()).alias(f"share_{label}"))
              .sort(f"N_{label}", descending=True)
        )

    matched_enriched = add_rural_flag(classify_product_from_loan_type(matched_orig))

    mix_product = None
    if _has(matched_enriched, "product") and _has(unmatched_enriched, "product"):
        mix_product = (
            _mix(matched_enriched, "product", "matched")
            .join(_mix(unmatched_enriched, "product", "unmatched"), on="product", how="outer")
            .fill_null(0.0)
            .sort("share_unmatched", descending=True)
        )

    mix_rural = None
    if _has(matched_enriched, "rural_indicator") and _has(unmatched_enriched, "rural_indicator"):
        mix_rural = (
            _mix(matched_enriched, "rural_indicator", "matched")
            .join(_mix(unmatched_enriched, "rural_indicator", "unmatched"), on="rural_indicator", how="outer")
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
        "unmatched_originations": unmatched_enriched.select(_safe_cols(unmatched_enriched, [
            "origination_id","activity_year","lei","loan_type","product","loan_amount","state_code","census_tract","rural_indicator"
        ]))
    }

def profile_unmatched_purchases(
    purchases_df: pl.DataFrame,
    matches_df: pl.DataFrame,
    top_n: int = 25,
) -> dict[str, pl.DataFrame]:
    """
    Returns:
      - unmatched_purchases_by_inst: top purchasers by unmatched count/share
      - unmatched_purchases_by_type: unmatched shares by purchaser_type (if available)
      - summary_unmatched_purchases_by_year
    """
    m = prepare_unique_matches(matches_df)
    matched_purchase_ids = m.select("purchase_id").unique()
    unmatched_pur = purchases_df.join(matched_purchase_ids, on="purchase_id", how="anti")

    total_unmatched = unmatched_pur.height
    by_year = (
        unmatched_pur.group_by("activity_year")
        .agg(pl.len().alias("N_unmatched"))
        .sort("activity_year")
    )

    # Purchaser name concentration
    by_inst = None
    if _has(unmatched_pur, "purchaser_name"):
        by_inst = (
            unmatched_pur
            .group_by("purchaser_name")
            .agg(pl.len().alias("N_unmatched"))
            .with_columns((pl.col("N_unmatched")/total_unmatched).alias("share"))
            .sort(["N_unmatched","purchaser_name"], descending=[True, False])
            .head(top_n)
        )

    # Purchaser type concentration
    by_type = None
    if _has(unmatched_pur, "purchaser_type"):
        by_type = (
            unmatched_pur
            .group_by("purchaser_type")
            .agg(pl.len().alias("N_unmatched"))
            .with_columns((pl.col("N_unmatched")/total_unmatched).alias("share"))
            .sort("N_unmatched", descending=True)
        )

    return {
        "summary_unmatched_purchases_by_year": by_year,
        "unmatched_purchases_by_inst_top": by_inst,
        "unmatched_purchases_by_type": by_type,
        "unmatched_purchases": unmatched_pur.select(_safe_cols(unmatched_pur, [
            "purchase_id","activity_year","purchaser_name","purchaser_type"
        ]))
    }

# ----------------------------
# Example usage
# ----------------------------
# originations_df, purchases_df, matches_df = ...
# orig_profile = profile_unmatched_originations(originations_df, matches_df, numeric_cols=["loan_amount"])
# purc_profile = profile_unmatched_purchases(purchases_df, matches_df, top_n=25)
# print(orig_profile["summary_unmatched_overall"])
# print(orig_profile["compare_numeric_matched_vs_unmatched"])
# print(purc_profile["unmatched_purchases_by_inst_top"])