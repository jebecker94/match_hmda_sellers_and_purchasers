# pip install polars pyarrow
import polars as pl

# ----------------------------
# Expected minimal schemas
# ----------------------------
# originations_df: one row per HMDA origination
#   ['origination_id','activity_year','lei','loan_type','loan_purpose','lien_status','state_code']
# purchases_df: one row per purchase record (from HMDA purchases or your constructed table)
#   ['purchase_id','activity_year','purchaser_type','purchaser_name']
# matches_df: one row per linkage (origination ↔ purchase)
#   ['origination_id','purchase_id','match_score']  # match_score optional

# Notes:
# - If you already pre-filtered originations to action_taken=1, great. If not, do that before passing in.
# - If your purchases table doesn’t have purchaser_type, keep that grouping out (code handles missing columns).
# - If a loan matches multiple purchases (servicing transfers), this code can either:
#     (a) dedupe to the top-scoring purchase *per origination* (default), or
#     (b) keep all (set keep_top_purchase=False).

def _safe_select(df: pl.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def prepare_unique_matches(
    matches_df: pl.DataFrame,
    keep_top_purchase: bool = True,
    score_col: str | None = "match_score",
) -> pl.DataFrame:
    if keep_top_purchase and (score_col is not None and score_col in matches_df.columns):
        # Keep highest-scoring purchase per origination
        return (
            matches_df
            .with_row_count("_ridx")  # tiebreaker
            .sort([pl.col("origination_id"), pl.col(score_col).desc(), pl.col("_ridx")])
            .unique(subset=["origination_id"], keep="first")
            .drop("_ridx")
        )
    # Otherwise, enforce uniqueness of (origination_id, purchase_id)
    return matches_df.unique(subset=["origination_id","purchase_id"])

def global_match_rates(
    originations_df: pl.DataFrame,
    purchases_df: pl.DataFrame,
    matches_df: pl.DataFrame,
    keep_top_purchase: bool = True,
) -> dict[str, pl.DataFrame]:
    # Deduplicate matches
    m = prepare_unique_matches(matches_df, keep_top_purchase=keep_top_purchase)

    # Basic counts
    n_orig = originations_df.height
    n_purc = purchases_df.height
    n_links = m.height

    # Originations matched at least once
    matched_orig = m.select("origination_id").unique().height
    # Purchases matched at least once
    matched_purc = m.select("purchase_id").unique().height

    overall = pl.DataFrame({
        "metric": [
            "originations_total",
            "purchases_total",
            "links_total",
            "originations_matched",
            "purchases_matched",
            "match_rate_originations",
            "match_rate_purchases",
            "links_per_matched_origination"
        ],
        "value": [
            n_orig,
            n_purc,
            n_links,
            matched_orig,
            matched_purc,
            matched_orig / n_orig if n_orig else None,
            matched_purc / n_purc if n_purc else None,
            n_links / matched_orig if matched_orig else None
        ],
    })

    # Join metadata needed for group rates
    mo = (
        m.join(originations_df.select(["origination_id","activity_year","lei","loan_type"]),
               on="origination_id", how="left")
         .rename({"activity_year":"orig_activity_year"})
    )

    # bring purchaser_type if available
    purchaser_cols = _safe_select(purchases_df, ["purchase_id","activity_year","purchaser_type"])
    if purchaser_cols:
        mo = mo.join(purchases_df.select(purchaser_cols), on="purchase_id", how="left") \
               .rename({"activity_year":"purc_activity_year"})

    def _rate_by(group_cols: list[str], base_df: pl.DataFrame, base_key: str):
        # base: denominator distinct units from base_df over the same grouping
        base = (
            base_df
            .group_by(group_cols)
            .agg(pl.len().alias("N_total"))
        )
        # matched: distinct base_key present in matched table
        matched = (
            mo
            .select(group_cols + [base_key])
            .unique()
            .group_by(group_cols)
            .agg(pl.len().alias("N_matched"))
        )
        out = base.join(matched, on=group_cols, how="left").with_columns(
            pl.col("N_matched").fill_null(0),
            (pl.col("N_matched") / pl.col("N_total")).alias("match_rate")
        )
        return out

    # By year (origination year)
    by_year = _rate_by(["activity_year"], originations_df, "origination_id")

    # By originator (LEI)
    by_lei = _rate_by(["lei"], originations_df, "origination_id")

    # By loan_type
    by_loan_type = _rate_by(["loan_type"], originations_df, "origination_id")

    # By purchaser_type (if available)
    if "purchaser_type" in mo.columns:
        # Denominator = purchases by purchaser_type; rate = share matched to an origination
        purc_base = purchases_df if "purchaser_type" in purchases_df.columns else None
        by_purchaser_type = _rate_by(["purchaser_type"], purc_base, "purchase_id") if purc_base is not None else None
    else:
        by_purchaser_type = None

    # Year × LEI (big table but useful)
    by_year_lei = _rate_by(["activity_year","lei"], originations_df, "origination_id")

    results = {
        "overall": overall,
        "by_year": by_year.sort("activity_year"),
        "by_lei": by_lei.sort("N_total", descending=True),
        "by_loan_type": by_loan_type.sort("N_total", descending=True),
        "by_purchaser_type": by_purchaser_type.sort("N_total", descending=True) if by_purchaser_type is not None else None,
        "by_year_lei": by_year_lei.sort(["activity_year","N_total"], descending=[False, True]),
    }
    return results

# ----------------------------
# Example usage
# ----------------------------
# results = global_match_rates(originations_df, purchases_df, matches_df, keep_top_purchase=True)
# print(results["overall"])
# print(results["by_year"].head())
# print(results["by_purchaser_type"])  # may be None if purchaser_type missing