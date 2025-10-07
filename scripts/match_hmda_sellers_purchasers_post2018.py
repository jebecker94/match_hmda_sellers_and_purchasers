#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:45:24 2023
Last updated on: Sat Feb 11 10:45:24 2023
@author: Jonathan E. Becker (jebecker3@wisc.edu)
"""

# Import Packages
import logging

import pandas as pd
import polars as pl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import HMDALoader
import config
from matching_support_functions import *


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Polars helper utilities


def _build_polars_filter_expression(column, operator, value):
    """Translate a tuple-based filter into a Polars expression.

    This helper mirrors the ``added_filters`` arguments accepted by the
    pandas-oriented :func:`matching_support_functions.load_data` routine.
    Only the comparison operators that the matching scripts currently rely on
    are implemented (``==``, ``!=``, ``in``, ``not in``, ``>=``, ``<=``, ``>``,
    and ``<``).
    """

    col_expr = pl.col(column)
    if operator == "==":
        return col_expr == value
    if operator == "!=":
        return col_expr != value
    if operator == "in":
        return col_expr.is_in(value)
    if operator == "not in":
        return ~col_expr.is_in(value)
    if operator == ">=":
        return col_expr >= value
    if operator == "<=":
        return col_expr <= value
    if operator == ">":
        return col_expr > value
    if operator == "<":
        return col_expr < value
    raise ValueError(f"Unsupported filter operator '{operator}' for Polars")


def _suffix_polars_columns(df: pl.DataFrame, suffix: str, exclude: set[str]) -> pl.DataFrame:
    """Append a suffix to all columns except those listed in ``exclude``."""

    rename_map = {
        column: f"{column}{suffix}"
        for column in df.columns
        if column not in exclude
    }
    return df.rename(rename_map)


def load_data_polars(
    data_folder,
    min_year: int = 2018,
    max_year: int = 2023,
    added_filters=None,
):
    """Load HMDA data as a Polars DataFrame.

    This is the Polars analogue to :func:`matching_support_functions.load_data`.
    The pandas helper is defined in ``scripts/matching_support_functions.py``
    and serves as a reference for the filtering logic implemented here.
    """

    hmda_filters = [("action_taken", "in", [1, 6])]
    if added_filters is not None:
        hmda_filters += list(added_filters)

    frames: list[pl.LazyFrame] = []
    for year in range(min_year, max_year + 1):
        file = HMDALoader.get_hmda_files(
            data_folder,
            min_year=year,
            max_year=year,
            extension="parquet",
        )[0]
        hmda_columns = get_match_columns(file)
        lazy = pl.scan_parquet(str(file), columns=hmda_columns)
        expr = None
        for column, operator, value in hmda_filters:
            filter_expr = _build_polars_filter_expression(column, operator, value)
            expr = filter_expr if expr is None else expr & filter_expr
        if expr is not None:
            lazy = lazy.filter(expr)
        lazy = lazy.filter(
            (~pl.col("purchaser_type").is_in([1, 2, 3, 4])) | (pl.col("action_taken") == 6)
        )
        frames.append(lazy)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames).collect()


def convert_numerics_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Convert string-coded HMDA numerics to numeric Polars dtypes."""

    exempt_cols = [
        "combined_loan_to_value_ratio",
        "interest_rate",
        "rate_spread",
        "loan_term",
        "prepayment_penalty_term",
        "intro_rate_period",
        "income",
        "multifamily_affordable_units",
        "property_value",
        "total_loan_costs",
        "total_points_and_fees",
        "origination_charges",
        "discount_points",
        "lender_credits",
    ]

    for column in exempt_cols:
        if column in df.columns:
            df = df.with_columns(
                pl.when(pl.col(column) == "Exempt")
                .then(-99999)
                .otherwise(pl.col(column))
                .cast(pl.Float64, strict=False)
                .alias(column)
            )

    if "total_units" in df.columns:
        unit_mapping = {
            "5-24": 5,
            "25-49": 6,
            "50-99": 7,
            "100-149": 8,
            ">149": 9,
        }
        df = df.with_columns(
            pl.col("total_units")
            .replace(unit_mapping)
            .cast(pl.Float64, strict=False)
            .alias("total_units")
        )

    for age_column in ["applicant_age", "co_applicant_age"]:
        if age_column in df.columns:
            age_mapping = {
                "<25": 1,
                "25-34": 2,
                "35-44": 3,
                "45-54": 4,
                "55-64": 5,
                "65-74": 6,
                ">74": 7,
            }
            df = df.with_columns(
                pl.col(age_column)
                .replace(age_mapping)
                .cast(pl.Float64, strict=False)
                .alias(age_column)
            )

    for column in ["applicant_age_above_62", "co_applicant_age_above_62"]:
        if column in df.columns:
            df = df.with_columns(
                pl.when(pl.col(column).str.to_lowercase() == "no")
                .then(0)
                .when(pl.col(column).str.to_lowercase() == "yes")
                .then(1)
                .when(pl.col(column).str.to_lowercase() == "na")
                .then(None)
                .otherwise(pl.col(column))
                .cast(pl.Float64, strict=False)
                .alias(column)
            )

    if "debt_to_income_ratio" in df.columns:
        dti_mapping = {
            "<20%": 10,
            "20%-<30%": 20,
            "30%-<36%": 30,
            "50%-60%": 50,
            ">60%": 60,
            "Exempt": -99999,
        }
        df = df.with_columns(
            pl.col("debt_to_income_ratio")
            .replace(dti_mapping)
            .cast(pl.Float64, strict=False)
            .alias("debt_to_income_ratio")
        )

    if "conforming_loan_limit" in df.columns:
        conforming_mapping = {"NC": 0, "C": 1, "U": 1111, "NA": -1111}
        df = df.with_columns(
            pl.col("conforming_loan_limit")
            .replace(conforming_mapping)
            .cast(pl.Float64, strict=False)
            .alias("conforming_loan_limit")
        )

    numeric_columns = [
        "activity_year",
        "loan_type",
        "loan_purpose",
        "occupancy_type",
        "loan_amount",
        "action_taken",
        "msa_md",
        "county_code",
        "applicant_race_1",
        "applicant_race_2",
        "applicant_race_3",
        "applicant_race_4",
        "applicant_race_5",
        "co_applicant_race_1",
        "co_applicant_race_2",
        "co_applicant_race_3",
        "co_applicant_race_4",
        "co_applicant_race_5",
        "applicant_sex",
        "co_applicant_sex",
        "income",
        "purchaser_type",
        "denial_reason_1",
        "denial_reason_2",
        "denial_reason_3",
        "edit_status",
        "sequence_number",
        "rate_spread",
        "tract_population",
        "tract_minority_population_percent",
        "ffiec_msa_md_median_family_income",
        "tract_to_msa_income_percentage",
        "tract_owner_occupied_units",
        "tract_one_to_four_family_homes",
        "tract_median_age_of_housing_units",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df = df.with_columns(
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
            )

    return df


def replace_missing_values_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Replace sentinel missing values with ``None`` using Polars."""

    if {"total_loan_costs", "total_points_and_fees", "origination_charges", "discount_points", "lender_credits"}.issubset(
        df.columns
    ):
        df = df.with_columns(
            (
                (pl.col("total_loan_costs") == 1111)
                & (pl.col("total_points_and_fees") == 1111)
                & (pl.col("origination_charges") == 1111)
                & (pl.col("discount_points") == 1111)
                & (pl.col("lender_credits") == 1111)
            )
            .cast(pl.Int8)
            .alias("i_ExemptFromFeesStrict")
        )
        df = df.with_columns(
            (
                (pl.col("total_loan_costs") == 1111)
                | (pl.col("total_points_and_fees") == 1111)
                | (pl.col("origination_charges") == 1111)
                | (pl.col("discount_points") == 1111)
                | (pl.col("lender_credits") == 1111)
            )
            .cast(pl.Int8)
            .alias("i_ExemptFromFeesWeak")
        )

    replace_columns = [
        "conforming_loan_limit",
        "construction_method",
        "income",
        "total_units",
        "lien_status",
        "multifamily_affordable_units",
        "total_loan_costs",
        "total_points_and_fees",
        "discount_points",
        "lender_credits",
        "origination_charges",
        "interest_rate",
        "intro_rate_period",
        "loan_term",
        "property_value",
        "total_units",
        "balloon_payment",
        "interest_only_payment",
        "negative_amortization",
        "open_end_line_of_credit",
        "other_nonamortizing_features",
        "prepayment_penalty_term",
        "reverse_mortgage",
        "business_or_commercial_purpose",
        "manufactured_home_land_property_",
        "manufactured_home_secured_proper",
    ]

    for column in replace_columns:
        if column in df.columns:
            df = df.with_columns(
                pl.when(
                    pl.col(column).is_in([-1111, 1111, 99999, -99999])
                    | (pl.col(column) <= 0)
                )
                .then(None)
                .otherwise(pl.col(column))
                .alias(column)
            )

    if {"intro_rate_period", "loan_term"}.issubset(df.columns):
        df = df.with_columns(
            pl.when(pl.col("intro_rate_period") == pl.col("loan_term"))
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("intro_rate_period"))
            .alias("intro_rate_period")
        )

    return df


def split_sellers_and_purchasers_polars(
    df: pl.DataFrame,
    crosswalk_folder: str,
    match_round: int = 1,
    file_suffix: str | None = None,
):
    """Split HMDA records into seller and purchaser subsets using Polars.

    The pandas equivalent lives in ``scripts/matching_support_functions.py`` as
    :func:`split_sellers_and_purchasers`.
    """

    if match_round > 1:
        suffix = file_suffix or ""
        crosswalk_path = (
            f"{crosswalk_folder}/hmda_seller_purchaser_matches_round{match_round-1}{suffix}.parquet"
        )
        cw = pl.read_parquet(crosswalk_path)
        df = df.join(
            cw.select(pl.col("HMDAIndex_s").alias("HMDAIndex")),
            on="HMDAIndex",
            how="anti",
        )
        df = df.join(
            cw.select(pl.col("HMDAIndex_p").alias("HMDAIndex")),
            on="HMDAIndex",
            how="anti",
        )

    purchasers = df.filter(pl.col("action_taken") == 6)
    sellers = df.filter(pl.col("action_taken") == 1)

    return sellers, purchasers


def numeric_matches_polars(
    df: pl.DataFrame,
    match_tolerances,
    drop_differences: bool = True,
) -> pl.DataFrame:
    """Filter candidate pairs using numeric tolerances in Polars."""

    for column, tolerance in match_tolerances.items():
        seller_col = f"{column}_s"
        purchaser_col = f"{column}_p"
        difference_col = f"{column}_difference"
        if seller_col in df.columns and purchaser_col in df.columns:
            start_obs = df.height
            df = df.with_columns(
                (pl.col(seller_col) - pl.col(purchaser_col)).alias(difference_col)
            )
            df = df.filter(
                pl.col(difference_col).abs() <= tolerance
                | pl.col(difference_col).is_null()
            )
            logger.debug(
                "Matching on %s drops %d observations",
                column,
                start_obs - df.height,
            )
            if drop_differences and difference_col in df.columns:
                df = df.drop(difference_col)

    return df


def weak_numeric_matches_polars(
    df: pl.DataFrame,
    match_tolerances,
    drop_differences: bool = True,
) -> pl.DataFrame:
    """Allow slight numeric mismatches while keeping best Polars matches."""

    for column, tolerance in match_tolerances.items():
        seller_col = f"{column}_s"
        purchaser_col = f"{column}_p"
        difference_col = f"{column}_difference"
        min_s_col = f"min_{column}_difference_s"
        min_p_col = f"min_{column}_difference_p"
        if seller_col in df.columns and purchaser_col in df.columns:
            start_obs = df.height
            df = df.with_columns(
                (pl.col(seller_col) - pl.col(purchaser_col)).alias(difference_col)
            )
            df = df.with_columns(
                pl.col(difference_col)
                .abs()
                .over("HMDAIndex_s")
                .min()
                .alias(min_s_col),
                pl.col(difference_col)
                .abs()
                .over("HMDAIndex_p")
                .min()
                .alias(min_p_col),
            )
            df = df.filter(
                (pl.col(difference_col).abs() <= tolerance)
                | (pl.col(min_s_col) > 0)
                | pl.col(min_s_col).is_null()
            )
            df = df.filter(
                (pl.col(difference_col).abs() <= tolerance)
                | (pl.col(min_p_col) > 0)
                | pl.col(min_p_col).is_null()
            )
            logger.debug(
                "Matching weakly on %s drops %d observations",
                column,
                start_obs - df.height,
            )
            if drop_differences:
                df = df.drop([difference_col, min_s_col, min_p_col], strict=False)

    return df


def perform_fee_matches_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Compute fee match counters using Polars expressions."""

    fee_columns = [
        "total_loan_costs",
        "total_points_and_fees",
        "origination_charges",
        "discount_points",
        "lender_credits",
    ]

    existing_fee_columns = [
        column for column in fee_columns if f"{column}_s" in df.columns and f"{column}_p" in df.columns
    ]

    if not existing_fee_columns:
        return df

    fee_match_exprs = [
        ((pl.col(f"{column}_s") == pl.col(f"{column}_p")) & pl.col(f"{column}_s").is_not_null()).cast(pl.Int32)
        for column in existing_fee_columns
    ]
    nonmissing_s_exprs = [pl.col(f"{column}_s").is_not_null().cast(pl.Int32) for column in existing_fee_columns]
    nonmissing_p_exprs = [pl.col(f"{column}_p").is_not_null().cast(pl.Int32) for column in existing_fee_columns]
    generous_exprs = [
        ((pl.col(f"{var1}_s") == pl.col(f"{var2}_p")) & pl.col(f"{var1}_s").is_not_null()).cast(pl.Int32)
        for var1 in existing_fee_columns
        for var2 in existing_fee_columns
    ]

    df = df.with_columns(
        pl.sum_horizontal(*fee_match_exprs).alias("NumberFeeMatches"),
        pl.sum_horizontal(*nonmissing_s_exprs).alias("NumberNonmissingFees_s"),
        pl.sum_horizontal(*nonmissing_p_exprs).alias("NumberNonmissingFees_p"),
        pl.max_horizontal(*generous_exprs).alias("i_GenerousFeeMatch"),
    )

    return df


def keep_uniques_polars(df: pl.DataFrame, one_to_one: bool = True) -> pl.DataFrame:
    """Restrict matches to unique seller/purchaser combinations in Polars."""

    df = df.with_columns(
        pl.len().over("HMDAIndex_s").alias("count_index_s"),
        pl.len().over("HMDAIndex_p").alias("count_index_p"),
    )

    logger.debug(
        "Match cardinality counts:\n%s",
        df.select(["count_index_s", "count_index_p"]).to_pandas().value_counts(),
    )

    df = df.filter(pl.col("count_index_p") == 1)

    if one_to_one:
        df = df.filter(pl.col("count_index_s") == 1)
    else:
        df = df.with_columns(
            (pl.col("purchaser_type_p") > 4).cast(pl.Int8).alias("temp"),
            pl.col("temp").max().over("HMDAIndex_s").alias("i_LoanHasSecondarySale"),
        )
        df = df.filter(
            (pl.col("count_index_s") == 1)
            | (
                (pl.col("count_index_s") == 2)
                & (pl.col("i_LoanHasSecondarySale") == 1)
            )
        )
        df = df.drop(["temp", "i_LoanHasSecondarySale"], strict=False)

    df = df.drop(["count_index_s", "count_index_p"], strict=False)

    return df


def match_sex_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Remove seller/purchaser pairs that conflict on reported sex codes."""

    for sex_column in ["applicant_sex", "co_applicant_sex"]:
        seller_col = f"{sex_column}_s"
        purchaser_col = f"{sex_column}_p"
        if seller_col in df.columns and purchaser_col in df.columns:
            df = df.filter(~((pl.col(seller_col) == 1) & pl.col(purchaser_col).is_in([2, 3, 5, 6])))
            df = df.filter(~((pl.col(seller_col) == 2) & pl.col(purchaser_col).is_in([1, 3, 5, 6])))
            df = df.filter(~((pl.col(seller_col) == 3) & pl.col(purchaser_col).is_in([1, 2, 5, 6])))
            df = df.filter(~((pl.col(seller_col) == 5) & pl.col(purchaser_col).is_in([1, 2, 3, 6])))
            df = df.filter(~((pl.col(seller_col) == 6) & pl.col(purchaser_col).is_in([1, 2, 3, 5])))

    return df


def match_age_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Drop candidate pairs with incompatible applicant ages using Polars."""

    if {"applicant_age_s", "applicant_age_p"}.issubset(df.columns):
        df = df.filter(
            (pl.col("applicant_age_s") == pl.col("applicant_age_p"))
            | pl.col("applicant_age_s").is_in([8888, 9999])
            | pl.col("applicant_age_p").is_in([8888, 9999])
        )

    if {"co_applicant_age_s", "co_applicant_age_p"}.issubset(df.columns):
        df = df.filter(
            (pl.col("co_applicant_age_s") == pl.col("co_applicant_age_p"))
            | pl.col("co_applicant_age_s").is_in([8888, 9999])
            | pl.col("co_applicant_age_p").is_in([8888, 9999])
        )
        df = df.filter(
            (pl.col("co_applicant_age_s") != 9999)
            | pl.col("co_applicant_age_p").is_in([8888, 9999])
        )
        df = df.filter(
            (pl.col("co_applicant_age_p") != 9999)
            | pl.col("co_applicant_age_s").is_in([8888, 9999])
        )

    return df


def match_race_polars(df: pl.DataFrame, strict: bool = False) -> pl.DataFrame:
    """Apply race consistency rules with Polars expressions."""

    race_columns = ["applicant_race", "co_applicant_race"]
    for race_column in race_columns:
        for race_number in range(1, 6):
            seller_col = f"{race_column}_{race_number}_s"
            purchaser_col = f"{race_column}_{race_number}_p"
            replacement_map = {21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2}
            other_map = {41: 4, 42: 4, 43: 4, 44: 4}
            if seller_col in df.columns:
                df = df.with_columns(pl.col(seller_col).replace(replacement_map).alias(seller_col))
                df = df.with_columns(pl.col(seller_col).replace(other_map).alias(seller_col))
            if purchaser_col in df.columns:
                df = df.with_columns(pl.col(purchaser_col).replace(replacement_map).alias(purchaser_col))
                df = df.with_columns(pl.col(purchaser_col).replace(other_map).alias(purchaser_col))

    for race_number in range(1, 7):
        df = df.filter(
            (pl.col("applicant_race_1_s") != race_number)
            | pl.col("applicant_race_1_p").is_in([race_number, 7, 8])
            | (pl.col("applicant_race_2_p") == race_number)
            | (pl.col("applicant_race_3_p") == race_number)
            | (pl.col("applicant_race_4_p") == race_number)
            | (pl.col("applicant_race_5_p") == race_number)
        )
        df = df.filter(
            (pl.col("applicant_race_1_p") != race_number)
            | pl.col("applicant_race_1_s").is_in([race_number, 7, 8])
            | (pl.col("applicant_race_2_s") == race_number)
            | (pl.col("applicant_race_3_s") == race_number)
            | (pl.col("applicant_race_4_s") == race_number)
            | (pl.col("applicant_race_5_s") == race_number)
        )

    df = df.filter(
        (pl.col("co_applicant_race_1_s") != 8)
        | pl.col("co_applicant_race_1_p").is_in([7, 8])
    )
    df = df.filter(
        (pl.col("co_applicant_race_1_p") != 8)
        | pl.col("co_applicant_race_1_s").is_in([7, 8])
    )

    for race_number in range(1, 7):
        df = df.filter(
            (pl.col("co_applicant_race_1_s") != race_number)
            | pl.col("co_applicant_race_1_p").is_in([race_number, 7, 8])
            | (pl.col("co_applicant_race_2_p") == race_number)
            | (pl.col("co_applicant_race_3_p") == race_number)
            | (pl.col("co_applicant_race_4_p") == race_number)
            | (pl.col("co_applicant_race_5_p") == race_number)
        )
        df = df.filter(
            (pl.col("co_applicant_race_1_p") != race_number)
            | pl.col("co_applicant_race_1_s").is_in([race_number, 7, 8])
            | (pl.col("co_applicant_race_2_s") == race_number)
            | (pl.col("co_applicant_race_3_s") == race_number)
            | (pl.col("co_applicant_race_4_s") == race_number)
            | (pl.col("co_applicant_race_5_s") == race_number)
        )

    if strict:
        df = df.filter(
            (pl.col("applicant_race_1_s") == pl.col("applicant_race_1_p"))
            | pl.col("applicant_race_1_s").is_in([7, 8])
            | pl.col("applicant_race_1_p").is_in([7, 8])
        )
        df = df.filter(
            (pl.col("co_applicant_race_1_s") == pl.col("co_applicant_race_1_p"))
            | pl.col("co_applicant_race_1_s").is_in([7, 8])
            | pl.col("co_applicant_race_1_p").is_in([7, 8])
        )

    return df


def match_ethnicity_polars(df: pl.DataFrame, strict: bool = False) -> pl.DataFrame:
    """Apply ethnicity matching rules using Polars expressions."""

    ethnicity_columns = ["applicant_ethnicity", "co_applicant_ethnicity"]
    for ethnicity_column in ethnicity_columns:
        for ethnicity_number in range(1, 6):
            seller_col = f"{ethnicity_column}_{ethnicity_number}_s"
            purchaser_col = f"{ethnicity_column}_{ethnicity_number}_p"
            replacement_map = {11: 1, 12: 1, 13: 1, 14: 1}
            if seller_col in df.columns:
                df = df.with_columns(pl.col(seller_col).replace(replacement_map).alias(seller_col))
            if purchaser_col in df.columns:
                df = df.with_columns(pl.col(purchaser_col).replace(replacement_map).alias(purchaser_col))

    for ethnicity_number in range(1, 4):
        df = df.filter(
            (pl.col("applicant_ethnicity_1_s") != ethnicity_number)
            | pl.col("applicant_ethnicity_1_p").is_in([ethnicity_number, 4, 5])
            | (pl.col("applicant_ethnicity_2_p") == ethnicity_number)
            | (pl.col("applicant_ethnicity_3_p") == ethnicity_number)
            | (pl.col("applicant_ethnicity_4_p") == ethnicity_number)
            | (pl.col("applicant_ethnicity_5_p") == ethnicity_number)
        )
        df = df.filter(
            (pl.col("applicant_ethnicity_1_p") != ethnicity_number)
            | pl.col("applicant_ethnicity_1_s").is_in([ethnicity_number, 4, 5])
            | (pl.col("applicant_ethnicity_2_s") == ethnicity_number)
            | (pl.col("applicant_ethnicity_3_s") == ethnicity_number)
            | (pl.col("applicant_ethnicity_4_s") == ethnicity_number)
            | (pl.col("applicant_ethnicity_5_s") == ethnicity_number)
        )

    df = df.filter(
        (pl.col("co_applicant_ethnicity_1_s") != 5)
        | pl.col("co_applicant_ethnicity_1_p").is_in([4, 5])
    )
    df = df.filter(
        (pl.col("co_applicant_ethnicity_1_p") != 5)
        | pl.col("co_applicant_ethnicity_1_s").is_in([4, 5])
    )

    for ethnicity_number in range(1, 4):
        df = df.filter(
            (pl.col("co_applicant_ethnicity_1_s") != ethnicity_number)
            | pl.col("co_applicant_ethnicity_1_p").is_in([ethnicity_number, 4, 5])
            | (pl.col("co_applicant_ethnicity_2_p") == ethnicity_number)
            | (pl.col("co_applicant_ethnicity_3_p") == ethnicity_number)
            | (pl.col("co_applicant_ethnicity_4_p") == ethnicity_number)
            | (pl.col("co_applicant_ethnicity_5_p") == ethnicity_number)
        )
        df = df.filter(
            (pl.col("co_applicant_ethnicity_1_p") != ethnicity_number)
            | pl.col("co_applicant_ethnicity_1_s").is_in([ethnicity_number, 4, 5])
            | (pl.col("co_applicant_ethnicity_2_s") == ethnicity_number)
            | (pl.col("co_applicant_ethnicity_3_s") == ethnicity_number)
            | (pl.col("co_applicant_ethnicity_4_s") == ethnicity_number)
            | (pl.col("co_applicant_ethnicity_5_s") == ethnicity_number)
        )

    if strict:
        df = df.filter(
            (pl.col("applicant_ethnicity_1_s") == pl.col("applicant_ethnicity_1_p"))
            | pl.col("applicant_ethnicity_1_s").is_in([4])
            | pl.col("applicant_ethnicity_1_p").is_in([4])
        )
        df = df.filter(
            (pl.col("co_applicant_ethnicity_1_s") == pl.col("co_applicant_ethnicity_1_p"))
            | pl.col("co_applicant_ethnicity_1_s").is_in([4])
            | pl.col("co_applicant_ethnicity_1_p").is_in([4])
        )

    return df


def save_crosswalk_polars(
    df: pl.DataFrame,
    save_folder: str,
    match_round: int = 1,
    file_suffix: str | None = None,
) -> None:
    """Persist Polars match results to a parquet crosswalk."""

    suffix = file_suffix or ""
    crosswalk_frames: list[pl.DataFrame] = []

    if match_round > 1:
        previous_path = f"{save_folder}/hmda_seller_purchaser_matches_round{match_round-1}{suffix}.parquet"
        crosswalk_frames.append(pl.read_parquet(previous_path))

    current = df.select(["HMDAIndex_s", "HMDAIndex_p"]).with_columns(
        pl.lit(match_round).alias("match_round")
    )
    crosswalk_frames.append(current)

    combined = pl.concat(crosswalk_frames)
    logger.info(
        "Crosswalk match counts by round:\n%s",
        combined.groupby("match_round").count().to_pandas(),
    )

    output_path = f"{save_folder}/hmda_seller_purchaser_matches_round{match_round}{suffix}.parquet"
    pq.write_table(combined.to_arrow(), output_path)


def match_hmda_sellers_purchasers_round1_polars(
    data_folder,
    save_folder,
    min_year: int = 2018,
    max_year: int = 2023,
    file_suffix: str | None = None,
):
    """Polars implementation of the first seller/purchaser match round."""

    frames: list[pl.DataFrame] = []
    match_columns = [
        "loan_type",
        "loan_amount",
        "census_tract",
        "occupancy_type",
        "loan_purpose",
    ]

    for year in range(min_year, max_year + 1):
        logger.info("Matching HMDA sellers and purchasers for year: %s (Polars)", year)
        df_year = load_data_polars(data_folder, min_year=year, max_year=year)
        df_year = convert_numerics_polars(df_year)
        df_year = replace_missing_values_polars(df_year)
        df_year = df_year.drop_nulls(subset=[col for col in match_columns if col in df_year.columns])
        if "census_tract" in df_year.columns:
            df_year = df_year.filter(~pl.col("census_tract").is_in(["", "NA"]))

        sellers, purchasers = split_sellers_and_purchasers_polars(df_year, save_folder)
        sellers = _suffix_polars_columns(sellers, "_s", set(match_columns))
        purchasers = _suffix_polars_columns(purchasers, "_p", set(match_columns))
        df_year = sellers.join(purchasers, on=match_columns, how="inner")

        match_tolerances = {"income": 1, "interest_rate": 0.0625}
        df_year = numeric_matches_polars(df_year, match_tolerances)

        weak_tolerances = {"interest_rate": 0.01}
        df_year = weak_numeric_matches_polars(df_year, weak_tolerances)

        df_year = perform_fee_matches_polars(df_year)
        df_year = df_year.filter(
            (pl.col("NumberFeeMatches") >= 1)
            | (pl.col("NumberNonmissingFees_s") == 0)
            | (pl.col("NumberNonmissingFees_p") == 0)
        )

        df_year = keep_uniques_polars(df_year)

        df_year = match_age_polars(df_year)
        df_year = match_sex_polars(df_year)
        df_year = match_race_polars(df_year)
        df_year = match_ethnicity_polars(df_year)

        match_tolerances = {
            "conforming_loan_limit": 0,
            "construction_method": 0,
            "discount_points": 5,
            "income": 1,
            "interest_rate": 0.0625,
            "intro_rate_period": 6,
            "lender_credits": 5,
            "lien_status": 0,
            "loan_term": 12,
            "open_end_line_of_credit": 0,
            "origination_charges": 5,
            "property_value": 20000,
            "total_units": 0,
            "applicant_age_above_62": 0,
            "co_applicant_age_above_62": 0,
        }
        df_year = numeric_matches_polars(df_year, match_tolerances)

        required_drop_columns = {
            "interest_rate_s",
            "interest_rate_p",
            "income_s",
            "property_value_s",
            "HMDAIndex_s",
        }
        if required_drop_columns.issubset(df_year.columns):
            drop_condition = (
                ((pl.col("interest_rate_s") - pl.col("interest_rate_p")).abs() >= 0.005)
                | pl.col("income_s").is_null()
                | pl.col("interest_rate_s").is_null()
                | pl.col("property_value_s").is_null()
            )
            df_year = df_year.with_columns(drop_condition.alias("i_DropObservation"))
            df_year = df_year.with_columns(
                pl.col("i_DropObservation").max().over("HMDAIndex_s").alias("i_DropSale")
            )
            df_year = df_year.filter(pl.col("i_DropSale") != 1)
            df_year = df_year.drop(["i_DropObservation", "i_DropSale"], strict=False)

        frames.append(df_year)

    if not frames:
        return

    matches = pl.concat(frames)
    save_crosswalk_polars(matches, save_folder, match_round=1, file_suffix=file_suffix)

#%% Match Functions
# Post-2018 Match
def match_hmda_sellers_purchasers_round1(data_folder, save_folder, min_year=2018, max_year=2023, file_suffix=None) :
    """
    Match originations and purchases for HMDA data after 2018.

    Parameters
    ----------
    data_folder : str
        Folder where raw HMDA files are stored.
    save_folder : str
        Folder where matches and match candidates will be saved.
    min_year : int, optional
        First year of data to be matched. The default is 2018.
    max_year : int, optional
        Last year of data to be matched. The default is 2023.

    Returns
    -------
    None.

    """

    # Match Year-by-Year
    df = []
    for year in range(min_year, max_year+1) :

        logger.info("Matching HMDA sellers and purchasers for year: %s", year)

        # Load Data
        df_a = load_data(data_folder, min_year=year, max_year=year)

        # Convert Numerics
        df_a = convert_numerics(df_a)

        # Replace Missings
        df_a = replace_missing_values(df_a)

        # Drop Observations with Missing Match Variables
        match_columns = [
            'loan_type',
            'loan_amount',
            'census_tract',
            'occupancy_type',
            'loan_purpose',
        ]
        df_a = df_a.dropna(subset=match_columns)
        df_a = df_a.query('census_tract not in ["","NA"]')

        # Split into Sellers/Purchasers and Merge
        df_a, df_purchaser = split_sellers_and_purchasers(df_a, save_folder)
        df_a = df_a.merge(df_purchaser, on=match_columns, suffixes=('_s','_p'))
        del df_purchaser

        # Keep Close Matches with tolerances
        match_tolerances = {
            'income': 1,
            'interest_rate': .0625,
        }
        df_a = numeric_matches(df_a, match_tolerances)

        # Weak Numeric Matches
        match_tolerances = {'interest_rate': .01}
        df_a = weak_numeric_matches(df_a, match_tolerances)

        # Check for Matches On Any Fee Variables
        df_a = perform_fee_matches(df_a)
        df_a = df_a.query('NumberFeeMatches >= 1 | NumberNonmissingFees_s == 0 | NumberNonmissingFees_p == 0')

        # Keep Unique Matches
        df_a = keep_uniques(df_a)

        # Use Demographics to Trim Matches
        df_a = match_age(df_a)
        df_a = match_sex(df_a)
        df_a = match_race(df_a)
        df_a = match_ethnicity(df_a)

        # Keep Close Matches with tolerances
        match_tolerances = {
            'conforming_loan_limit': 0,
            'construction_method': 0,
            'discount_points': 5,
            'income': 1,
            'interest_rate': .0625,
            'intro_rate_period': 6,
            'lender_credits': 5,
            'lien_status': 0,
            'loan_term': 12,
            'open_end_line_of_credit': 0,
            'origination_charges': 5,
            'property_value': 20000,
            'total_units': 0,
            'applicant_age_above_62': 0,
            'co_applicant_age_above_62': 0,
        }
        df_a = numeric_matches(df_a, match_tolerances)

        # Clean Up
        df_a['i_DropObservation'] = (np.abs(df_a['interest_rate_s'] - df_a['interest_rate_p']) >= .005) | pd.isna(df_a['income_s']) | pd.isna(df_a['interest_rate_s']) | pd.isna(df_a['property_value_s'])
        df_a['i_DropSale'] = df_a.groupby(['HMDAIndex_s'])['i_DropObservation'].transform('max')
        df_a = df_a.query('i_DropSale != 1')

        # Add to Crosswalks
        df.append(df_a)
        del df_a

    # Combine Matches and Save Crosswalk
    df = pd.concat(df)
    save_crosswalk(df, save_folder, match_round=1, file_suffix=file_suffix)

# Round 2: Cross-year matches
def match_hmda_sellers_purchasers_round2(data_folder, save_folder, min_year=2018, max_year=2023, file_suffix=None) :
    """
    Round 2 of seller-purchaser matches for 2018 onward.

    Parameters
    ----------
    save_folder : str
        Folder where crosswalks are saved.

    Returns
    -------
    None.

    """

    # Combine Seller and Purchaser Data
    df = load_data(data_folder, min_year=min_year, max_year=max_year)

    # Replace Missings
    df = replace_missing_values(df)

    # Drop Observations with Missing Match Variables
    match_columns = ['loan_type',
                     'loan_amount',
                     'census_tract',
                     'occupancy_type',
                     'loan_purpose']
    df = df.dropna(subset = match_columns)
    df = df.query('census_tract != ""')

    # Split into Sellers/Purchasers
    df, df_purchaser = split_sellers_and_purchasers(df, save_folder, match_round=2, file_suffix=file_suffix)
    df = df.merge(df_purchaser, on=match_columns, suffixes=('_s','_p'))
    del df_purchaser

    # Year Matches
    df = df.query('activity_year_s <= activity_year_p')

    # Keep Close Matches with tolerances
    match_tolerances = {'income': 1,
                        'interest_rate': .0625,
                        }
    df = numeric_matches(df, match_tolerances)

    # Weak Numeric Matches
    match_tolerances = {'interest_rate': .01}
    df = weak_numeric_matches(df, match_tolerances)

    # Check for Matches On Any Fee Variables
    df = perform_fee_matches(df)
    df = df.query('NumberFeeMatches >= 1 | NumberNonmissingFees_s == 0 | NumberNonmissingFees_p == 0')

    # Keep Unique Matches
    df = keep_uniques(df)

    # Use Demographics to Trim Matches
    df = match_age(df)
    df = match_sex(df)
    df = match_race(df)
    df = match_ethnicity(df)

    # Keep Close Matches with tolerances
    match_tolerances = {'conforming_loan_limit': 0,
                        'construction_method': 0,
                        'discount_points': 5,
                        'income': 1,
                        'interest_rate': .01,
                        'intro_rate_period': 6,
                        'lender_credits': 5,
                        'lien_status': 0,
                        'loan_term': 12,
                        'open_end_line_of_credit': 0,
                        'origination_charges': 5,
                        'property_value': 20000,
                        'total_units': 0,
                        'applicant_age_above_62': 0,
                        'co_applicant_age_above_62': 0,
                        }
    df = numeric_matches(df, match_tolerances)

    # Clean Up
    df['i_DropObservation'] = (np.abs(df['interest_rate_s'] - df['interest_rate_p']) >= .005) | pd.isna(df['income_s']) | pd.isna(df['interest_rate_s']) | pd.isna(df['property_value_s'])
    df['i_DropSale'] = df.groupby(['HMDAIndex_s'])['i_DropObservation'].transform('max')
    df = df.query('i_DropSale != 1')
    df = df.drop(columns=['i_DropObservation','i_DropSale'], errors='ignore')

    # Save Crosswalk
    save_crosswalk(df, save_folder, match_round=2, file_suffix=file_suffix)

# Round 3: Match Across Years w/
def match_hmda_sellers_purchasers_round3(data_folder, save_folder, min_year=2018, max_year=2023, file_suffix=None) :
    """
    Round 3 of seller-purchaser matches for 2018 onward.

    Parameters
    ----------
    data_folder : str
        DESCRIPTION.
    save_folder : str
        DESCRIPTION.
    min_year : int, optional
        DESCRIPTION. The default is 2018.
    max_year : int, optional
        DESCRIPTION. The default is 2022.

    Returns
    -------
    None.

    """

    # Combine Seller and Purchaser Data
    df = load_data(data_folder, min_year = min_year, max_year = max_year)

    # Replace Missings
    df = replace_missing_values(df)

    # Drop Observations with Missing Match Variables
    match_columns = ['loan_type',
                     'loan_amount',
                     'census_tract',
                     'occupancy_type',
                     'loan_purpose']
    df = df.dropna(subset = match_columns)
    df = df.query('census_tract != ""')

    # Split Sellers and Purchasers
    df, df_purchaser = split_sellers_and_purchasers(df, save_folder, match_round=3, file_suffix=file_suffix)
    df = df.merge(df_purchaser, on=match_columns, suffixes = ('_s','_p'))
    del df_purchaser

    # Year Matches
    df = df.query('activity_year_s <= activity_year_p')

    # Keep Close Matches with tolerances
    match_tolerances = {'income': 1,
                        'interest_rate': .0625,
                        'conforming_loan_limit': 0,
                        'construction_method': 0,
                        'intro_rate_period': 6,
                        'lien_status': 0,
                        'open_end_line_of_credit': 0,
                        'total_units': 0,
                        }
    df = numeric_matches(df, match_tolerances)

    # Age, Sex, Ethnicity, and Race Matches
    df = match_age(df)
    df = match_sex(df)
    df = match_race(df)
    df = match_ethnicity(df)

    # Perform Numeric and Fee Matches
    df = perform_fee_matches(df)
    df = df.query('i_GenerousFeeMatch == 1 | NumberNonmissingFees_s == 0 | NumberNonmissingFees_p == 0')

    # Weak Numeric Matches
    match_tolerances = {'interest_rate': .01}
    df = weak_numeric_matches(df, match_tolerances)

    # Keep Unique Matches
    df = keep_uniques(df, one_to_one=False)

    # Numeric Matches Post Uniques
    match_tolerances = {'income': 1,
                        'interest_rate': .0625,
                        'loan_term': 12,
                        'property_value': 20000,
                        }
    df = numeric_matches_post_unique(df, match_tolerances)

    # Save Crosswalk
    save_crosswalk(df, save_folder, match_round=3, file_suffix=file_suffix)

# Round 4: Match without Loan Purpose Match; Keep Tight Fee/Rate/Income Matches
def match_hmda_sellers_purchasers_round4(data_folder, save_folder, min_year=2018, max_year=2023, file_suffix=None) :
    """
    Round 4 of seller-purchaser matches for 2018 onward.

    Parameters
    ----------
    data_folder : str
        DESCRIPTION.
    save_folder : str
        DESCRIPTION.
    min_year : int, optional
        DESCRIPTION. The default is 2018.
    max_year : int, optional
        DESCRIPTION. The default is 2023.

    Returns
    -------
    None.

    """

    # Combine Seller and Purchaser Data
    df = load_data(data_folder, min_year=min_year, max_year=max_year)

    # Replace Missings
    df = replace_missing_values(df)

    # Drop Observations with Missing Match Variables
    df['i_Purchase'] = 1*(df['loan_purpose'] == 1)
    match_columns = ['loan_type',
                     'loan_amount',
                     'census_tract',
                     'occupancy_type',
                     'i_Purchase']
    df = df.dropna(subset=match_columns)
    df = df.query('census_tract != ""')

    #
    df, df_purchaser = split_sellers_and_purchasers(df, save_folder, match_round=4, file_suffix=file_suffix)
    df = df.merge(df_purchaser, on=match_columns, suffixes=('_s','_p'))
    del df_purchaser

    # Year Matches
    df = df.query('activity_year_s <= activity_year_p')

    # Keep Close Matches with tolerances
    match_tolerances = {'income': 1,
                        'interest_rate': .0625,
                        'conforming_loan_limit': 0,
                        'construction_method': 0,
                        'intro_rate_period': 6,
                        'lien_status': 0,
                        'open_end_line_of_credit': 0,
                        'total_units': 0,
                        }
    df = numeric_matches(df, match_tolerances)

    # Allow Non-matching refi types
    df = df.query('loan_purpose_s == loan_purpose_p | loan_purpose_s in [31,32] | loan_purpose_p in [31,32]')

    # Age, Sex, Ethnicity, and Race Matches
    df = match_age(df)
    df = match_sex(df)
    df = match_race(df)
    df = match_ethnicity(df)

    # Perform Numeric and Fee Matches
    df = perform_fee_matches(df)
    df = df.query('i_GenerousFeeMatch == 1 | NumberNonmissingFees_s == 0 | NumberNonmissingFees_p == 0')

    # Weak Numeric Matches
    match_tolerances = {'interest_rate': .01}
    df = weak_numeric_matches(df, match_tolerances)

    # Keep Unique Matches
    df = keep_uniques(df, one_to_one=False)

    # Numeric Matches Post Uniques
    match_tolerances = {'income': 1,
                        'interest_rate': .0625,
                        'loan_term': 12,
                        'property_value': 20000,
                        }
    df = numeric_matches_post_unique(df, match_tolerances)

    # Save Crosswalk
    save_crosswalk(df, save_folder, match_round=4, file_suffix=file_suffix)
    
# Round 5: Allow for slight loan amount mismatches
def match_hmda_sellers_purchasers_round5(data_folder, save_folder, min_year=2018, max_year=2023, file_suffix=None) :
    """
    Round 5 of seller-purchaser matches for 2018 onward.

    Parameters
    ----------
    data_folder : TYPE
        DESCRIPTION.
    save_folder : TYPE
        DESCRIPTION.
    min_year : TYPE, optional
        DESCRIPTION. The default is 2018.
    max_year : TYPE, optional
        DESCRIPTION. The default is 2022.

    Returns
    -------
    None.

    """

    df = []
    for year in range(min_year, max_year+1) :

        # Load HMDA Data
        df_a = load_data(data_folder, min_year=year, max_year=year)

        # Replace Missings
        df_a = replace_missing_values(df_a)

        # Drop Observations with Missing Match Variables
        df_a['i_Purchase'] = 1*(df_a['loan_purpose'] == 1)
        df_a['LoanAmountMatch'] = df_a['loan_amount']
        match_columns = ['loan_type',
                         'census_tract',
                         'occupancy_type',
                         'i_Purchase',
                         'LoanAmountMatch']
        df_a = df_a.dropna(subset = match_columns)
        df_a = df_a.query('census_tract != ""')

        # Split Sold and Purchased Loans
        df_a, df_purchaser = split_sellers_and_purchasers(df_a, save_folder, match_round=5, file_suffix=file_suffix)

        # Create Seller Match Candidates
        df_a1 = df_a.copy()
        df_a1['LoanAmountMatch'] = df_a1['LoanAmountMatch']-10000
        df_a2 = df_a.copy()
        df_a2['LoanAmountMatch'] = df_a2['LoanAmountMatch']+10000
        df_a = pd.concat([df_a1, df_a, df_a2])
        del df_a1, df_a2

        # Match
        df_a = df_a.merge(df_purchaser, on=match_columns, suffixes=('_s','_p'))
        del df_purchaser

        # Keep Close Matches with tolerances
        match_tolerances = {'income': 1,
                            'interest_rate': .0625,
                            }
        df_a = numeric_matches(df_a, match_tolerances)

        # Weak Numeric Matches
        match_tolerances = {'interest_rate': .01}
        df_a = weak_numeric_matches(df_a, match_tolerances)

        # Check for Matches On Any Fee Variables
        df_a = perform_fee_matches(df_a)
        df_a = df_a.query('NumberFeeMatches >= 1 | NumberNonmissingFees_s == 0 | NumberNonmissingFees_p == 0')

        # Keep Unique Matches
        df_a = keep_uniques(df_a)

        # Use Demographics to Trim Matches
        df_a = match_age(df_a)
        df_a = match_sex(df_a)
        df_a = match_race(df_a)
        df_a = match_ethnicity(df_a)

        # Keep Close Matches with tolerances
        match_tolerances = {'conforming_loan_limit': 0,
                            'construction_method': 0,
                            'discount_points': 5,
                            'income': 1,
                            'interest_rate': .0625,
                            'intro_rate_period': 6,
                            'lender_credits': 5,
                            'lien_status': 0,
                            'loan_term': 12,
                            'open_end_line_of_credit': 0,
                            'origination_charges': 5,
                            'property_value': 20000,
                            'total_units': 0,
                            'applicant_age_above_62': 0,
                            'co_applicant_age_above_62': 0,
                            'loan_amount': 10000,
                            }
        df_a = numeric_matches(df_a, match_tolerances)
        
        # Clean Up
        df_a['i_DropObservation'] = (np.abs(df_a['interest_rate_s'] - df_a['interest_rate_p']) >= .005) | pd.isna(df_a['income_s']) | pd.isna(df_a['interest_rate_s']) | pd.isna(df_a['property_value_s'])
        df_a['i_DropSale'] = df_a.groupby(['HMDAIndex_s'])['i_DropObservation'].transform('max')
        df_a = df_a.query('i_DropSale != 1')

        # Add to Crosswalks
        df.append(df_a)
        del df_a

    # Combine Matches
    df = pd.concat(df)

    # Drop Negative Loan Differences
    df['LoanDiff'] = df.loan_amount_s-df.loan_amount_p
    df = df.query('LoanDiff >= 0')
    
    # Drop Secondary Sales
    df = df.query('purchaser_type_p in [0,1,2,3,4]')

    # Save Crosswalk
    save_crosswalk(df, save_folder, match_round=5, file_suffix=file_suffix)

# Round 6: Affiliate Matches
def match_hmda_sellers_purchasers_round6(data_folder, save_folder, min_year=2018, max_year=2023, file_suffix=None) :
    """
    Round 5 of seller-purchaser matches for 2018 onward. Match for affiliates.

    Parameters
    ----------
    data_folder : TYPE
        DESCRIPTION.
    save_folder : TYPE
        DESCRIPTION.
    min_year : TYPE, optional
        DESCRIPTION. The default is 2018.
    max_year : TYPE, optional
        DESCRIPTION. The default is 2022.

    Returns
    -------
    None.

    """

    # Get Affiliates
    affiliated_leis = get_affiliates(data_folder, save_folder, 5, min_year=2018, max_year=2023, strict=False, file_suffix=file_suffix)

    # Load HMDA Data
    df = load_data(data_folder, min_year=min_year, max_year=max_year)

    # Keep Only Affiliate Sales among Originations
    df = df.query('action_taken == 6 | purchaser_type == 8')

    # Replace Missings
    df = replace_missing_values(df)

    # Drop Observations with Missing Match Variables
    df['i_Purchase'] = 1*(df['loan_purpose'] == 1)
    match_columns = ['loan_type',
                     'loan_amount',
                     'county_code',
                     'occupancy_type',
                     'i_Purchase']
    df = df.dropna(subset = match_columns)

    # Split Sold and Purchased Loans
    df, df_purchaser = split_sellers_and_purchasers(df, save_folder, match_round=5, file_suffix=file_suffix)
    df = df.merge(df_purchaser, on=match_columns, suffixes=('_s','_p'))
    del df_purchaser

    # Year Matches
    df = df.query('activity_year_s <= activity_year_p')

    # Keep Close Matches with tolerances
    match_tolerances = {'interest_rate': .0625,
                        'conforming_loan_limit': 0,
                        'construction_method': 0,
                        'intro_rate_period': 6,
                        'lien_status': 0,
                        'open_end_line_of_credit': 0,
                        'total_units': 0,
                        }
    df = numeric_matches(df, match_tolerances)

    # Allow Non-matching refi types
    df = df.query('loan_purpose_s == loan_purpose_p or loan_purpose_s in [31,32] or loan_purpose_p in [31,32]')

    # Age, Sex, Ethnicity, and Race Matches
    df = match_age(df)
    df = match_sex(df)
    df = match_race(df)
    df = match_ethnicity(df)

    # Perform Numeric and Fee Matches
    df = perform_fee_matches(df)
    df = df.query('i_GenerousFeeMatch == 1 or NumberNonmissingFees_s == 0 | NumberNonmissingFees_p == 0')

    # Drop Candidate Matches with No Generous Fee Match where Fees have Any Match
    df['i_GoodMatch'] = 1*(df['NumberFeeMatches'] >= 2)
    df['i_SaleHasFeeMatch'] = df.groupby(['HMDAIndex_s'])['i_GoodMatch'].transform('max')
    df['i_PurchaseHasFeeMatch'] = df.groupby(['HMDAIndex_p'])['i_GoodMatch'].transform('max')
    df = df.query('i_GenerousFeeMatch == 1 | i_SaleHasFeeMatch == 0')
    df = df.query('i_GenerousFeeMatch == 1 | i_PurchaseHasFeeMatch == 0')

    # Drop Candidate Matches with No Generous Fee Match where Fees have Any Match
    df['IncomeDifference'] = df['income_s'] - df['income_p']
    df['i_PerfectIncomeMatch'] = df['income_s'] == df['income_p']
    df.loc[pd.isna(df['income_s']) | pd.isna(df['income_p']), 'i_PerfectIncomeMatch'] = None
    df['i_SaleHasIncomeMatch'] = df.groupby(['HMDAIndex_s'])['i_PerfectIncomeMatch'].transform('max')
    df['i_PurchaseHasIncomeMatch'] = df.groupby(['HMDAIndex_p'])['i_PerfectIncomeMatch'].transform('max')
    df = df.query('abs(IncomeDifference) <= 1 | i_SaleHasIncomeMatch != 1')
    df = df.query('abs(IncomeDifference) <= 1 | i_PurchaseHasIncomeMatch != 1')

    # Weak Numeric Matches
    match_tolerances = {'interest_rate': .01}
    df = weak_numeric_matches(df, match_tolerances)

    # Keep Only Affiliate Matches
    df = df.merge(affiliated_leis, on=['lei_s','lei_p'])

    # Keep Unique Matches
    df = keep_uniques(df, one_to_one=False)

    # Numeric Matches Post Uniques
    match_tolerances = {'income': 1,
                        'interest_rate': .0625,
                        'loan_term': 12,
                        'property_value': 20000,
                        }
    df = numeric_matches_post_unique(df, match_tolerances)

    # Save Crosswalk
    save_crosswalk(df, save_folder, match_round=6, file_suffix=file_suffix)

# Round 7: Match with Purchaser Type
def match_hmda_sellers_purchasers_round7(data_folder, save_folder, min_year=2018, max_year=2023, file_suffix=None) :
    """
    Round 7 of seller-purchaser matches for 2018 onward.

    Parameters
    ----------
    data_folder : TYPE
        DESCRIPTION.
    save_folder : TYPE
        DESCRIPTION.
    min_year : TYPE, optional
        DESCRIPTION. The default is 2018.
    max_year : TYPE, optional
        DESCRIPTION. The default is 2022.

    Returns
    -------
    None.

    """

    # Create Crosswalks for Sellers, Purchasers, and Types
    previous_match_round = 5
    purchaser_types = pd.read_parquet(f"{save_folder}/hmda_seller_purchaser_relationships_w_types_round{previous_match_round}{file_suffix}.parquet")

    # Load Data
    df = load_data(data_folder, min_year=min_year, max_year=max_year)

    # Replace Missings
    df = replace_missing_values(df)

    # Drop Observations with Missing Match Variables
    df['i_Purchase'] = 1*(df['loan_purpose'] == 1)
    match_columns = ['loan_type',
                     'loan_amount',
                     'census_tract',
                     'occupancy_type',
                     'i_Purchase']
    df = df.dropna(subset = match_columns)
    df = df.query('census_tract != ""')

    # Split Sold and Purchased Loans
    df, df_purchaser = split_sellers_and_purchasers(df, save_folder, match_round=7, file_suffix=file_suffix)
    df = df.merge(df_purchaser, on=match_columns, suffixes=('_s','_p'))
    del df_purchaser

    # Year Matches
    df = df.query('activity_year_s <= activity_year_p')

    # Drop Purchaser Type Mismatches
    df = df.merge(purchaser_types[['lei_s','lei_p','activity_year_s','purchaser_type_s']],
                  on = ['lei_s','lei_p','activity_year_s','purchaser_type_s'],
                  how = 'left',
                  indicator = True,
                  )
    df['i_PurchaserTypeMatch'] = 1*(df._merge == 'both')
    df = df.drop(columns = ['_merge'])
    df = df.query('i_PurchaserTypeMatch == 1 | purchaser_type_s == 0')

    # Allow Non-matching refi types
    df = df.query('loan_purpose_s == loan_purpose_p or (loan_purpose_s in [31,32] and loan_purpose_p in [31,32])')

    # Keep Close Matches with tolerances
    match_tolerances = {'interest_rate': .0625,
                        'conforming_loan_limit': 0,
                        'construction_method': 0,
                        'intro_rate_period': 6,
                        'lien_status': 0,
                        'open_end_line_of_credit': 0,
                        'total_units': 0,
                        }
    df = numeric_matches(df, match_tolerances)

    # Age, Sex, Ethnicity, and Race Matches
    df = match_age(df)
    df = match_sex(df)
    df = match_race(df)
    df = match_ethnicity(df)

    # Perform Numeric and Fee Matches
    df = perform_fee_matches(df)
    df = df.query('i_GenerousFeeMatch == 1 or NumberNonmissingFees_s == 0 | NumberNonmissingFees_p == 0')
    
    # Drop Candidate Matches with No Generous Fee Match where Fees have Any Match
    df['RateDifference'] = df['interest_rate_s'] - df['interest_rate_p']
    df['i_GoodRateMatch'] = abs(df['RateDifference']) < .001
    df.loc[pd.isna(df['interest_rate_s']) | pd.isna(df['interest_rate_p']), 'i_GoodRateMatch'] = None
    df['i_SaleHasRateMatch'] = df.groupby(['HMDAIndex_s'])['i_GoodRateMatch'].transform('max')
    df['i_PurchaseHasRateMatch'] = df.groupby(['HMDAIndex_p'])['i_GoodRateMatch'].transform('max')
    df = df.query('abs(RateDifference) < .001 | i_SaleHasRateMatch != 1')
    df = df.query('abs(RateDifference) < .001 | i_PurchaseHasRateMatch != 1')

    # Drop Candidate Matches with No Generous Fee Match where Fees have Any Match
    df['i_GoodMatch'] = 1*(df['NumberFeeMatches'] >= 2)
    df['i_SaleHasFeeMatch'] = df.groupby(['HMDAIndex_s'])['i_GoodMatch'].transform('max')
    df['i_PurchaseHasFeeMatch'] = df.groupby(['HMDAIndex_p'])['i_GoodMatch'].transform('max')
    df = df.query('i_GenerousFeeMatch == 1 | i_SaleHasFeeMatch == 0')
    df = df.query('i_GenerousFeeMatch == 1 | i_PurchaseHasFeeMatch == 0')

    # Drop Candidate Matches with No Generous Fee Match where Fees have Any Match
    df['IncomeDifference'] = df['income_s'] - df['income_p']
    df['i_PerfectIncomeMatch'] = df['income_s'] == df['income_p']
    df.loc[pd.isna(df['income_s']) | pd.isna(df['income_p']), 'i_PerfectIncomeMatch'] = None
    df['i_SaleHasIncomeMatch'] = df.groupby(['HMDAIndex_s'])['i_PerfectIncomeMatch'].transform('max')
    df['i_PurchaseHasIncomeMatch'] = df.groupby(['HMDAIndex_p'])['i_PerfectIncomeMatch'].transform('max')
    df = df.query('abs(IncomeDifference) <= 1 | i_SaleHasIncomeMatch != 1')
    df = df.query('abs(IncomeDifference) <= 1 | i_PurchaseHasIncomeMatch != 1')
    
    # Keep Unique Matches
    df = keep_uniques(df, one_to_one=False)
    
    # Numeric Matches Post Uniques
    match_tolerances = {'income': 1,
                        'interest_rate': .01,
                        'loan_term': 0,
                        'property_value': 10000,
                        }
    df = numeric_matches_post_unique(df, match_tolerances)

    # Save Crosswalk
    save_crosswalk(df, save_folder, match_round=7, file_suffix=file_suffix)

# Round 8: Match without PurchaserType=0 Originations
def match_hmda_sellers_purchasers_round8(data_folder, save_folder, min_year=2018, max_year=2023, file_suffix=None) :
    """
    Round 8 of seller-purchaser matches for 2018 onward.

    Parameters
    ----------
    data_folder : TYPE
        DESCRIPTION.
    save_folder : TYPE
        DESCRIPTION.
    min_year : TYPE, optional
        DESCRIPTION. The default is 2018.
    max_year : TYPE, optional
        DESCRIPTION. The default is 2022.

    Returns
    -------
    None.

    """

    # Load Data
    df = load_data(data_folder, min_year=min_year, max_year=max_year)

    df = df.query('action_taken == 6 | purchaser_type != 0')

    # Replace Missings
    df = replace_missing_values(df)

    # Drop Observations with Missing Match Variables
    df['i_Purchase'] = 1*(df['loan_purpose'] == 1)
    match_columns = ['loan_type',
                     'loan_amount',
                     'census_tract',
                     'occupancy_type',
                     'i_Purchase']
    df = df.dropna(subset = match_columns)
    df = df.query('census_tract != ""')

    # Split Sold and Purchased Loans
    df, df_purchaser = split_sellers_and_purchasers(df, save_folder, match_round=8, file_suffix=file_suffix)
    df = df.merge(df_purchaser, on=match_columns, suffixes=('_s','_p'))
    del df_purchaser

    # Drop Type Mismatches
    # mmerge lei rt_lei activity_year using "$match_folder/seller_purchaser_match_types.dta", umatch(s_lei p_lei s_activity_year) unmatched(master)
    # drop if ~missing(s_purchaser_type) & purchaser_type != 0 & purchaser_type != s_purchaser_type

    # Keep LEIs with Known Connections
    # mmerge lei rt_lei using "$match_folder/large_seller_purchaser_relationships.dta", umatch(s_lei p_lei) unmatched(master)
    # gen i_LEIMatch = (_merge == 3)
    # bys lei: egen i_SellerLEIHasMatch = max(i_LEIMatch)
    # bys rt_lei: egen i_PurchaserLEIHasMatch = max(i_LEIMatch)
    # tab i_LEIMatch i_SellerLEIHasMatch
    # tab i_LEIMatch i_PurchaserLEIHasMatch
    # drop if i_LEIMatch == 0 & i_SellerLEIHasMatch == 1 & i_PurchaserLEIHasMatch == 1

    # Year Matches
    df = df.query('activity_year_s <= activity_year_p')

    # Allow Non-matching refi types
    df = df.query('loan_purpose_s == loan_purpose_p or (loan_purpose_s in [31,32] and loan_purpose_p in [31,32])')

    # Keep Close Matches with tolerances
    match_tolerances = {'interest_rate': .0625,
                        'conforming_loan_limit': 0,
                        'construction_method': 0,
                        'intro_rate_period': 6,
                        'lien_status': 0,
                        'open_end_line_of_credit': 0,
                        'total_units': 0,
                        }
    df = numeric_matches(df, match_tolerances)

    # Age, Sex, Ethnicity, and Race Matches
    df = match_age(df)
    df = match_sex(df)
    df = match_race(df)
    df = match_ethnicity(df)

    # Perform Numeric and Fee Matches
    df = perform_fee_matches(df)
    df = df.query('i_GenerousFeeMatch == 1 or NumberNonmissingFees_s == 0 | NumberNonmissingFees_p == 0')

    # Drop Candidate Matches with No Generous Fee Match where Fees have Any Match
    df['RateDifference'] = df['interest_rate_s'] - df['interest_rate_p']
    df['i_GoodRateMatch'] = abs(df['RateDifference']) < .001
    df.loc[pd.isna(df['interest_rate_s']) | pd.isna(df['interest_rate_p']), 'i_GoodRateMatch'] = None
    df['i_SaleHasRateMatch'] = df.groupby(['HMDAIndex_s'])['i_GoodRateMatch'].transform('max')
    df['i_PurchaseHasRateMatch'] = df.groupby(['HMDAIndex_p'])['i_GoodRateMatch'].transform('max')
    df = df.query('abs(RateDifference) < .001 | i_SaleHasRateMatch != 1')
    df = df.query('abs(RateDifference) < .001 | i_PurchaseHasRateMatch != 1')

    # Drop Candidate Matches with No Generous Fee Match where Fees have Any Match
    df['i_GoodMatch'] = 1*(df['NumberFeeMatches'] >= 2)
    df['i_SaleHasFeeMatch'] = df.groupby(['HMDAIndex_s'])['i_GoodMatch'].transform('max')
    df['i_PurchaseHasFeeMatch'] = df.groupby(['HMDAIndex_p'])['i_GoodMatch'].transform('max')
    df = df.query('i_GenerousFeeMatch == 1 | i_SaleHasFeeMatch == 0')
    df = df.query('i_GenerousFeeMatch == 1 | i_PurchaseHasFeeMatch == 0')

    # Drop Candidate Matches with No Generous Fee Match where Fees have Any Match
    df['IncomeDifference'] = df['income_s'] - df['income_p']
    df['i_PerfectIncomeMatch'] = df['income_s'] == df['income_p']
    df.loc[pd.isna(df['income_s']) | pd.isna(df['income_p']), 'i_PerfectIncomeMatch'] = None
    df['i_SaleHasIncomeMatch'] = df.groupby(['HMDAIndex_s'])['i_PerfectIncomeMatch'].transform('max')
    df['i_PurchaseHasIncomeMatch'] = df.groupby(['HMDAIndex_p'])['i_PerfectIncomeMatch'].transform('max')
    df = df.query('abs(IncomeDifference) <= 1 | i_SaleHasIncomeMatch != 1')
    df = df.query('abs(IncomeDifference) <= 1 | i_PurchaseHasIncomeMatch != 1')

    # Keep Unique Matches
    df = keep_uniques(df, one_to_one=False)

    # Numeric Matches Post Uniques
    match_tolerances = {'income': 2,
                        'interest_rate': .01,
                        'loan_term': 12,
                        'property_value': 30000,
                        }
    df = numeric_matches_post_unique(df, match_tolerances)

    # Save Crosswalk
    save_crosswalk(df, save_folder, match_round=8, file_suffix=file_suffix)

# Create Matched File
def create_matched_file(data_folder, match_folder, min_year=2018, max_year=2023, match_round=1, file_suffix=None) :
    """
    Creates a file with all HMDA data fields for matched sold/purchased loans.

    Parameters
    ----------
    data_folder : TYPE
        DESCRIPTION.
    match_folder : TYPE
        DESCRIPTION.
    min_year : TYPE, optional
        DESCRIPTION. The default is 2018.
    max_year : TYPE, optional
        DESCRIPTION. The default is 2022.
    match_round : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """

    # Load Crosswalk
    cw = pq.read_table(f"{match_folder}/hmda_seller_purchaser_matches_round{match_round}{file_suffix}.parquet").to_pandas()

    # Combine Seller and Purchaser Data
    df_seller = []
    for year in range(min_year, max_year+1) :
        logger.info('Keeping matched sold loans from year: %s', year)
        file = HMDALoader.get_hmda_files(data_folder, min_year=year, max_year=year, extension='parquet')[0]
        df_a = pq.read_table(file, filters=[('action_taken','in',[1])]).to_pandas(date_as_object = False)
        df_a = df_a.merge(cw, left_on = ['HMDAIndex'], right_on = ['HMDAIndex_s'], how = 'inner')
        df_seller.append(df_a)
        del df_a
    df_seller = pd.concat(df_seller)
    df_seller = df_seller.drop(columns = ['HMDAIndex'])

    # Combine Seller and Purchaser Data
    df_purchaser = []
    for year in range(min_year, max_year+1) :
        logger.info('Keeping matched purchased loans from year: %s', year)
        file = HMDALoader.get_hmda_files(data_folder, min_year=year, max_year=year, extension='parquet')[0]
        df_a = pq.read_table(file, filters=[('action_taken','in',[6])]).to_pandas(date_as_object = False)
        df_a = df_a.merge(cw, left_on=['HMDAIndex'], right_on=['HMDAIndex_p'])
        df_purchaser.append(df_a)
        del df_a
    df_purchaser = pd.concat(df_purchaser)
    df_purchaser = df_purchaser.drop(columns = ['HMDAIndex'])

    # Merge Sellers and Purchasers
    match_columns = ['HMDAIndex_s','HMDAIndex_p','match_round']
    df = df_seller.merge(df_purchaser, on=match_columns, suffixes=('_s','_p'))
    del df_seller, df_purchaser

    # Sort and Save Combined Data
    df = df[df.columns.sort_values()]
    df = df[match_columns+[x for x in df.columns if x not in match_columns]]
    df = pa.Table.from_pandas(df, preserve_index = False)
    pq.write_table(df, f'{match_folder}/hmda_seller_purchaser_matched_loans_round{match_round}{file_suffix}.parquet')

#%% Match Support Functions
# Get Purchaser Type Counts
def get_affiliates(data_folder, match_folder, match_round, min_year=2018, max_year=2023, strict=False, file_suffix=None) :
    """
    Get list of affiliate institutions for sellers and purchasers

    Parameters
    ----------
    data_folder : str
        DESCRIPTION.
    match_folder : str
        DESCRIPTION.
    match_round : int
        DESCRIPTION.
    min_year : int, optional
        DESCRIPTION. The default is 2018.
    max_year : int, optional
        DESCRIPTION. The default is 2023.
    strict : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    cw : pandas DataFrame
        DESCRIPTION.

    """

    # Load Crosswalk
    df = pd.read_parquet(f'{match_folder}/hmda_seller_purchaser_matched_loans_round{match_round}{file_suffix}.parquet',
                         columns = ['HMDAIndex_s','HMDAIndex_p','activity_year_s','purchaser_type_s','purchaser_type_p','lei_s','lei_p'])

    # # Count Sold Loans
    df['CountSoldLoan'] = df.groupby(['HMDAIndex_s'])['HMDAIndex_s'].transform('count')

    # Keep Unique Matches Only
    if strict :
        df = df.query('CountSoldLoan == 1')
        df = df.drop(columns = ['CountSoldLoan'])
    else :
        df = df.query('CountSoldLoan == 1 or purchaser_type_s not in [1,2,3,4]')
        df = df.drop(columns = ['CountSoldLoan'])
        df['CountSoldLoan'] = df.groupby(['HMDAIndex_s'])['HMDAIndex_s'].transform('count')
        df = df.query('CountSoldLoan == 1')
        df = df.drop(columns = ['CountSoldLoan'])

    # Drop Loans With Unknown Purchaser
    df = df.query('purchaser_type_s != 0')

    # Count Matches between LEIs at the Year Level
    df['CountLEIMatches'] = df.groupby(['lei_s','lei_p','activity_year_s'])['activity_year_s'].transform('count')
    df['CountLEIPurchaserTypeMatches'] = df.groupby(['lei_s','lei_p','activity_year_s','purchaser_type_s'])['activity_year_s'].transform('count')

    #
    df = df.query('purchaser_type_s == 8')
    df['index'] = df.groupby(['lei_s','lei_p','activity_year_s'])['HMDAIndex_s'].rank('dense')
    df = df.drop_duplicates(subset = ['lei_s','lei_p','activity_year_s'])

    # Keep Only Good Matches for Common LEIs
    df = df.loc[df['CountLEIPurchaserTypeMatches']/df['CountLEIMatches'] >= .95]
    df = df.query('CountLEIPurchaserTypeMatches >= 10')

    # Keep Unique Matches
    df = df[['lei_s','lei_p']].drop_duplicates()

    # Sort and Save
    df = df.sort_values(by = ['lei_s','lei_p'])
    df.to_csv(f'{match_folder}/affiliate_lei_matches_round{match_round}{file_suffix}.csv', index = False)

    # Load TS/Panel Data and Merge in Names
    lender_folder = DATA_DIR
    lei_names = pd.read_csv(f'{lender_folder}/hmda_lenders_combined_2018-2022.csv',
                            sep = '|',
                            usecols = ['lei', 'respondent_name_panel', 'respondent_name_ts'],
                            )
    for column in lei_names.columns :
        lei_names[column] = lei_names[column].str.upper()
    lei_names = lei_names.drop_duplicates()

    # Merge in Names
    df = df.merge(lei_names, left_on = ['lei_s'], right_on = ['lei'])
    df = df.drop(columns = ['lei'])
    df = df.merge(lei_names, left_on = ['lei_p'], right_on = ['lei'], suffixes = ('_s','_p'))
    df = df.drop(columns = ['lei'])
    df = df.drop_duplicates()

    # Keep Unique LEI Combinations
    df_u = df[['lei_s','lei_p']].drop_duplicates()

    # Return Crosswalk with Names
    return df_u

# Get Purchaser Type Counts
def get_purchaser_type_counts(data_folder, match_folder, match_round, min_year=2018, max_year=2023, strict=False, file_suffix=None) :
    """
    Purchaser types are not unique, but appear to be unique (or almost unique)
    within a given seller-year

    Parameters
    ----------
    data_folder : TYPE
        DESCRIPTION.
    match_folder : TYPE
        DESCRIPTION.
    match_round : TYPE
        DESCRIPTION.
    min_year : TYPE, optional
        DESCRIPTION. The default is 2018.
    max_year : TYPE, optional
        DESCRIPTION. The default is 2022.

    Returns
    -------
    None.

    """

    # Load Crosswalk
    df = pd.read_parquet(f'{match_folder}/hmda_seller_purchaser_matched_loans_round{match_round}{file_suffix}.parquet',
                         columns=['HMDAIndex_s','HMDAIndex_p','activity_year_s','purchaser_type_s','purchaser_type_p','lei_s','lei_p'])

    # # Count Sold Loans
    df['CountSoldLoan'] = df.groupby(['HMDAIndex_s'])['HMDAIndex_s'].transform('count')

    # Keep Unique Matches Only
    if strict :
        df = df.query('CountSoldLoan == 1')
        df = df.drop(columns = ['CountSoldLoan'])
    else :
        df = df.query('CountSoldLoan == 1 or purchaser_type_s not in [1,2,3,4]')
        df = df.drop(columns = ['CountSoldLoan'])
        df['CountSoldLoan'] = df.groupby(['HMDAIndex_s'])['HMDAIndex_s'].transform('count')
        df = df.query('CountSoldLoan == 1')
        df = df.drop(columns = ['CountSoldLoan'])

    # Drop Loans With Unknown Purchaser
    df = df.query('purchaser_type_s != 0')

    # Count Matches between LEIs at the Year Level
    df['CountLEIMatches'] = df.groupby(['lei_s','lei_p','activity_year_s'])['activity_year_s'].transform('count')
    df['CountLEIPurchaserTypeMatches'] = df.groupby(['lei_s','lei_p','activity_year_s','purchaser_type_s'])['activity_year_s'].transform('count')

    # Keep Good Matches for Frequent Lender Matches
    df = df.loc[df['CountLEIPurchaserTypeMatches']/df['CountLEIMatches'] >= .99]
    df = df.query('CountLEIPurchaserTypeMatches >= 50')

    # Save
    df = df[['lei_s', 'lei_p', 'purchaser_type_s', 'activity_year_s','CountLEIMatches', 'CountLEIPurchaserTypeMatches']]
    df = df.drop_duplicates(subset = ['lei_s','lei_p','activity_year_s','purchaser_type_s'])
    df = df.sort_values(by = ['lei_s','lei_p','activity_year_s','purchaser_type_s'])
    df = pa.Table.from_pandas(df, preserve_index = False)
    pq.write_table(df, f'{match_folder}/hmda_seller_purchaser_relationships_w_types_round{match_round}{file_suffix}.parquet')

# Get Purchaser Type Counts
def get_lei_match_counts(data_folder, match_folder, match_round, min_year=2018, max_year=2023, strict=False) :
    """
    Gets seller-purchaser relationships between large originators and investors.

    Parameters
    ----------
    data_folder : TYPE
        DESCRIPTION.
    match_folder : TYPE
        DESCRIPTION.
    match_round : TYPE
        DESCRIPTION.
    min_year : TYPE, optional
        DESCRIPTION. The default is 2018.
    max_year : TYPE, optional
        DESCRIPTION. The default is 2022.
    strict : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """

    # Load Crosswalk
    df = pd.read_parquet(f'{match_folder}/hmda_seller_purchaser_matched_loans_round{match_round}.parquet',
                     columns = ['HMDAIndex_s','HMDAIndex_p','activity_year_s','purchaser_type_s','purchaser_type_p','lei_s','lei_p'],
                     )

    # # Count Sold Loans
    df['CountSoldLoan'] = df.groupby(['HMDAIndex_s'])['HMDAIndex_s'].transform('count')

    # Keep Unique Matches Only
    if strict :
        df = df.query('CountSoldLoan == 1')
        df = df.drop(columns = ['CountSoldLoan'])
    else :
        df = df.query('CountSoldLoan == 1 or purchaser_type_s not in [1,2,3,4]')
        df = df.drop(columns = ['CountSoldLoan'])
        df['CountSoldLoan'] = df.groupby(['HMDAIndex_s'])['HMDAIndex_s'].transform('count')
        df = df.query('CountSoldLoan == 1')
        df = df.drop(columns = ['CountSoldLoan'])

    # Count Seller Loans, Purchaser Loans, and Matches
    df['CountSeller'] = df.groupby(['lei_s'])['lei_s'].transform('count')
    df['CountPurchaser'] = df.groupby(['lei_p'])['lei_p'].transform('count')
    df['CountMatches'] = df.groupby(['lei_s','lei_p'])['lei_s'].transform('count')

    # Keep Single Match Observations
    df = df[['lei_s','lei_p','CountSeller','CountPurchaser','CountMatches']].drop_duplicates()

    # Drop Extremely Poor Matches
    df = df.loc[(df['CountMatches']/df['CountSeller'] >= 0.001) | (df['CountMatches']/df['CountPurchaser'] >= 0.001)]
    df = df.loc[(df['CountMatches']/df['CountSeller'] >= 0.005) | (df['CountMatches']/df['CountPurchaser'] >= 0.0005)]
    df = df.loc[(df['CountMatches']/df['CountSeller'] >= 0.0005) | (df['CountMatches']/df['CountPurchaser'] >= 0.005)]
    df = df.loc[(df['CountMatches']/df['CountSeller'] >= 0.0001) | (df['CountMatches']/df['CountPurchaser'] >= 0.01)]
    df = df.loc[(df['CountMatches']/df['CountSeller'] >= 0.01) | (df['CountMatches']/df['CountPurchaser'] >= 0.0001)]
    df = df.query('CountSeller >= 1000 and CountPurchaser >= 1000')

    # Create Crosswalk for Large Sellers and Purchasers
    df = df[['lei_s','lei_p']]
    df = pa.Table.from_pandas(df, preserve_index = False)
    pq.write_table(df, f"{match_folder}/large_seller_purchaser_relationships_round{match_round}.parquet")

#%% Main Routine
if __name__ == '__main__' :

    logging.basicConfig(level=logging.INFO)

    # Unzip HMDA Data
    DATA_DIR = config.DATA_DIR
    DATA_FOLDER = DATA_DIR / 'clean'
    SAVE_FOLDER = DATA_DIR / 'match_data/match_sellers_purchasers_post2018'
    file_suffix = '_202409'

    # Conduct Matches in Rounds
    # match_hmda_sellers_purchasers_round1(DATA_FOLDER, SAVE_FOLDER, min_year=2018, max_year=2023, file_suffix=file_suffix)
    # match_hmda_sellers_purchasers_round2(DATA_FOLDER, SAVE_FOLDER, min_year=2018, max_year=2023, file_suffix=file_suffix)
    # match_hmda_sellers_purchasers_round3(DATA_FOLDER, SAVE_FOLDER, min_year=2018, max_year=2023, file_suffix=file_suffix)
    # match_hmda_sellers_purchasers_round4(DATA_FOLDER, SAVE_FOLDER, min_year=2018, max_year=2023, file_suffix=file_suffix)
    # match_hmda_sellers_purchasers_round5(DATA_FOLDER, SAVE_FOLDER, min_year=2018, max_year=2023, file_suffix=file_suffix)
    
    # create_matched_file(DATA_FOLDER, SAVE_FOLDER, min_year=2018, max_year=2023, match_round=5, file_suffix=file_suffix)
    # get_purchaser_type_counts(DATA_FOLDER, SAVE_FOLDER, 5, min_year=2018, max_year=2023, file_suffix=file_suffix)

    # match_hmda_sellers_purchasers_round6(DATA_FOLDER, SAVE_FOLDER, min_year=2018, max_year=2023, file_suffix=file_suffix)
    # match_hmda_sellers_purchasers_round7(DATA_FOLDER, SAVE_FOLDER, min_year=2018, max_year=2023, file_suffix=file_suffix)
    # match_hmda_sellers_purchasers_round8(DATA_FOLDER, SAVE_FOLDER, min_year=2018, max_year=2023, file_suffix=file_suffix)

    # create_matched_file(DATA_FOLDER, SAVE_FOLDER, min_year=2018, max_year=2023, match_round=8, file_suffix=file_suffix)
    
    # ## Examine Match Data
    # match_folder = SAVE_FOLDER
    # match_round = 8
    # df = pd.read_parquet(f'{match_folder}/hmda_seller_purchaser_matched_loans_round{match_round}{file_suffix}.parquet')
