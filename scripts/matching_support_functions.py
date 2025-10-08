# Import Packages
from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence

import logging

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import HMDALoader


logger = logging.getLogger(__name__)

FilterCondition = tuple[str, str, object]

# Get Match Columns
def get_match_columns(file: str | os.PathLike[str]) -> list[str] :
    """Read the HMDA parquet metadata to identify usable match columns.

    Parameters
    ----------
    file : str or os.PathLike[str]
        File to load for columns.

    Returns
    -------
    list[str]
        Column names that should be retained for matching routines.
    """

    # Load File Column Names
    columns = pq.read_metadata(file).schema.names

    # Drop Columns Not Used in Match
    drop_columns = [
        'denial_reason',
        'denial_reason_1',
        'denial_reason_2',
        'denial_reason_3',
        'denial_reason_4',
        'aus',
        'aus_1',
        'aus_2',
        'aus_3',
        'aus_4',
        'aus_5',
        'applicant_credit_score_type',
        'co_applicant_credit_score_type',
        'initially_payable_to_institution',
        'submission_of_application',
        'tract_population',
        'tract_minority_population_percent',
        'ffiec_msa_md_median_family_income',
        'tract_to_msa_income_percentage',
        'tract_owner_occupied_units',
        'tract_one_to_four_family_homes',
        'tract_median_age_of_housing_units',
        'derived_loan_product_type',
        'derived_dwelling_category',
        'derived_ethnicity',
        'derived_race',
        'derived_sex',
    ]
    columns = [x for x in columns if x not in drop_columns]
    
    # Return Columns
    return columns

# Load HMDA Data
def load_data(
    data_folder: str,
    min_year: int = 2018,
    max_year: int = 2023,
    added_filters: Optional[Sequence[FilterCondition]] = None,
) -> pd.DataFrame :
    """Combine HMDA data files and keep only originations and purchases.

    Parameters
    ----------
    data_folder : str
        Folder where HMDA data files are located.
    min_year : int, optional
        Minimum year of data to include (inclusive). The default is 2018.
    max_year : int, optional
        Maximum year of data to include (inclusive). The default is 2023.
    added_filters : Sequence[tuple[str, str, object]], optional
        Additional pyarrow filter tuples to apply when reading the parquet
        files. Each filter follows the (column, operator, value) convention
        expected by :func:`pandas.read_parquet`.

    Returns
    -------
    pandas.DataFrame
        Combined HMDA data covering the requested years.
    """

    # Set Filters
    hmda_filters: list[FilterCondition] = [('action_taken','in',[1,6])]
    if added_filters is not None :
        hmda_filters += list(added_filters)

    # Combine Seller and Purchaser Data
    df_list = []
    for year in range(min_year, max_year+1) :
        file = HMDALoader.get_hmda_files(data_folder, min_year=year, max_year=year, extension='parquet')[0]
        hmda_columns = get_match_columns(file)
        df_a = pd.read_parquet(file, columns=hmda_columns, filters=hmda_filters)
        df_a = df_a.query('purchaser_type not in [1,2,3,4] | action_taken == 6')
        df_list.append(df_a)
        del df_a
    df = pd.concat(df_list)

    # Return DataFrame
    return df


def load_data_polars(
    data_folder: str,
    min_year: int = 2018,
    max_year: int = 2023,
    added_filters: Optional[Sequence[FilterCondition]] = None,
) -> pl.DataFrame :
    """Combine HMDA data files and keep only originations and purchases.

    This function mirrors :func:`load_data` but returns a :class:`polars.DataFrame`.

    Parameters
    ----------
    data_folder : str
        Folder where HMDA data files are located.
    min_year : int, optional
        Minimum year of data to include (inclusive). The default is 2018.
    max_year : int, optional
        Maximum year of data to include (inclusive). The default is 2023.
    added_filters : Sequence[tuple[str, str, object]], optional
        Additional pyarrow filter tuples to apply when reading the parquet
        files. Each filter follows the (column, operator, value) convention
        expected by :func:`pyarrow.parquet.read_table`.

    Returns
    -------
    polars.DataFrame
        Combined HMDA data covering the requested years.
    """

    hmda_filters: list[FilterCondition] = [('action_taken','in',[1,6])]
    if added_filters is not None :
        hmda_filters += list(added_filters)

    df_list: list[pl.DataFrame] = []
    for year in range(min_year, max_year+1) :
        file = HMDALoader.get_hmda_files(data_folder, min_year=year, max_year=year, extension='parquet')[0]
        hmda_columns = get_match_columns(file)
        table = pq.read_table(file, columns=hmda_columns, filters=hmda_filters)
        df_a = pl.from_arrow(table)
        df_a = df_a.filter(
            (~pl.col('purchaser_type').is_in([1, 2, 3, 4])) | (pl.col('action_taken') == 6)
        )
        df_list.append(df_a)

    if not df_list :
        return pl.DataFrame()

    return pl.concat(df_list, how='vertical_relaxed')

# Replace Missing Values
def replace_missing_values(df: pd.DataFrame) -> pd.DataFrame :
    """Replace numeric sentinel codes with ``None`` for easier comparisons.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with numerics for missing values.

    Returns
    -------
    pandas.DataFrame
        Data with NoneTypes for missing values.
    """

    # Note Loans Exempt from Fee Reporting
    df['i_ExemptFromFeesStrict'] = (df['total_loan_costs'] == 1111) & (df['total_points_and_fees'] == 1111) & (df['origination_charges'] == 1111) & (df['discount_points'] == 1111) & (df['lender_credits'] == 1111)
    df['i_ExemptFromFeesWeak'] = (df['total_loan_costs'] == 1111) | (df['total_points_and_fees'] == 1111) | (df['origination_charges'] == 1111) | (df['discount_points'] == 1111) | (df['lender_credits'] == 1111)

    # Columns to replace missing values
    replace_columns = ['conforming_loan_limit',
                       'construction_method',
                       'income',
                       'total_units',
                       'lien_status',
                       'multifamily_affordable_units',
                       'total_loan_costs',
                       'total_points_and_fees',
                       'discount_points',
                       'lender_credits',
                       'origination_charges',
                       'interest_rate',
                       'intro_rate_period',
                       'loan_term',
                       'property_value',
                       'balloon_payment',
                       'interest_only_payment',
                       'negative_amortization',
                       'open_end_line_of_credit',
                       'other_nonamortizing_features',
                       'prepayment_penalty_term',
                       'reverse_mortgage',
                       'business_or_commercial_purpose',
                       'manufactured_home_land_property_',
                       'manufactured_home_secured_proper',
                       ]

    # Rplace Missing Values
    for col in replace_columns :
        if col in df.columns :
            df.loc[df[col].isin([-1111,1111,99999,-99999]), col] = None
            df.loc[df[col] <= 0, col] = None

    # Replace Weird Introductory Rate Periods
    df.loc[df['intro_rate_period'] == df['loan_term'], 'intro_rate_period'] = None

    # Return DataFrame
    return df


def replace_missing_values_polars(df: pl.DataFrame) -> pl.DataFrame :
    """Replace numeric sentinel codes with ``None`` for easier comparisons.

    This function mirrors :func:`replace_missing_values` but operates on a
    :class:`polars.DataFrame`.

    Parameters
    ----------
    df : polars.DataFrame
        Data with numerics for missing values.

    Returns
    -------
    polars.DataFrame
        Data with NoneTypes for missing values.
    """

    fee_columns = [
        'total_loan_costs',
        'total_points_and_fees',
        'origination_charges',
        'discount_points',
        'lender_credits',
    ]

    if all(col in df.columns for col in fee_columns) :
        df = df.with_columns([
            (
                (pl.col('total_loan_costs') == 1111)
                & (pl.col('total_points_and_fees') == 1111)
                & (pl.col('origination_charges') == 1111)
                & (pl.col('discount_points') == 1111)
                & (pl.col('lender_credits') == 1111)
            ).alias('i_ExemptFromFeesStrict'),
            (
                (pl.col('total_loan_costs') == 1111)
                | (pl.col('total_points_and_fees') == 1111)
                | (pl.col('origination_charges') == 1111)
                | (pl.col('discount_points') == 1111)
                | (pl.col('lender_credits') == 1111)
            ).alias('i_ExemptFromFeesWeak'),
        ])

    replace_columns = ['conforming_loan_limit',
                       'construction_method',
                       'income',
                       'total_units',
                       'lien_status',
                       'multifamily_affordable_units',
                       'total_loan_costs',
                       'total_points_and_fees',
                       'discount_points',
                       'lender_credits',
                       'origination_charges',
                       'interest_rate',
                       'intro_rate_period',
                       'loan_term',
                       'property_value',
                       'balloon_payment',
                       'interest_only_payment',
                       'negative_amortization',
                       'open_end_line_of_credit',
                       'other_nonamortizing_features',
                       'prepayment_penalty_term',
                       'reverse_mortgage',
                       'business_or_commercial_purpose',
                       'manufactured_home_land_property_',
                       'manufactured_home_secured_proper',
                       ]

    sentinel_values = [-1111, 1111, 99999, -99999]
    replace_exprs: list[pl.Expr] = []
    for col in replace_columns :
        if col in df.columns :
            replace_exprs.append(
                pl.when(
                    pl.col(col).is_in(sentinel_values)
                    | (
                        pl.col(col)
                        .cast(pl.Float64, strict=False)
                        .is_not_null()
                        & (pl.col(col).cast(pl.Float64, strict=False) <= 0)
                    )
                )
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

    if replace_exprs :
        df = df.with_columns(replace_exprs)

    if {'intro_rate_period', 'loan_term'}.issubset(set(df.columns)) :
        df = df.with_columns(
            pl.when(pl.col('intro_rate_period') == pl.col('loan_term'))
            .then(None)
            .otherwise(pl.col('intro_rate_period'))
            .alias('intro_rate_period')
        )

    return df

# Convert Numerics
def convert_numerics(df: pd.DataFrame) -> pd.DataFrame :
    """Cast HMDA string codes to numeric dtypes where applicable.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw HMDA data where many numeric fields are represented as strings.

    Returns
    -------
    pandas.DataFrame
        Data frame with numeric columns converted to ``float``/``int`` types
        when possible.
    """

    # Replace Exempt w/ -99999
    exempt_cols = ['combined_loan_to_value_ratio',
                   'interest_rate',
                   'rate_spread',
                   'loan_term',
                   'prepayment_penalty_term',
                   'intro_rate_period',
                   'income',
                   'multifamily_affordable_units',
                   'property_value',
                   'total_loan_costs',
                   'total_points_and_fees',
                   'origination_charges',
                   'discount_points',
                   'lender_credits',
                   ]
    for col in exempt_cols :
        df.loc[df[col] == "Exempt", col] = -99999
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

    # Clean Units
    col = 'total_units'
    df.loc[df[col] == "5-24", col] = 5
    df.loc[df[col] == "25-49", col] = 6
    df.loc[df[col] == "50-99", col] = 7
    df.loc[df[col] == "100-149", col] = 8
    df.loc[df[col] == ">149", col] = 9
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean Age
    for col in ['applicant_age', 'co_applicant_age'] :
        df.loc[df[col] == "<25", col] = 1
        df.loc[df[col] == "25-34", col] = 2
        df.loc[df[col] == "35-44", col] = 3
        df.loc[df[col] == "45-54", col] = 4
        df.loc[df[col] == "55-64", col] = 5
        df.loc[df[col] == "65-74", col] = 6
        df.loc[df[col] == ">74", col] = 7
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

    # Clean Age Dummy Variables
    for col in ['applicant_age_above_62', 'co_applicant_age_above_62'] :
        df.loc[df[col].isin(["No","no","NO"]), col] = 0
        df.loc[df[col].isin(["Yes","yes","YES"]), col] = 1
        df.loc[df[col].isin(["Na","na","NA"]), col] = np.nan
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

	# Clean Debt-to-Income
    col = 'debt_to_income_ratio'
    df.loc[df[col] == "<20%", col] = 10
    df.loc[df[col] == "20%-<30%", col] = 20
    df.loc[df[col] == "30%-<36%", col] = 30
    df.loc[df[col] == "50%-60%", col] = 50
    df.loc[df[col] == ">60%", col] = 60
    df.loc[df[col] == "Exempt", col] = -99999
    df[col] = pd.to_numeric(df[col], errors = 'coerce')

	# Clean Conforming Loan Limit
    col = 'conforming_loan_limit'
    if col in df.columns :
        df.loc[df[col] == "NC", col] = 0
        df.loc[df[col] == "C", col] = 1
        df.loc[df[col] == "U", col] = 1111
        df.loc[df[col] == "NA", col] = -1111
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

    # Numeric and Categorical Columns
    numeric_columns = [
        'activity_year',
        'loan_type',
        'loan_purpose',
        'occupancy_type',
        'loan_amount',
        'action_taken',
        'msa_md',
        'county_code',
        'applicant_race_1',
        'applicant_race_2',
        'applicant_race_3',
        'applicant_race_4',
        'applicant_race_5',
        'co_applicant_race_1',
        'co_applicant_race_2',
        'co_applicant_race_3',
        'co_applicant_race_4',
        'co_applicant_race_5',
        'applicant_sex',
        'co_applicant_sex',
        'income',
        'purchaser_type',
        'denial_reason_1',
        'denial_reason_2',
        'denial_reason_3',
        'edit_status',
        'sequence_number',
        'rate_spread',
        'tract_population',
        'tract_minority_population_percent',
        'ffiec_msa_md_median_family_income',
        'tract_to_msa_income_percentage',
        'tract_owner_occupied_units',
        'tract_one_to_four_family_homes',
        'tract_median_age_of_housing_units',
    ]

    # Convert Columns to Numeric
    for numeric_column in numeric_columns :
        if numeric_column in df.columns :
            df[numeric_column] = pd.to_numeric(df[numeric_column], errors='coerce')

    # Return DataFrame
    return df


def convert_numerics_polars(df: pl.DataFrame) -> pl.DataFrame :
    """Cast HMDA string codes to numeric dtypes where applicable.

    This function mirrors :func:`convert_numerics` but operates on a
    :class:`polars.DataFrame`.

    Parameters
    ----------
    df : polars.DataFrame
        Raw HMDA data where many numeric fields are represented as strings.

    Returns
    -------
    polars.DataFrame
        Data frame with numeric columns converted to ``float``/``int`` types
        when possible.
    """

    exempt_cols = ['combined_loan_to_value_ratio',
                   'interest_rate',
                   'rate_spread',
                   'loan_term',
                   'prepayment_penalty_term',
                   'intro_rate_period',
                   'income',
                   'multifamily_affordable_units',
                   'property_value',
                   'total_loan_costs',
                   'total_points_and_fees',
                   'origination_charges',
                   'discount_points',
                   'lender_credits',
                   ]

    for col in exempt_cols :
        if col in df.columns :
            df = df.with_columns(
                pl.when(pl.col(col) == "Exempt")
                .then(pl.lit(-99999))
                .otherwise(pl.col(col))
                .alias(col)
            )
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))

    if 'total_units' in df.columns :
        mapping = {
            "5-24": 5,
            "25-49": 6,
            "50-99": 7,
            "100-149": 8,
            ">149": 9,
        }
        df = df.with_columns(
            pl.col('total_units')
            .replace(mapping, return_dtype=pl.Float64)
            .cast(pl.Float64, strict=False)
            .alias('total_units')
        )

    for col in ['applicant_age', 'co_applicant_age'] :
        if col in df.columns :
            mapping = {
                "<25": 1,
                "25-34": 2,
                "35-44": 3,
                "45-54": 4,
                "55-64": 5,
                "65-74": 6,
                ">74": 7,
            }
            df = df.with_columns(
                pl.col(col)
                .replace(mapping, return_dtype=pl.Float64)
                .cast(pl.Float64, strict=False)
                .alias(col)
            )

    for col in ['applicant_age_above_62', 'co_applicant_age_above_62'] :
        if col in df.columns :
            mapping = {"No": 0, "no": 0, "NO": 0, "Yes": 1, "yes": 1, "YES": 1}
            df = df.with_columns(
                pl.col(col)
                .replace(mapping, default=None, return_dtype=pl.Float64)
                .alias(col)
            )

    if 'debt_to_income_ratio' in df.columns :
        mapping = {
            "<20%": 10,
            "20%-<30%": 20,
            "30%-<36%": 30,
            "50%-60%": 50,
            ">60%": 60,
            "Exempt": -99999,
        }
        df = df.with_columns(
            pl.col('debt_to_income_ratio')
            .replace(mapping, default=None, return_dtype=pl.Float64)
            .cast(pl.Float64, strict=False)
            .alias('debt_to_income_ratio')
        )

    col = 'conforming_loan_limit'
    if col in df.columns :
        mapping = {"NC": 0, "C": 1, "U": 1111, "NA": -1111}
        df = df.with_columns(
            pl.col(col)
            .replace(mapping, default=None, return_dtype=pl.Float64)
            .cast(pl.Float64, strict=False)
            .alias(col)
        )

    numeric_columns = [
        'activity_year',
        'loan_type',
        'loan_purpose',
        'occupancy_type',
        'loan_amount',
        'action_taken',
        'msa_md',
        'county_code',
        'applicant_race_1',
        'applicant_race_2',
        'applicant_race_3',
        'applicant_race_4',
        'applicant_race_5',
        'co_applicant_race_1',
        'co_applicant_race_2',
        'co_applicant_race_3',
        'co_applicant_race_4',
        'co_applicant_race_5',
        'applicant_sex',
        'co_applicant_sex',
        'income',
        'purchaser_type',
        'denial_reason_1',
        'denial_reason_2',
        'denial_reason_3',
        'edit_status',
        'sequence_number',
        'rate_spread',
        'tract_population',
        'tract_minority_population_percent',
        'ffiec_msa_md_median_family_income',
        'tract_to_msa_income_percentage',
        'tract_owner_occupied_units',
        'tract_one_to_four_family_homes',
        'tract_median_age_of_housing_units',
    ]

    numeric_exprs = [
        pl.col(col).cast(pl.Float64, strict=False).alias(col)
        for col in numeric_columns
        if col in df.columns
    ]
    if numeric_exprs :
        df = df.with_columns(numeric_exprs)

    return df

# Keep Only Observations with Potential Matches on Match Columns
def keep_potential_matches(df: pd.DataFrame, match_columns: Sequence[str]) -> pd.DataFrame :
    """Filter to loans that have at least one potential counterpart.

    Parameters
    ----------
    df : pandas.DataFrame
        Data.
    match_columns : Sequence[str]
        Columns for match.

    Returns
    -------
    pandas.DataFrame
        Data with only observations that have potential match.
    """

    # Keep Potential Matches Based on County or Census Tract
    df['i_HasPurchase'] = df.groupby(match_columns)['action_taken'].transform(lambda x: max(x == 6))
    df['i_HasSale'] = df.groupby(match_columns)['action_taken'].transform(lambda x: max(x == 1))

    # Keep Loans With Potential Matches and Drop Potential Match Indicators
    df = df.query('i_HasSale and i_HasPurchase')

    # Drop Dummies
    df = df.drop(columns = ['i_HasSale', 'i_HasPurchase'])

    # Return DataFrame
    return df


def keep_potential_matches_polars(df: pl.DataFrame, match_columns: Sequence[str]) -> pl.DataFrame :
    """Filter to loans that have at least one potential counterpart."""

    df = df.with_columns([
        pl.col('action_taken').eq(6).any().over(match_columns).alias('i_HasPurchase'),
        pl.col('action_taken').eq(1).any().over(match_columns).alias('i_HasSale'),
    ])

    df = df.filter(pl.col('i_HasSale') & pl.col('i_HasPurchase'))

    return df.drop(['i_HasSale', 'i_HasPurchase'])

# Split Sellers and Purchasers
def split_sellers_and_purchasers(
    df: pd.DataFrame,
    crosswalk_folder: str,
    match_round: int = 1,
    file_suffix: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame] :
    """Split the pooled HMDA data into seller and purchaser subsets.

    Parameters
    ----------
    df : pandas.DataFrame
        Matched seller and purchaser candidates in long form.
    crosswalk_folder : str
        Directory containing previously generated match crosswalks.
    match_round : int, optional
        Match iteration. Rounds above one drop prior matches. The default is 1.
    file_suffix : str, optional
        Optional suffix appended to crosswalk filenames.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Seller observations followed by purchaser observations.
    """

    # If Crosswalk is Provided, Drop Existing Matches
    if match_round > 1 :

        # Load Crosswalk
        suffix = file_suffix or ""
        cw = pq.read_table(f'{crosswalk_folder}/hmda_seller_purchaser_matches_round{match_round-1}{suffix}.parquet')

        # Drop Sellers and Purchasers Already Matched
        df = pa.Table.from_pandas(df, preserve_index=False, safe=False)
        df = df.join(cw, keys=['HMDAIndex'], right_keys=['HMDAIndex_s'], join_type='left anti')
        df = df.join(cw, keys=['HMDAIndex'], right_keys=['HMDAIndex_p'], join_type='left anti')
        df = df.to_pandas()
        
    # Separate Out by Action Taken
    df_purchaser = df.query('action_taken == 6')
    df = df.query('action_taken == 1')
    df_seller = df

    # Return Sellers and Purchasers
    return df_seller, df_purchaser

# Match Sex
def match_sex(df: pd.DataFrame) -> pd.DataFrame :
    """Drop candidate pairs that conflict on applicant or co-applicant sex.

    Parameters
    ----------
    df : pandas.DataFrame
        Candidate seller/purchaser combinations.

    Returns
    -------
    pandas.DataFrame
        Candidate pairs where the reported sex codes are compatible.
    """
    
    # Replace Mismatches on Applicant and Co-Applicant Sex
    for sex_column in ['applicant_sex', 'co_applicant_sex'] :
        indexDrop = df[(df[f'{sex_column}_s'] == 1) & df[f'{sex_column}_p'].isin([2, 3, 5, 6])].index
        df = df.drop(indexDrop)
        indexDrop = df[(df[f'{sex_column}_s'] == 2) & df[f'{sex_column}_p'].isin([1, 3, 5, 6])].index
        df = df.drop(indexDrop)
        indexDrop = df[(df[f'{sex_column}_s'] == 3) & df[f'{sex_column}_p'].isin([1, 2, 5, 6])].index
        df = df.drop(indexDrop)
        indexDrop = df[(df[f'{sex_column}_s'] == 5) & df[f'{sex_column}_p'].isin([1, 2, 3, 6])].index
        df = df.drop(indexDrop)
        indexDrop = df[(df[f'{sex_column}_s'] == 6) & df[f'{sex_column}_p'].isin([1, 2, 3, 5])].index
        df = df.drop(indexDrop)

    # Return Matched DataFrame
    return df

# Match Age
def match_age(df: pd.DataFrame) -> pd.DataFrame :
    """Drop candidate pairs that conflict on applicant or co-applicant age.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with unmatched ages.

    Returns
    -------
    pandas.DataFrame
        Data with matched ages.
    """
    
    # Replace Mismatches on Applicant and Co-Applicant Sex
    df = df.query('applicant_age_s == applicant_age_p or applicant_age_s in [8888,9999] or applicant_age_p in [8888,9999]')
    df = df.query('co_applicant_age_s == co_applicant_age_p or co_applicant_age_s in [8888,9999] or co_applicant_age_p in [8888,9999]')

    # Co Applicant Age
    df = df.query('co_applicant_age_s != 9999 or co_applicant_age_p in [8888,9999]')
    df = df.query('co_applicant_age_p != 9999 or co_applicant_age_s in [8888,9999]')

    # Return Matched DataFrame
    return df

# Match Race
def match_race(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame :
    """Apply a series of race-based consistency checks to candidate matches.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with unmatched races.
    strict : bool, optional
        Whether to require exact matches on the primary race field. The
        default is ``False``.

    Returns
    -------
    pandas.DataFrame
        Data with matched races.
    """

    # Replace Race Subcategories
    for race_column in ['applicant_race', 'co_applicant_race'] :
        for race_number in range(1, 5+1) :
            df.loc[df[f'{race_column}_{race_number}_s'].isin([21, 22, 23, 24, 25, 26, 27]), f'{race_column}_{race_number}_s'] = 2
            df.loc[df[f'{race_column}_{race_number}_p'].isin([21, 22, 23, 24, 25, 26, 27]), f'{race_column}_{race_number}_p'] = 2
            df.loc[df[f'{race_column}_{race_number}_s'].isin([41, 42, 43, 44]), f'{race_column}_{race_number}_s'] = 4
            df.loc[df[f'{race_column}_{race_number}_p'].isin([41, 42, 43, 44]), f'{race_column}_{race_number}_p'] = 4

    # Applicant Race Match
    for race_number in range(1, 6+1) :
        df = df.query(f'applicant_race_1_s != {race_number} or applicant_race_1_p in [{race_number}, 7, 8] or applicant_race_2_p == {race_number} or applicant_race_3_p == {race_number} or applicant_race_4_p == {race_number} or applicant_race_5_p == {race_number}')
        df = df.query(f'applicant_race_1_p != {race_number} or applicant_race_1_s in [{race_number}, 7, 8] or applicant_race_2_s == {race_number} or applicant_race_3_s == {race_number} or applicant_race_4_s == {race_number} or applicant_race_5_s == {race_number}')

    # Co-Applicant Race Match
    df = df.query('co_applicant_race_1_s != 8 or co_applicant_race_1_p in [7, 8]')
    df = df.query('co_applicant_race_1_p != 8 or co_applicant_race_1_s in [7, 8]')
    for race_number in range(1, 6+1) :
        df = df.query(f'co_applicant_race_1_s != {race_number} or co_applicant_race_1_p in [{race_number}, 7, 8] or co_applicant_race_2_p == {race_number} or co_applicant_race_3_p == {race_number} or co_applicant_race_4_p == {race_number} or co_applicant_race_5_p == {race_number}')
        df = df.query(f'co_applicant_race_1_p != {race_number} or co_applicant_race_1_s in [{race_number}, 7, 8] or co_applicant_race_2_s == {race_number} or co_applicant_race_3_s == {race_number} or co_applicant_race_4_s == {race_number} or co_applicant_race_5_s == {race_number}')

    # Strict Race Matches
    if strict :
        df = df.query('applicant_race_1_s == applicant_race_1_p or applicant_race_1_s in [7,8] or applicant_race_1_p in [7,8]')
        df = df.query('co_applicant_race_1_s == co_applicant_race_1_p or co_applicant_race_1_s in [7,8] or co_applicant_race_1_p in [7,8]')

    # Return Matched DataFrame
    return df

# Match Ethnicity
def match_ethnicity(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame :
    """Apply a series of ethnicity-based consistency checks to candidates.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with unmatched ethnicities.
    strict : bool, optional
        Whether to require exact matches on the primary ethnicity field. The
        default is ``False``.

    Returns
    -------
    pandas.DataFrame
        Data with matched ethnicities.
    """

    # Replace Race Subcategories
    for ethnicity_column in ['applicant_ethnicity', 'co_applicant_ethnicity'] :
        for ethnicity_number in range(1, 5+1) :
            df.loc[df[f'{ethnicity_column}_{ethnicity_number}_s'].isin([11, 12, 13, 14]), f'{ethnicity_column}_{ethnicity_number}_s'] = 1
            df.loc[df[f'{ethnicity_column}_{ethnicity_number}_p'].isin([11, 12, 13, 14]), f'{ethnicity_column}_{ethnicity_number}_p'] = 1
        
    # Applicant Race Match
    for ethnicity_number in range(1, 3+1) :
        df = df.query(f'applicant_ethnicity_1_s != {ethnicity_number} or applicant_ethnicity_1_p in [{ethnicity_number}, 4, 5] or applicant_ethnicity_2_p == {ethnicity_number} or applicant_ethnicity_3_p == {ethnicity_number} or applicant_ethnicity_4_p == {ethnicity_number} or applicant_ethnicity_5_p == {ethnicity_number}')
        df = df.query(f'applicant_ethnicity_1_p != {ethnicity_number} or applicant_ethnicity_1_s in [{ethnicity_number}, 4, 5] or applicant_ethnicity_2_s == {ethnicity_number} or applicant_ethnicity_3_s == {ethnicity_number} or applicant_ethnicity_4_s == {ethnicity_number} or applicant_ethnicity_5_s == {ethnicity_number}')

    # Co-Applicant Race Match
    df = df.query('co_applicant_ethnicity_1_s != 5 or co_applicant_ethnicity_1_p in [4, 5]')
    df = df.query('co_applicant_ethnicity_1_p != 5 or co_applicant_ethnicity_1_s in [4, 5]')
    for ethnicity_number in range(1, 3+1) :
        df = df.query(f'co_applicant_ethnicity_1_s != {ethnicity_number} or co_applicant_ethnicity_1_p in [{ethnicity_number}, 4, 5] or co_applicant_ethnicity_2_p == {ethnicity_number} or co_applicant_ethnicity_3_p == {ethnicity_number} or co_applicant_ethnicity_4_p == {ethnicity_number} or co_applicant_ethnicity_5_p == {ethnicity_number}')
        df = df.query(f'co_applicant_ethnicity_1_p != {ethnicity_number} or co_applicant_ethnicity_1_s in [{ethnicity_number}, 4, 5] or co_applicant_ethnicity_2_s == {ethnicity_number} or co_applicant_ethnicity_3_s == {ethnicity_number} or co_applicant_ethnicity_4_s == {ethnicity_number} or co_applicant_ethnicity_5_s == {ethnicity_number}')

    # Replace Mismatches on Applicant and Co-Applicant Sex
    if strict :
        df = df.query('applicant_ethnicity_1_s == applicant_ethnicity_1_p or applicant_ethnicity_1_s in [4] or applicant_ethnicity_1_p in [4]')
        df = df.query('co_applicant_ethnicity_1_s == co_applicant_ethnicity_1_p or co_applicant_ethnicity_1_s in [4] or co_applicant_ethnicity_1_p in [4]')

    # Return Matched DataFrame
    return df

# Perform Numeric Matches
def perform_income_matches(df: pd.DataFrame) -> pd.DataFrame :
    """Create auxiliary income fields to support tolerant matching rules.

    Parameters
    ----------
    df : pandas.DataFrame
        Data before income differences are removed.

    Returns
    -------
    pandas.DataFrame
        Data after income differences are removed.
    """

    # Alternative Income Variables
    df['income_fix_s'] = df['income_s']/1000
    df.loc[df['income_s'] > 10000, 'income_fix_s'] = None
    df['income_fix_p'] = df['income_p']/1000
    df.loc[df['income_p'] > 10000, 'income_fix_p'] = None

    #
    df['income_diff'] = df['income_s'] - df['income_p']
    df['income_diff_fix_1'] = df['income_fix_s'] - df['income_p']
    df['income_diff_fix_2'] = df['income_s'] - df['income_fix_p']
    # gen i_ExactIncomeMatch = abs(income_diff) == 0 | abs(income_diff_fix_1) < 1 | abs(income_diff_fix_2) < 1 if ~missing(income_diff)

    # Drop New Columns
    df = df.drop(columns = ['income_fix_s','income_fix_p','income_diff','income_diff_fix_1','income_diff_fix_2'])

    # Return DataFrame
    return df

# Numeric Matches
def numeric_matches(
    df: pd.DataFrame,
    match_tolerances: Mapping[str, float | int],
    drop_differences: bool = True,
) -> pd.DataFrame :
    """Keep only records that agree within provided numeric tolerances.

    Parameters
    ----------
    df : pandas.DataFrame
        Data.
    match_tolerances : Mapping[str, float | int]
        Dictionary of match columns and tolerances.
    drop_differences : bool, optional
        Whether to drop the created value differences. The default is True.

    Returns
    -------
    pandas.DataFrame
        Data.
    """

    # Drop One Column at a Time
    for column,tolerance in match_tolerances.items() :

        if (f'{column}_s' in df.columns) and (f'{column}_p' in df.columns) :

            # Count for Dropped Observations
            start_obs = df.shape[0]

            # Compute Numeric Differences
            df[f'{column}_difference'] = df[f'{column}_s'] - df[f'{column}_p']

            # Drop Large Numeric Differences
            # df = df.loc[~(np.abs(df[f'{column}_difference']) > tolerance)]
            df = df.query(f'abs({column}_difference) <= {tolerance} or {column}_difference.isnull()')

            # Display Progress
            logger.debug(
                "Matching on %s drops %d observations",
                column,
                start_obs - df.shape[0],
            )

            # Drop Difference Columns
            if drop_differences :
                df = df.drop(columns = [f'{column}_difference'])

    # Return DataFrame
    return df

# Numeric Matches
def weak_numeric_matches(
    df: pd.DataFrame,
    match_tolerances: Mapping[str, float | int],
    drop_differences: bool = True,
) -> pd.DataFrame :
    """Allow slight numeric disagreement while preserving best matches.

    Parameters
    ----------
    df : pandas.DataFrame
        Data.
    match_tolerances : Mapping[str, float | int]
        Dictionary of match columns and tolerances.
    drop_differences : bool, optional
        Whether to drop the created value differences. The default is True.

    Returns
    -------
    pandas.DataFrame
        Data.
    """

    # Drop One Column at a Time
    for column,tolerance in match_tolerances.items() :
        
        if (f'{column}_s' in df.columns) and (f'{column}_p' in df.columns) :

            # Count for Dropped Observations
            start_obs = df.shape[0]

            # Compute Numeric Differences
            df[f'{column}_difference'] = df[f'{column}_s'] - df[f'{column}_p']

            # Compute Minimum Differences
            df['temp'] = np.abs(df[f'{column}_difference'])
            df[f'min_{column}_difference_s'] = df.groupby(['HMDAIndex_s'])['temp'].transform('min')
            df[f'min_{column}_difference_p'] = df.groupby(['HMDAIndex_p'])['temp'].transform('min')
            df = df.drop(columns = ['temp'])

            # Drop Large Numeric Differences
            df = df.loc[~(np.abs(df[f'{column}_difference']) > tolerance) | (df[f'min_{column}_difference_s'] > 0) | pd.isna(df[f'min_{column}_difference_s'])]
            df = df.loc[~(np.abs(df[f'{column}_difference']) > tolerance) | (df[f'min_{column}_difference_p'] > 0) | pd.isna(df[f'min_{column}_difference_p'])]

            # Display Progress
            logger.debug(
                "Matching weakly on %s drops %d observations",
                column,
                start_obs - df.shape[0],
            )

            # Drop Difference Columns
            if drop_differences :
                df = df.drop(columns = [f'{column}_difference',f'min_{column}_difference_s',f'min_{column}_difference_p'], errors='ignore')

    return df

# Numeric Matches after Uniques
def numeric_matches_post_unique(
    df: pd.DataFrame,
    match_tolerances: Mapping[str, float | int],
    drop_differences: bool = True,
) -> pd.DataFrame :
    """Re-apply numeric tolerances after enforcing uniqueness constraints.

    Parameters
    ----------
    df : pandas.DataFrame
        Data.
    match_tolerances : Mapping[str, float | int]
        Dictionary of match columns and tolerances.
    drop_differences : bool, optional
        Whether to drop the created value differences. The default is True.

    Returns
    -------
    pandas.DataFrame
        Data.
    """
    
    # Count for Dropped Observations
    start_obs = df.shape[0]
    
    df['i_DropObservation'] = 0

    # Drop One Column at a Time
    for column,tolerance in match_tolerances.items() :

        # Compute Numeric Differences
        df[f'{column}_difference'] = df[f'{column}_s'] - df[f'{column}_p']

        # Drop Large Numeric Differences
        df.loc[np.abs(df[f'{column}_difference']) > tolerance, 'i_DropObservation'] = 1

        # Drop Difference Columns
        if drop_differences :
            df = df.drop(columns = [f'{column}_difference'])

    # Drop All Sold Loans When One Bad Match Exists
    df['i_DropSale'] = df.groupby(['HMDAIndex_s'])['i_DropObservation'].transform('max')
    df = df.query('i_DropSale == 0')
    df = df.drop(columns = ['i_DropObservation','i_DropSale'])

    # Display Progress
    logger.debug(
        "Numeric post-unique filtering removed %d observations",
        start_obs - df.shape[0],
    )

    # Return DataFrame
    return df

# Perform Fee Matches
def perform_fee_matches(df: pd.DataFrame) -> pd.DataFrame :
    """Count non-missing fee variables and flag matching fee structures.

    Parameters
    ----------
    df : pandas.DataFrame
        Data without match or nonmissing counts.

    Returns
    -------
    pandas.DataFrame
        Data with match and nonmissing counts.
    """

    # Initialize Fee Match Variables
    df['NumberFeeMatches'] = 0 #if i_ExemptFromFeesStrict != 1
    df['NumberNonmissingFees_s'] = 0 #if i_ExemptFromFeesStrict != 1
    df['NumberNonmissingFees_p'] = 0 #if i_ExemptFromFeesStrict != 1

    # Update Fee Match Variables
    for fee_column in ['total_loan_costs', 'total_points_and_fees', 'origination_charges', 'discount_points', 'lender_credits'] :
        df['NumberFeeMatches'] = df['NumberFeeMatches'] + (df[f'{fee_column}_s'] == df[f'{fee_column}_p'])*(df[f'{fee_column}_s'] is not None)
        df['NumberNonmissingFees_s'] = df['NumberNonmissingFees_s'] + ~pd.isna(df[f'{fee_column}_s'])
        df['NumberNonmissingFees_p'] = df['NumberNonmissingFees_p'] + ~pd.isna(df[f'{fee_column}_p'])

    # Generous Fee Match
    df['i_GenerousFeeMatch'] = 0 #if i_ExemptFromFeesStrict != 1
    for var1 in ['total_loan_costs', 'total_points_and_fees', 'origination_charges', 'discount_points', 'lender_credits'] :
    	for var2 in ['total_loan_costs', 'total_points_and_fees', 'origination_charges', 'discount_points', 'lender_credits'] :
            df.loc[((df[f'{var1}_s'] == df[f'{var2}_p']) & ~pd.isna(df[f'{var1}_s'])), 'i_GenerousFeeMatch'] = 1

    # Return DataFrame
    return df


def perform_fee_matches_polars(df: pl.DataFrame) -> pl.DataFrame :
    """Count non-missing fee variables and flag matching fee structures."""

    fee_columns = ['total_loan_costs', 'total_points_and_fees', 'origination_charges', 'discount_points', 'lender_credits']
    seller_cols = {f'{col}_s' for col in fee_columns}
    purchaser_cols = {f'{col}_p' for col in fee_columns}
    if not seller_cols.issubset(df.columns) or not purchaser_cols.issubset(df.columns) :
        return df

    match_exprs = [
        (
            (pl.col(f'{fee}_s') == pl.col(f'{fee}_p'))
            & pl.col(f'{fee}_s').is_not_null()
        ).cast(pl.Int64)
        for fee in fee_columns
    ]
    nonmissing_s_exprs = [pl.col(f'{fee}_s').is_not_null().cast(pl.Int64) for fee in fee_columns]
    nonmissing_p_exprs = [pl.col(f'{fee}_p').is_not_null().cast(pl.Int64) for fee in fee_columns]

    generous_exprs = [
        (
            (pl.col(f'{var1}_s') == pl.col(f'{var2}_p'))
            & pl.col(f'{var1}_s').is_not_null()
        )
        for var1 in fee_columns
        for var2 in fee_columns
    ]

    df = df.with_columns([
        pl.sum_horizontal(match_exprs).alias('NumberFeeMatches'),
        pl.sum_horizontal(nonmissing_s_exprs).alias('NumberNonmissingFees_s'),
        pl.sum_horizontal(nonmissing_p_exprs).alias('NumberNonmissingFees_p'),
        pl.any_horizontal(generous_exprs).cast(pl.Int64).alias('i_GenerousFeeMatch'),
    ])

    return df

# Keep Uniques
def keep_uniques(df: pd.DataFrame, one_to_one: bool = True) -> pd.DataFrame :
    """Restrict matches so each purchaser links to a single sale.

    Parameters
    ----------
    df : pandas.DataFrame
        Data before unique matches are enforced.
    one_to_one : bool, optional
        Whether to only keep unique seller matches. The default is True.

    Returns
    -------
    pandas.DataFrame
        Data after unique matches are enforced.
    """

    # Keep Unique Loans
    df['count_index_s'] = df.groupby(['HMDAIndex_s'])['HMDAIndex_s'].transform('count')
    df['count_index_p'] = df.groupby(['HMDAIndex_p'])['HMDAIndex_p'].transform('count')

    # Display
    logger.debug(
        "Match cardinality counts:\n%s",
        df[['count_index_s', 'count_index_p']].value_counts(),
    )

    # Keep Purchased Loans w/ Unique Match
    df = df.query('count_index_p == 1')

    # Keep Uniques
    if one_to_one :
        df = df.query('count_index_s == 1')

    # Keep Loans Where Sale Matches Multiple Purchases if One Known to Be Secondary Sale
    else :
        
        # Keep Unique Loans
        df['temp'] = 1*(df['purchaser_type_p'] > 4)
        df['i_LoanHasSecondarySale'] = df.groupby(['HMDAIndex_s'])['temp'].transform('max')
        df = df.query('count_index_s == 1 or (count_index_s == 2 & i_LoanHasSecondarySale == 1)')
        df = df.drop(columns = ['i_LoanHasSecondarySale'])

    # Drop Index Counts
    df = df.drop(columns = ['count_index_s','count_index_p'])

    # Return DataFrame
    return df


def keep_uniques_polars(df: pl.DataFrame, one_to_one: bool = True) -> pl.DataFrame :
    """Restrict matches so each purchaser links to a single sale."""

    df = df.with_columns([
        pl.len().over('HMDAIndex_s').alias('count_index_s'),
        pl.len().over('HMDAIndex_p').alias('count_index_p'),
    ])

    logger.debug(
        "Match cardinality counts:\n%s",
        df.select(['count_index_s', 'count_index_p']).to_pandas().value_counts(),
    )

    df = df.filter(pl.col('count_index_p') == 1)

    if one_to_one :
        df = df.filter(pl.col('count_index_s') == 1)
    else :
        if 'purchaser_type_p' in df.columns :
            df = df.with_columns(
                (pl.col('purchaser_type_p') > 4).cast(pl.Int8).alias('temp')
            )
            df = df.with_columns(
                pl.col('temp').max().over('HMDAIndex_s').alias('i_LoanHasSecondarySale')
            )
            df = df.filter(
                (pl.col('count_index_s') == 1)
                | (
                    (pl.col('count_index_s') == 2)
                    & (pl.col('i_LoanHasSecondarySale') == 1)
                )
            )
            df = df.drop('i_LoanHasSecondarySale')
            df = df.drop('temp')

    df = df.drop(['count_index_s', 'count_index_p'])

    return df

# Save Crosswalk
def save_crosswalk(
    df: pd.DataFrame,
    save_folder: str,
    match_round: int = 1,
    file_suffix: Optional[str] = None,
) -> None :
    """Persist the current set of matches to a parquet crosswalk file.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing seller and purchaser HMDA indices.
    save_folder : str
        Destination directory for the crosswalk parquet file.
    match_round : int, optional
        Iteration number indicating which matching round produced the crosswalk.
        The default is 1.
    file_suffix : str, optional
        Optional suffix appended to crosswalk filenames.

    Returns
    -------
    None
    """
    
    # Add Previous Round Crosswalk
    cw = []
    if match_round > 1 :
        suffix = file_suffix or ""
        cw.append(pq.read_table(f'{save_folder}/hmda_seller_purchaser_matches_round{match_round-1}{suffix}.parquet').to_pandas())

    # Extract HMDA Index variables    
    cw_a = df[['HMDAIndex_s','HMDAIndex_p']]

    # Add Match Round Variable
    cw_a['match_round'] = match_round

    # Append and Concatenate
    cw.append(cw_a)
    cw = pd.concat(cw)

    # Save Crosswalk
    logger.info("Crosswalk match counts by round:\n%s", cw.match_round.value_counts())
    cw = pa.Table.from_pandas(cw, preserve_index=False)
    suffix = file_suffix or ""
    pq.write_table(cw, f'{save_folder}/hmda_seller_purchaser_matches_round{match_round}{suffix}.parquet')


def save_crosswalk_polars(
    df: pl.DataFrame,
    save_folder: str,
    match_round: int = 1,
    file_suffix: Optional[str] = None,
) -> None :
    """Persist the current set of matches to a parquet crosswalk file."""

    cw_tables: list[pl.DataFrame] = []
    if match_round > 1 :
        suffix = file_suffix or ""
        previous = pl.read_parquet(
            f'{save_folder}/hmda_seller_purchaser_matches_round{match_round-1}{suffix}.parquet'
        )
        cw_tables.append(previous)

    cw_a = df.select(['HMDAIndex_s', 'HMDAIndex_p'])
    cw_a = cw_a.with_columns(pl.lit(match_round).alias('match_round'))
    cw_tables.append(cw_a)

    cw = pl.concat(cw_tables, how='vertical_relaxed') if len(cw_tables) > 1 else cw_tables[0]

    logger.info(
        "Crosswalk match counts by round:\n%s",
        cw.group_by('match_round').count().to_pandas().set_index('match_round')['count'],
    )

    suffix = file_suffix or ""
    pq.write_table(cw.to_arrow(), f'{save_folder}/hmda_seller_purchaser_matches_round{match_round}{suffix}.parquet')
