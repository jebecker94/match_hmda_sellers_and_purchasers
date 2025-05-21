# Import Packages
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import HMDALoader

# Get Match Columns
def get_match_columns(file) :
    """
    Get columns used for matching.

    Parameters
    ----------
    file : str
        File to load for columns.

    Returns
    -------
    columns : list
        Columns used for matching.

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
def load_data(data_folder, min_year=2018, max_year=2023, added_filters=[]) :
    """
    Combine HMDA data after 2018, keeping originations and purchases only. For
    use primarily in matching after the first round.

    Parameters
    ----------
    data_folder : str
        Folder where HMDA data files are located.
    min_year : int, optional
        Minimum year of data to include (inclusive). The default is 2018.
    max_year : int, optional
        Maximum year of data to include (inclusive). The default is 2023.

    Returns
    -------
    df : pandas DataFrame
        Combined HMDA data.

    """

    # Set Filters
    hmda_filters = [('action_taken','in',[1,6])]
    hmda_filters += added_filters

    # Combine Seller and Purchaser Data
    df = []
    for year in range(min_year, max_year+1) :
        file = HMDALoader.get_hmda_files(data_folder, min_year=year, max_year=year, extension='parquet')[0]
        hmda_columns = get_match_columns(file)
        df_a = pd.read_parquet(file, columns=hmda_columns, filters=hmda_filters)
        df_a = df_a.query('purchaser_type not in [1,2,3,4] | action_taken == 6')
        df.append(df_a)
        del df_a
    df = pd.concat(df)

    # Return DataFrame
    return df

# Replace Missing Values
def replace_missing_values(df) :
    """
    Replace missing numerics with NoneTypes.

    Parameters
    ----------
    df : pandas DataFrame
        Data with numerics for missing values.

    Returns
    -------
    df : pandas DataFrame
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

# Convert Numerics
def convert_numerics(df) :
    """
    Destring numeric HMDA variables after 2018.

    Parameters
    ----------
    df : pandas DataFrame
        DESCRIPTION.

    Returns
    -------
    df : pandas DataFrame
        DESCRIPTION.

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

# Keep Only Observations with Potential Matches on Match Columns
def keep_potential_matches(df, match_columns) :
    """
    Before splitting, keep only loans which have at least one candidate match.

    Parameters
    ----------
    df : pandas DataFrame
        Data.
    match_columns : list
        Columns for match.

    Returns
    -------
    df : pandas DataFrame
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

# Split Sellers and Purchasers
def split_sellers_and_purchasers(df, crosswalk_folder, match_round=1, file_suffix=None) :
    """
    Split data into sellers and purchasers.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    cw : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df_seller : TYPE
        DESCRIPTION.
    df_purchaser : TYPE
        DESCRIPTION.

    """

    # If Crosswalk is Provided, Drop Existing Matches
    if match_round > 1 :

        # Load Crosswalk
        cw = pq.read_table(f'{crosswalk_folder}/hmda_seller_purchaser_matches_round{match_round-1}{file_suffix}.parquet')

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
def match_sex(df) :
    """
    Match on Applicant and Co-applicant Sex.

    Parameters
    ----------
    df : pandas DataFrame
        DESCRIPTION.

    Returns
    -------
    df : pandas DataFrame
        DESCRIPTION.

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
def match_age(df) :
    """
    Match on Applicant and Co-applicant Age.

    Parameters
    ----------
    df : pandas DataFrame
        Data with unmatched ages.

    Returns
    -------
    df : pandas DataFrame
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
def match_race(df, strict = False) :
    """
    Perform race matches

    Parameters
    ----------
    df : pandas DataFrame
        Data with unmatched races.

    Returns
    -------
    df : pandas DataFrame
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
def match_ethnicity(df, strict = False) :
    """
    Perform ethnicity matches

    Parameters
    ----------
    df : pandas DataFrame
        Data with unmatched ethnicities.

    Returns
    -------
    df : pandas DataFrame
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
def perform_income_matches(df) :
    """
    Matches with alternative income variables.

    Parameters
    ----------
    df : pandas DataFrame
        Data before income differences are removed.

    Returns
    -------
    df : ppandas DataFrame
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
def numeric_matches(df, match_tolerances, verbose = False, drop_differences = True) :
    """
    Matches for numeric columns.

    Parameters
    ----------
    df : pandas DataFrame
        Data.
    match_tolerances : dictionary
        Dictionary of match columns and tolerances.
    verbose : boolean, optional
        Whether to display number of dropped observations. The default is False.
    drop_differences : boolean, optional
        Whether to drop the created value differences. The default is True.

    Returns
    -------
    df : pandas DataFrame
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
            if verbose :
                print('Matching on', column, 'drops',  start_obs-df.shape[0], 'observations')

            # Drop Difference Columns
            if drop_differences :
                df = df.drop(columns = [f'{column}_difference'])

    # Return DataFrame
    return df

# Numeric Matches
def weak_numeric_matches(df, match_tolerances, verbose = False, drop_differences = True) :
    """
    Matches for numeric columns.

    Parameters
    ----------
    df : pandas DataFrame
        Data.
    match_tolerances : dictionary
        Dictionary of match columns and tolerances.
    verbose : boolean, optional
        Whether to display number of dropped observations. The default is False.
    drop_differences : boolean, optional
        Whether to drop the created value differences. The default is True.

    Returns
    -------
    df : pandas DataFrame
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
            if verbose :
                print('Matching weakly on', column, 'drops',  start_obs-df.shape[0], 'observations')

            # Drop Difference Columns
            if drop_differences :
                df = df.drop(columns = [f'{column}_difference',f'min_{column}_difference_s',f'min_{column}_difference_p'], errors='ignore')

    return df

# Numeric Matches after Uniques
def numeric_matches_post_unique(df, match_tolerances, verbose = False, drop_differences = True) :
    """
    Matches for numeric columns.

    Parameters
    ----------
    df : pandas DataFrame
        Data.
    match_tolerances : dictionary
        Dictionary of match columns and tolerances.
    verbose : boolean, optional
        Whether to display number of dropped observations. The default is False.
    drop_differences : boolean, optional
        Whether to drop the created value differences. The default is True.

    Returns
    -------
    df : pandas DataFrame
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
    if verbose :
        print('Matching on', column, 'drops',  start_obs-df.shape[0], 'observations')

    # Return DataFrame
    return df

# Perform Fee Matches
def perform_fee_matches(df) :
    """
    Count the number of fee variables with nonmissing values and matches.

    Parameters
    ----------
    df : pandas DataFrame
        Data without match or nonmissing counts.

    Returns
    -------
    df : pandas DataFrame
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

# Keep Uniques
def keep_uniques(df, one_to_one = True, verbose = True) :
    """
    Keep unique matches or matches where a single origination matches to many
    purchasers where one purchaser has a secondary sale.

    Parameters
    ----------
    df : pandas DataFrame
        Data before unique matches are enforced.
    one_to_one : Boolean, optional
        Whether to only keep unique seller matches. The default is True.
    verbose : Boolean, optional
        Whether to display match counts before dropping. The default is True.

    Returns
    -------
    df : pandas DataFrame
        Data after unique matches are enforced.

    """

    # Keep Unique Loans
    df['count_index_s'] = df.groupby(['HMDAIndex_s'])['HMDAIndex_s'].transform('count')
    df['count_index_p'] = df.groupby(['HMDAIndex_p'])['HMDAIndex_p'].transform('count')

    # Display
    if verbose :
        print(df[['count_index_s','count_index_p']].value_counts())

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

# Save Crosswalk
def save_crosswalk(df, save_folder, match_round = 1, file_suffix=None) :
    """
    Create and save a crosswalk from the data.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    save_folder : TYPE
        DESCRIPTION.
    match_round : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    
    # Add Previous Round Crosswalk
    cw = []
    if match_round > 1 :
        cw.append(pq.read_table(f'{save_folder}/hmda_seller_purchaser_matches_round{match_round-1}{file_suffix}.parquet').to_pandas())

    # Extract HMDA Index variables    
    cw_a = df[['HMDAIndex_s','HMDAIndex_p']]

    # Add Match Round Variable
    cw_a['match_round'] = match_round

    # Append and Concatenate
    cw.append(cw_a)
    cw = pd.concat(cw)

    # Save Crosswalk
    print(cw.match_round.value_counts())
    cw = pa.Table.from_pandas(cw, preserve_index=False)
    pq.write_table(cw, f'{save_folder}/hmda_seller_purchaser_matches_round{match_round}{file_suffix}.parquet')
