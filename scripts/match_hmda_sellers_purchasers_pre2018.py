#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:45:24 2023
Last updated on: Saturday March 22 07:45:00 2025
@author: Jonathan E. Becker
"""

# Import Packages
import glob
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import config

#%% Local Functions
# Save Crosswalk
def save_crosswalk(df, save_folder, match_round = 1) :
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
        cw.append(pd.read_csv(f'{save_folder}/hmda_seller_purchaser_matches_round{match_round-1}.csv'))

    # Extract HMDA Index variables    
    cw_a = df[['HMDAIndex_s','HMDAIndex_p']]

    # Add Match Round Variable
    cw_a['match_round'] = match_round

    # Append and Concatenate
    cw.append(cw_a)
    cw = pd.concat(cw)

    # Save Crosswalk
    cw.to_csv(f'{save_folder}/hmda_seller_purchaser_matches_round{match_round}.csv',
              index = False,
              )
    
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
        
        # Count for Dropped Observations
        start_obs = df.shape[0]

        # Compute Numeric Differences
        df[f'{column}_difference'] = df[f'{column}_s'] - df[f'{column}_p']

        # Drop Large Numeric Differences
        df = df.loc[~(np.abs(df[f'{column}_difference']) > tolerance)]
        
        # Display Progress
        if verbose :
            print('Matching on', column, 'drops',  start_obs-df.shape[0], 'observations')

        # Drop Difference Columns
        if drop_differences :
            df = df.drop(columns = [f'{column}_difference'])

    # Return DataFrame
    return df

# Drop Nonmatch Columns
def drop_nonmatch_columns(df) :
    """
    Drop HMDA columns not used for matches.

    Parameters
    ----------
    df : pandas DataFrame
        Data with all HMDA columns.

    Returns
    -------
    df : pandas DataFrame
        Data without irrelevant HMDA columns.

    """
    
    # Drop Columns
    drop_cols = ['denial_reason',
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
                 'submission_of_application']
    
    drop_cols = [x for x in drop_cols if x in df.columns]
    
    df = df.drop(columns = drop_cols)
    
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

    # Columns to replace missing values
    replace_columns = ['conforming_loan_limit',
                       'construction_method',
                       'income',
                       'total_units',
                       'lien_status',
                       'open_end_line_of_credit',
                       'multifamily_affordable_units',
                       'discount_points',
                       'lender_credits',
                       'origination_charges',
                       'interest_rate',
                       'intro_rate_period',
                       'loan_term',
                       'property_value']

    # Rplace Missing Values
    for col in replace_columns :
        df.loc[df[col].isin([-1111,1111,99999,-99999]), col] = None

    # Return DataFrame
    return df

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

    # Drop Bad Sex Matches
    df = df.query('applicant_sex_s != 1 | applicant_sex_p != 2')
    df = df.query('applicant_sex_s != 2 | applicant_sex_p != 1')
    df = df.query('co_applicant_sex_s != 1 | co_applicant_sex_p != 2')
    df = df.query('co_applicant_sex_s != 2 | co_applicant_sex_p != 1')
    df = df.query('co_applicant_sex_s != 5 | co_applicant_sex_p not in [1,2]')
    df = df.query('co_applicant_sex_s not in [1,2] | co_applicant_sex_p != 5')

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

# Match Race
def match_ethnicity(df) :
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


    # Replace Mismatches on Applicant and Co-Applicant Sex
    for ethnicity_number in range(1,3+1) :
        df = df.query(f'applicant_ethnicity_s != {ethnicity_number} or applicant_ethnicity_p in [{ethnicity_number}, 4]')
        df = df.query(f'applicant_ethnicity_p != {ethnicity_number} or applicant_ethnicity_s in [{ethnicity_number}, 4]')
        df = df.query(f'co_applicant_ethnicity_s != {ethnicity_number} or co_applicant_ethnicity_p in [{ethnicity_number}, 4]')
        df = df.query(f'co_applicant_ethnicity_p != {ethnicity_number} or co_applicant_ethnicity_s in [{ethnicity_number}, 4]')
    
    # Co-Applicant Present
    df = df.query('co_applicant_ethnicity_s != 5 or co_applicant_ethnicity_p in [4, 5]')
    df = df.query('co_applicant_ethnicity_p != 5 or co_applicant_ethnicity_s in [4, 5]')

    # Return Matched DataFrame
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
    loan_id_columns_s = ['activity_year_s','respondent_id_s','agency_code_s','sequence_number_s','HMDAIndex_s']
    loan_id_columns_s = [x for x in loan_id_columns_s if x in df.columns]
    loan_id_columns_p = ['activity_year_p','respondent_id_p','agency_code_p','sequence_number_p','HMDAIndex_p']
    loan_id_columns_p = [x for x in loan_id_columns_p if x in df.columns]

    # Keep Unique Loans
    df['count_index_s'] = df.groupby(loan_id_columns_s, dropna = False)['activity_year_s'].transform('count')
    df['count_index_p'] = df.groupby(loan_id_columns_p, dropna = False)['activity_year_p'].transform('count')

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
        df['i_LoanHasSecondarySale'] = df.groupby(loan_id_columns_s, dropna = False)['temp'].transform('max')
        df = df.query('count_index_s == 1 or (count_index_s == 2 & i_LoanHasSecondarySale == 1)')
        df = df.drop(columns = ['i_LoanHasSecondarySale','temp'])

    # Drop Index Counts
    df = df.drop(columns = ['count_index_s','count_index_p'])

    # Return DataFrame
    return df

# Load Data
def load_data(data_folder, min_year=2007, max_year=2017) :
    """
    Combine HMDA data after 2018, keeping originations and purchases only. For
    use primarily in matching after the first round.

    Parameters
    ----------
    data_folder : str
        Folder where HMDA data files are located.
    min_year : int, optional
        Minimum year of data to include (inclusive). The default is 2007.
    max_year : int, optional
        Maximum year of data to include (inclusive). The default is 2017.

    Returns
    -------
    df : pandas DataFrame
        Combined HMDA data.

    """
    
    # Combine Seller and Purchaser Data
    df = []
    for year in range(min_year, max_year+1) :
        file = glob.glob(f'{data_folder}/*{year}*.parquet')[0]
        df_a = pq.read_table(file, filters=[('action_taken','in',[1,6])]).to_pandas(date_as_object = False)
        df_a = df_a.query('action_taken == 6 | purchaser_type not in [1,2,3,4]')
        df.append(df_a)
        del df_a
    df = pd.concat(df)
    
    # Return DataFrame
    return df

#%% Match Functions
# Match HMDA Sellers and Purchasers
def match_hmda_sellers_purchasers_round1(data_folder, save_folder, min_year=2007, max_year=2017) :

    df = []
    for year in range(min_year, max_year+1) :

        # Display Progress
        print('Matching HMDA sellers and purchasers for year:', year)

        # Load Data
        df_a = load_data(data_folder, min_year=year, max_year=year)

        # Drop Columns
        drop_columns = ['denial_reason_1', 'denial_reason_2', 'denial_reason_3',
                        'edit_status','application_date_indicator','tract_one_to_four_family_units',
                        'derived_loan_product_type','derived_dwelling_category',
                        'derived_ethnicity','derived_race','derived_sex','lien_status',
                        'rate_spread', 'hoepa_status']
        df_a = df_a.drop(columns=drop_columns, errors='ignore')

        # Keep Originations/Purchases
        df_a['LoanAmountBin'] = 10*np.floor(df_a['loan_amount']/10)+5
        match_columns = ['loan_type', 'census_tract', 'occupancy_type', 'loan_purpose','LoanAmountBin']
        df_a = df_a.dropna(subset = match_columns)
        df_a = df_a.query('census_tract!=""')

        # Split into Sellers/Purchasers and Merge
        df_purchaser = df_a.query('action_taken == 6')
        df_a = df_a.query('action_taken == 1')

        # Expand LoanAmountBins for Sold Loans
        df_temp = df_a.loc[df_a.loan_amount%10 < 5]
        df_temp['LoanAmountBin'] = df_temp['LoanAmountBin']-10
        df_a = pd.concat([df_a, df_temp])
        del df_temp

        # Merge Sellers and Purchasers
        df_a = df_a.merge(df_purchaser, on=match_columns, suffixes=('_s','_p'))
        del df_purchaser

        # Income Matches
        df_a.loc[df_a['income_s'].isin([9999]), 'income_s'] = None
        df_a.loc[df_a['income_p'].isin([9999]), 'income_p'] = None
        df_a['IncomeDifference'] = df_a['income_s'] - df_a['income_p']
        df_a = df_a.query('abs(IncomeDifference) <= 5 | IncomeDifference.isnull()')

        # Make Sure Lenders are Different
        df_a = df_a.query('respondent_id_s!=respondent_id_p | agency_code_s!=agency_code_p')

        # Keep Unique Loans
        df_a = keep_uniques(df_a, one_to_one=True, verbose=True)

        # Keep Similar Loan Amounts and Incomes
        df_a['SizeDifference'] = df_a.loan_amount_s-df_a.loan_amount_p
        df_a = df_a.query('SizeDifference in [0,1]')
        df_a = df_a.query('abs(IncomeDifference) <= 0 | IncomeDifference.isnull()')

        # Drop Demographic Mismatches
        df_a = match_sex(df_a)
        df_a = match_race(df_a)
        df_a = match_ethnicity(df_a)

        # Drop Likely Balance Sheet Loans and Secondary Sales
        df_a = df_a.query('purchaser_type_s > 0')
        df_a = df_a.query('purchaser_type_p <= 4')
        
        # Property Type Match
        df_a = df_a.query('property_type_s==property_type_p')

        # Display Progress and Append
        print('Adding', len(df_a), 'matches from year', year)
        df.append(df_a)
        del df_a

    # Combine Matches
    df = pd.concat(df)
    
    # Sort Columns for Inspection
    df = df[df.columns.sort_values()]

    # Keep Subset of Columns as Crosswalk
    loan_id_columns_s = ['activity_year_s','respondent_id_s','agency_code_s','sequence_number_s','HMDAIndex_s']
    loan_id_columns_s = [x for x in loan_id_columns_s if x in df.columns]
    loan_id_columns_p = ['activity_year_p','respondent_id_p','agency_code_p','sequence_number_p','HMDAIndex_p']
    loan_id_columns_p = [x for x in loan_id_columns_p if x in df.columns]
    df = df[loan_id_columns_s + loan_id_columns_p]

    # Match Round
    df['MatchRound'] = 1

    # Save Crosswalk
    df = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(df, f'{save_folder}/pre2017_hmda_seller_purchaser_matches_round1.parquet')

# Match HMDA Sellers and Purchasers
def match_hmda_sellers_purchasers_round2(data_folder, save_folder, min_year = 2007, max_year = 2017) :

    # Load Previous Rounds Crosswalk
    cw = pd.read_parquet(f'{save_folder}/pre2017_hmda_seller_purchaser_matches_round1.parquet')

    df = []
    for year in range(min_year, max_year+1) :

        # Display Progress
        print('Matching HMDA sellers and purchasers for year:', year)

        # Load Data
        df_a = load_data(data_folder, min_year=year, max_year=year)

        # Drop Columns
        drop_columns = ['denial_reason_1', 'denial_reason_2', 'denial_reason_3',
                        'edit_status','application_date_indicator','tract_one_to_four_family_units',
                        'derived_loan_product_type','derived_dwelling_category',
                        'derived_ethnicity','derived_race','derived_sex','lien_status',
                        'rate_spread', 'hoepa_status']
        df_a = df_a.drop(columns = drop_columns, errors='ignore')

        # Drop Already Matched
        loan_id_columns = ['activity_year','respondent_id','agency_code','sequence_number','HMDAIndex']
        loan_id_columns = [x for x in loan_id_columns if x in df_a.columns]
        df_a = df_a.merge(cw[[x+'_s' for x in loan_id_columns]],
                         left_on = loan_id_columns,
                         right_on = [x+'_s' for x in loan_id_columns],
                         how = 'left',
                         indicator = True,
                         )
        df_a = df_a.query('_merge != "both"')
        df_a = df_a.drop(columns = ['_merge']+[x+'_s' for x in loan_id_columns])
        df_a = df_a.merge(cw[[x+'_p' for x in loan_id_columns]],
                         left_on = loan_id_columns,
                         right_on = [x+'_p' for x in loan_id_columns],
                         how = 'left',
                         indicator = True,
                         )
        df_a = df_a.query('_merge != "both"')
        df_a = df_a.drop(columns = ['_merge']+[x+'_p' for x in loan_id_columns])

        # Append Candidate Matches
        df.append(df_a)
        del df_a

    # Combine Matches
    df = pd.concat(df)

    # Drop Observations with Missing Match Columns
    match_columns = ['loan_type','loan_amount','census_tract','occupancy_type','property_type','loan_purpose']
    df = df.dropna(subset = match_columns)
    df = df.query('census_tract!=""')

    # Split into Sellers/Purchasers and Merge
    df_purchaser = df.query('action_taken == 6')
    df = df.query('action_taken == 1')

    # Merge
    df = df.merge(df_purchaser, on=match_columns, suffixes=('_s','_p'))
    del df_purchaser

    # Year Matches
    df = df.query('activity_year_s <= activity_year_p')
    df = df.query('activity_year_s+1 >= activity_year_p')

    # Income Matches
    df.loc[df['income_s'].isin([9999]), 'income_s'] = None
    df.loc[df['income_p'].isin([9999]), 'income_p'] = None
    df['IncomeDifference'] = df['income_s'] - df['income_p']
    df = df.query('abs(IncomeDifference) <= 5 | IncomeDifference.isnull()')
    
    # Make Sure Lenders are Different
    df = df.query('respondent_id_s!=respondent_id_p | agency_code_s!=agency_code_p')

    # Keep Unique Loans
    df = keep_uniques(df, one_to_one=True, verbose=True)
    
    # Drop Demographic Mismatches
    df = match_sex(df)
    df = match_race(df)
    df = match_ethnicity(df)
    
    # Tight Income Match
    df = df.query('abs(IncomeDifference) <= 1 | IncomeDifference.isnull()')
    
    # Year and Purchaser Type Matches
    df['i_YearMatch'] = 1*(df.activity_year_s==df.activity_year_p)
    df = df.query('i_YearMatch==1 | purchaser_type_s==0')
    df = df.query('i_YearMatch!=1 | purchaser_type_s!=0')

    # Drop Secondary Sales
    df = df.query('purchaser_type_p <= 4')

    # Keep ID Columns
    loan_id_columns_s = ['activity_year_s','respondent_id_s','agency_code_s','sequence_number_s','HMDAIndex_s']
    loan_id_columns_s = [x for x in loan_id_columns_s if x in df.columns]
    loan_id_columns_p = ['activity_year_p','respondent_id_p','agency_code_p','sequence_number_p','HMDAIndex_p']
    loan_id_columns_p = [x for x in loan_id_columns_p if x in df.columns]
    df = df[loan_id_columns_s + loan_id_columns_p]

    # Save Crosswalks
    df['MatchRound'] = 2
    df = pd.concat([cw, df])
    del cw
    df = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(df, f'{save_folder}/pre2017_hmda_seller_purchaser_matches_round2.parquet')

#%% Main Routine
if __name__ == '__main__' :

    # Set Folder Paths
    DATA_DIR = config.DATA_DIR
    DATA_FOLDER = DATA_DIR / 'clean'
    SAVE_FOLDER = DATA_DIR / 'match_data/match_sellers_purchasers_pre2018'

    # Unzip HMDA Data
    # match_hmda_sellers_purchasers_round1(DATA_FOLDER, SAVE_FOLDER, min_year=2007, max_year=2017)
    # match_hmda_sellers_purchasers_round2(DATA_FOLDER, SAVE_FOLDER, min_year=2007, max_year=2017)
