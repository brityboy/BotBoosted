import numpy as np
import pandas as pd
import csv
from collections import Counter

"""
This module is used to host the different functions that load the training
data into a single manageable dataframe.

These are a series of helper functions that extract the raw information from
the training data csv files that were put together by Stefano Cresci,
Roberto Di Pietro, Marinella Petrocchi, Angelo Spognardi, and Maurizio
Tesconi for their paper "Fame for sale: efficient detection of fake Twitter
followers." http://mib.projects.iit.cnr.it/dataset.html is the link to their
data. The main orientation of this script is to compile the different csv
files that this research team put together into one single and properly
labeled csv.

As the dataset I used for this project may not be shared, this module
only demonstrates the the kind of features and transformations I did with
this dataset so that should there be anyone who wishes to pursue further
research in this area, a method already exists for collecting the data
into a single pandas dataframe

Example:
    file_list = human_users+fake_users
    checkdata = get_first_row_of_all_csv_files_in_a_list(file_list)
    column_list = get_intersection_columns_for_different_csv_files(checkdata)
    df = extract_columns_from_multiple_csvs(column_list,
                                            file_list)
    df.to_csv('data/training_users.csv')

    file_list = human_tweets+fake_tweets
    checkdata = get_first_row_of_all_csv_files_in_a_list(file_list)
    column_list = get_intersection_columns_for_different_csv_files(checkdata)
    df = extract_columns_from_multiple_csvs(column_list,
                                            file_list)
    df.to_csv('data/training_tweets.csv')
"""


ds1_genuine_tweets = 'data/datasets_full.csv/genuine_accounts.csv/tweets.csv'
ds1_genuine_users = 'data/datasets_full.csv/genuine_accounts.csv/users.csv'
ds1_sb1_tweets = 'data/datasets_full.csv/social_spambots_1.csv/tweets.csv'
ds1_sb1_users = 'data/datasets_full.csv/social_spambots_1.csv/users.csv'
ds1_sb2_tweets = 'data/datasets_full.csv/social_spambots_2.csv/tweets.csv'
ds1_sb2_users = 'data/datasets_full.csv/social_spambots_2.csv/users.csv'
ds1_sb3_tweets = 'data/datasets_full.csv/social_spambots_3.csv/tweets.csv'
ds1_sb3_users = 'data/datasets_full.csv/social_spambots_3.csv/users.csv'
ds1_ts1_tweets = 'data/datasets_full.csv/traditional_spambots_1.csv/tweets.csv'
ds1_ts1_users = 'data/datasets_full.csv/traditional_spambots_1.csv/users.csv'
ds1_ts2_tweets = 'data/datasets_full.csv/traditional_spambots_2.csv/tweets.csv'
ds1_ts2_users = 'data/datasets_full.csv/traditional_spambots_2.csv/users.csv'
ds1_ts3_tweets = 'data/datasets_full.csv/traditional_spambots_3.csv/tweets.csv'
ds1_ts3_users = 'data/datasets_full.csv/traditional_spambots_3.csv/users.csv'
ds1_ts4_tweets = 'data/datasets_full.csv/traditional_spambots_4.csv/tweets.csv'
ds1_ts4_users = 'data/datasets_full.csv/traditional_spambots_4.csv/users.csv'
ds1_ff_tweets = 'data/datasets_full.csv/fake_followers.csv/tweets.csv'
ds1_ff_users = 'data/datasets_full.csv/fake_followers.csv/users.csv'
ds2_tfp_tweets = 'data/TFP.csv/tweets.csv'
ds2_tfp_users = 'data/TFP.csv/users.csv'
ds2_e13_tweets = 'data/E13.csv/tweets.csv'
ds2_e13_users = 'data/E13.csv/users.csv'
ds2_fsf_tweets = 'data/FSF.csv/tweets.csv'
ds2_fsf_users = 'data/FSF.csv/users.csv'
ds2_int_tweets = 'data/INT.csv/tweets.csv'
ds2_int_users = 'data/INT.csv/users.csv'
ds2_twt_tweets = 'data/TWT.csv/tweets.csv'
ds2_twt_users = 'data/TWT.csv/users.csv'

human_tweets = [ds1_genuine_tweets, ds2_e13_tweets, ds2_tfp_tweets]
fake_tweets = [ds1_sb1_tweets, ds1_sb2_tweets, ds1_sb3_tweets,
               ds1_ts1_tweets, ds2_fsf_tweets, ds2_int_tweets,
               ds2_twt_tweets]
human_users = [ds1_genuine_users, ds2_e13_users, ds2_tfp_users]
fake_users = [ds1_sb1_users, ds1_sb2_users, ds1_sb3_users, ds1_ts1_users,
              ds2_fsf_users, ds2_int_users, ds2_twt_users]
filename_dict = {ds1_genuine_tweets: 'hum_tw', ds2_e13_tweets: "e13_tw",
                 ds2_tfp_tweets: "tfp_tw", ds1_sb1_tweets: "sb1_tw",
                 ds1_sb2_tweets: "sb2_tw", ds1_sb3_tweets: "sb3_tw",
                 ds1_ts1_tweets: "ts1_tw", ds2_fsf_tweets: "fsf_tw",
                 ds2_int_tweets: "int_tw", ds2_twt_tweets: "twt_tw",
                 ds1_genuine_users: "hum1_us", ds2_e13_users: "e13_us",
                 ds2_tfp_users: "tfp_us", ds1_sb1_users: "sb1_us",
                 ds1_sb2_users: "sb2_us", ds1_sb3_users: "sb3_us",
                 ds1_ts1_users: "ts1_us", ds2_fsf_users: "fsf_us",
                 ds2_int_users: "int_us", ds2_twt_users: "twt_us"}
label_dict = {ds1_genuine_tweets: 0, ds2_e13_tweets: 0,
              ds2_tfp_tweets: 0, ds1_sb1_tweets: 1,
              ds1_sb2_tweets: 1, ds1_sb3_tweets: 1,
              ds1_ts1_tweets: 1, ds2_fsf_tweets: 1,
              ds2_int_tweets: 1, ds2_twt_tweets: 1,
              ds1_genuine_users: 0, ds2_e13_users: 0,
              ds2_tfp_users: 0, ds1_sb1_users: 1,
              ds1_sb2_users: 1, ds1_sb3_users: 1,
              ds1_ts1_users: 1, ds2_fsf_users: 1,
              ds2_int_users: 1, ds2_twt_users: 1}


def open_csv_file_as_dataframe(filename):
    """
    Args:
        filename (str): this is the name of the file that will be opened
        as a dataframe
    Returns
        df (pandas DataFrame): contents of csv file, null bytes and other
        items removed in a dataframe, with column headers which is in the
        first row of each file
    """
    text_list = []
    with open(filename, 'r') as csvfile:
        opencsvfile = csv.reader(x.replace('\0', '').replace('\n', '')
                                 for x in csvfile)
        for row in opencsvfile:
            text_list.append(row)
    columns = text_list[0]
    df = pd.DataFrame(text_list[1:], columns=columns)
    return df


def get_first_row_of_all_csv_files_in_a_list(file_list):
    """
    Args:
        file_list (list): list of csv files that will be processed
    Returns:
        output_dict (dictionary): where the keys are the columns of the
        different text files and the values are the number of files
        these columns occur in
    """
    output_list = []
    for file_name in file_list:
        with open(file_name, 'r') as f:
            first_line = f.readline()
            first_line = first_line.replace('"', ''). \
                replace('\n', '').replace('\r', '').split(',')

            output_list += first_line
    output_dict = Counter(output_list)
    return output_dict


def extract_columns_from_multiple_csvs(column_list, csv_list):
    """
    Args:
        column_list (list): list of columns to extract from the different csvs
        csv_list (list): list of the different csvs to get the data from
    Returns
        compiled_df (pandas DataFrame): a dataframe that has all the
        columns from the different csvs
    """
    compiled_df = pd.DataFrame(columns=np.append(column_list,
                                                 ['file', 'label']))
    for csv_file in csv_list:
        print(csv_file)
        df = open_csv_file_as_dataframe(csv_file)
        df.columns = [c.replace('\n', '').replace('\r',
                                                  '') for c in df.columns]
        df = df[column_list]
        df['file'] = filename_dict[csv_file]
        df['label'] = label_dict[csv_file]
        compiled_df = pd.concat([compiled_df, df])
    return compiled_df


def get_intersection_columns_for_different_csv_files(output_dict):
    """
    Args:
        output_dict (dictionary): a dictionary that has the keys as
        columns and the values as the number of csv files they occur in
    Returns:
        column_list (list): a list of columns that has the columns which occur
        in all csv files that will be loaded into a dataframe
    """
    column_list = []
    maxval = max(output_dict.values())
    for k, v in output_dict.iteritems():
        if v == maxval:
            column_list.append(k)
    return column_list


if __name__ == "__main__":
    file_list = human_users+fake_users
    checkdata = get_first_row_of_all_csv_files_in_a_list(file_list)
    column_list = get_intersection_columns_for_different_csv_files(checkdata)
    df = extract_columns_from_multiple_csvs(column_list,
                                            file_list)
    df.to_csv('data/training_users.csv')
