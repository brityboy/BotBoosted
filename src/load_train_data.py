import numpy as np
import pandas as pd
import csv
from collections import Counter


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
# ds2_tfp_followers = 'data/TFP.csv/followers.csv'
# ds2_tfp_friends = 'data/TFP.csv/friends.csv'
ds2_e13_tweets = 'data/E13.csv/tweets.csv'
ds2_e13_users = 'data/E13.csv/users.csv'
# ds2_e13_followers = 'data/E13.csv/followers.csv'
# ds2_e13_friends = 'data/E13.csv/friends.csv'
ds2_fsf_tweets = 'data/FSF.csv/tweets.csv'
ds2_fsf_users = 'data/FSF.csv/users.csv'
# ds2_fsf_followers = 'data/FSF.csv/followers.csv'
# ds2_fsf_friends = 'data/FSF.csv/friends.csv'
ds2_int_tweets = 'data/INT.csv/tweets.csv'
ds2_int_users = 'data/INT.csv/users.csv'
# ds2_int_followers = 'data/INT.csv/followers.csv'
# ds2_int_friends = 'data/INT.csv/friends.csv'
ds2_twt_tweets = 'data/TWT.csv/tweets.csv'
ds2_twt_users = 'data/TWT.csv/users.csv'
# ds2_twt_followers = 'data/TWT.csv/followers.csv'
# ds2_twt_friends = 'data/TWT.csv/friends.csv'
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


def load_data_into_dataframe(filename):
    '''
    INPUT
         - filename: name of file
    OUTPUT
         - pandas DataFrame

    returns contents of csv file into a dataframe
    '''
    df = pd.read_csv(filename)
    return df


def open_text_file(filename):
    '''
    INPUT
         - filename: name of file
    OUTPUT
         - np.array

    returns contents of filename as objects in an array
    '''
    text_list = []
    with open(filename) as f:
        for line in f:
            text_list.append(line.replace('\n', ''))
    return np.array(text_list)


def open_csv_file(filename):
    '''
    INPUT
         - filename: name of file
    OUTPUT
         - list

    returns csv file rows into a list
    '''
    text_list = []
    with open(filename, 'r') as csvfile:
        # opencsvfile = csv.reader(codecs.open(filename, 'rU', 'utf-16'))
        opencsvfile = csv.reader(x.replace('\0', '').replace('\n', '')
                                 for x in csvfile)
        for row in opencsvfile:
            text_list.append(row)
    return text_list


def open_csv_file_as_dataframe(filename):
    '''
    INPUT
         - filename: name of file
    OUTPUT
         - pandas dataframe

    returns contents of csv file, null bytes and other items removed
    in a dataframe, with column headers
    '''
    text_list = []
    with open(filename, 'r') as csvfile:
        opencsvfile = csv.reader(x.replace('\0', '').replace('\n', '')
                                 for x in csvfile)
        for row in opencsvfile:
            text_list.append(row)
    columns = text_list[0]
    df = pd.DataFrame(text_list[1:], columns=columns)
    return df


def give_basic_data_information(filename):
    '''
    INPUT
         - filename: this is the file
    OUTPUT
         - prints the head, info and shape of a DataFrame

    returns nothing
    '''
    df = open_csv_file_as_dataframe(filename)
    print(df.head())
    print(df.info())
    print(df.shape)


def get_first_row_of_all_csv_files_in_a_list(file_list):
    '''
    INPUT
         - file_list: list of csv files
    OUTPUT
         - dictionary which has keys as columns and values as number of files
         these columns occur in
    '''
    output_list = []
    for file_name in file_list:
        with open(file_name, 'r') as f:
            first_line = f.readline()
            first_line = first_line.replace('"', ''). \
                replace('\n', '').replace('\r', '').split(',')

            output_list += first_line
    return Counter(output_list)


def extract_columns_from_multiple_csvs(column_list, csv_list):
    '''
    INPUT
         - column_list: list of columns to extract from the different csvs
         - csv_list: list of the different csvs to get the data from
    OUTPUT
         - compiled_df: a dataframe that has all the columns from the csvs
    '''
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


def get_intersection_columns_for_different_csv_files(checkdata):
    '''
    INPUT
        - checkdata: a dictionary that has the keys as columns and the values
        as the number of csv files they occur in
    OUTPUT
         - column_list: a list of columns that has the columns which occur
         in all csv files
    '''
    column_list = []
    maxval = max(checkdata.values())
    for k, v in checkdata.iteritems():
        if v == maxval:
            column_list.append(k)
    return column_list


if __name__ == "__main__":
    column_list = ['user_id', 'favorite_count', 'num_hashtags', 'text',
                   'source', 'num_mentions', 'timestamp', 'geo', 'place',
                   'retweet_count', 'reply_count']
    df = extract_columns_from_multiple_csvs(column_list,
                                            human_tweets+fake_tweets)
    df.to_csv('data/training_tweets.csv')
