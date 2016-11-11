import numpy as np
import pandas as pd
import csv
# import codecs
from collections import defaultdict


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
ds2_tfp_followers = 'data/TFP.csv/followers.csv'
ds2_tfp_friends = 'data/TFP.csv/friends.csv'
ds2_e13_tweets = 'data/E13.csv/tweets.csv'
ds2_e13_users = 'data/E13.csv/users.csv'
ds2_e13_followers = 'data/E13.csv/followers.csv'
ds2_e13_friends = 'data/E13.csv/friends.csv'
ds2_fsf_tweets = 'data/FSF.csv/tweets.csv'
ds2_fsf_users = 'data/FSF.csv/users.csv'
ds2_fsf_followers = 'data/FSF.csv/followers.csv'
ds2_fsf_friends = 'data/FSF.csv/friends.csv'
ds2_int_tweets = 'data/INT.csv/tweets.csv'
ds2_int_users = 'data/INT.csv/users.csv'
ds2_int_followers = 'data/INT.csv/followers.csv'
ds2_int_friends = 'data/INT.csv/friends.csv'
ds2_twt_tweets = 'data/TWT.csv/tweets.csv'
ds2_twt_users = 'data/TWT.csv/users.csv'
ds2_twt_followers = 'data/TWT.csv/followers.csv'
ds2_twt_friends = 'data/TWT.csv/friends.csv'
human_tweets = [ds1_genuine_tweets, ds2_e13_tweets, ds2_tfp_tweets]
fake_tweets = [ds1_sb1_tweets, ds1_sb2_tweets, ds1_sb3_tweets,
               ds1_ts1_tweets, ds2_fsf_tweets, ds2_int_tweets,
               ds2_twt_tweets]
human_users = [ds1_genuine_users, ds2_e13_users, ds2_tfp_users]
fake_users = [ds1_sb1_users, ds1_sb2_users, ds1_sb3_users, ds1_ts1_users,
              ds2_fsf_users, ds2_int_users, ds2_twt_users]


def load_data_into_dataframe(filename):
    df = pd.read_csv(path)
    return df


def open_text_file(filename):
    text_list = []
    with open(filename) as f:
        for line in f:
            text_list.append(line.replace('\n', ''))
    return np.array(text_list)


def open_csv_file(filename):
    text_list = []
    with open(filename, 'r') as csvfile:
        # opencsvfile = csv.reader(codecs.open(filename, 'rU', 'utf-16'))
        opencsvfile = csv.reader(x.replace('\0', '').replace('\n', '') for x in csvfile)
        for row in opencsvfile:
            text_list.append(row)
    return text_list


def open_csv_file_as_dataframe(filename):
    text_list = []
    with open(filename, 'r') as csvfile:
        opencsvfile = csv.reader(x.replace('\0', '').replace('\n', '') for x in csvfile)
        for row in opencsvfile:
            text_list.append(row)
    columns = text_list[0]
    df = pd.DataFrame(text_list[1:], columns=columns)
    return df


def brute_open_text_file_as_df(filename):
    text_list = []
    with open(filename, 'r') as f:
        for line in f:
            if len(line.split(',')) == 25:
                text_list.append(line.replace('\n', '').split(','))
    columns = text_list[0]
    df = pd.DataFrame(text_list[1:], columns=columns)
    return df


def check_file_integrity(filelist):
    info_list = []
    for content in (filelist):
        df = open_csv_file_as_dataframe(content)
        info_list.append((content, df.columns))
    check = np.array([list(item[1]) for item in info_list])
    allcolumns = set([item for row in check for item in row])
    checkdict = defaultdict(list)
    for column in allcolumns:
        for filename in check:
            if column in filename:
                checkdict[column].append(1)
            else:
                checkdict[column].append(0)
    checkdata = pd.DataFrame.from_dict(checkdict, orient='index')
    checkdata.columns = [item[0] for item in info_list]
    return checkdata

def give_basic_data_information(filename):
    df = open_csv_file_as_dataframe(filename)
    print df.head()
    print df.info()
    print df.shape


if __name__ == "__main__":
    checkdata = check_file_integrity(human_tweets+fake_tweets)
    # give_basic_data_information(ds2_fsf_tweets)
