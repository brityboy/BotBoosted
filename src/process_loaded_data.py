from load_train_data import extract_columns_from_multiple_csvs
from collections import defaultdict
import pandas as pd
import csv

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


def extract_features_from_tweet_csv_files(csv_list):
    '''
    INPUT
         - csv_list: list of csv files
    OUTPUT
         - df

    returns a dataframe with the screen_name and id of the user, along
    with specified feature columns which are going to be done line by line so
    as not to be memory intensive (the mongodb will be handled this way
    as well). This approach will be taken and multiprocessing and threading
    will be used if possible in order to speed up the extraction from both
    the csv files and the mongo databases
    targets to extract right now are:
    a) has been included in another user's favorites (favorite_count)
    get total number of times a users tweet has been
    favorited (sum favorite_count)
    csv - favorite_count
    mongo - favorite_count
    b) has used a hashtag in the tweet
    csv - num_hashtags
    mongo - len(document['entities']['hashtags'])
    c) has logged into twitter via iphone (source)
    csv - source
    mongo - source
    d) has mentioned another user
    csv - num_mentions
    mongo - len(document['entities']['user_mentions'])
    '''
    result = defaultdict(defaultdict)
    for csv_file in csv_list:
        print csv_file
        with open(csv_file, 'r') as csvfile:
            next(csvfile)
            opencsvfile = csv.reader(x.replace('\0', '').replace('\n', '')
                                     for x in csvfile)
            for i, row in enumerate(opencsvfile):
                print csv_file, i
                if len(row) == 19:
                    if 'favorite_count' not in result[row[4]]:
                        result[row[4]]['favorite_count'] = 0
                    if 'num_hashtags' not in result[row[4]]:
                        result[row[4]]['num_hashtags'] = 0
                    if 'iphone_source' not in result[row[4]]:
                        result[row[4]]['iphone_source'] = 0
                    if 'num_mentions' not in result[row[4]]:
                        result[row[4]]['num_mentions'] = 0
                    if row[14] in ['0', 'NULL']:
                        result[row[4]]['favorite_count'] += 0
                    else:
                        result[row[4]]['favorite_count'] += int(row[14])
                    if row[15] in ['0', 'NULL']:
                        result[row[4]]['num_hashtags'] += 0
                    else:
                        result[row[4]]['num_hashtags'] += int(row[15])
                    if 'iPhone' in row[3]:
                        result[row[4]]['iphone_source'] += 1
                    else:
                        result[row[4]]['iphone_source'] += 0
                    if row[17] in ['0', 'NULL']:
                        result[row[4]]['num_mentions'] += 0
                    else:
                        result[row[4]]['num_mentions'] += int(row[17])
                elif len(row) == 25:
                    pass
    csvfile.close()
    return result

if __name__ == "__main__":
    # column_list = ['id', 'geo_enabled', 'followers_count',
    #                'friends_count', 'statuses_count', 'listed_count',
    #                'favourites_count', 'created_at']
    # df = extract_columns_from_multiple_csvs(column_list,
    #                                         human_users +
    #                                         fake_users)
    result = extract_features_from_tweet_csv_files([ds2_e13_tweets, ds2_int_tweets, ds2_tfp_tweets, ds2_twt_tweets, ds2_fsf_tweets])
