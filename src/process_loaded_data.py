from load_train_data import extract_columns_from_multiple_csvs
from collections import defaultdict
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


def extract_features_from_tweet_csv_files(csv_list):
    '''
    INPUT
         - csv_list: list of csv files
    OUTPUT
         - dictionary

    returns a nested dictionary where the key is the username, the values are
    the features, and then the values of the feature keys are the values
    This approach will be taken and multiprocessing and threading
    will be used if possible in order to speed up the extraction from both
    the csv files and the mongo databases
    targets to extract right now are:
    a) has been included in another user's favorites (favorite_count)
    get total number of times a users tweet has been
    favorited (sum favorite_count)
    csv - favorite_count
    b) has used a hashtag in the tweet
    csv - num_hashtags
    c) has logged into twitter via iphone (source)
    csv - source
    d) has mentioned another user
    csv - num_mentions
    '''
    result = defaultdict(defaultdict)
    for csv_file in csv_list:
        print(csv_file)
        with open(csv_file, 'r') as csvfile:
            next(csvfile)
            opencsvfile = csv.reader(x.replace('\0', '').replace('\n', '')
                                     for x in csvfile)
            for i, row in enumerate(opencsvfile):
                print(csv_file, i)
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
                    if 'favorite_count' not in result[row[4]]:
                        result[row[3]]['favorite_count'] = 0
                    if 'num_hashtags' not in result[row[4]]:
                        result[row[3]]['num_hashtags'] = 0
                    if 'iphone_source' not in result[row[4]]:
                        result[row[3]]['iphone_source'] = 0
                    if 'num_mentions' not in result[row[4]]:
                        result[row[3]]['num_mentions'] = 0
                    if row[14] in ['0', 'NULL']:
                        result[row[3]]['favorite_count'] += 0
                    else:
                        result[row[3]]['favorite_count'] += int(row[14])
                    if row[18] in ['0', 'NULL']:
                        result[row[3]]['num_hashtags'] += 0
                    else:
                        result[row[3]]['num_hashtags'] += int(row[18])
                    if 'iPhone' in row[2]:
                        result[row[3]]['iphone_source'] += 1
                    else:
                        result[row[3]]['iphone_source'] += 0
                    if row[20] in ['0', 'NULL']:
                        result[row[3]]['num_mentions'] += 0
                    else:
                        result[row[3]]['num_mentions'] += int(row[20])
    csvfile.close()
    return result


def combine_user_info_with_feature_dict(df, feature_dict):
    '''
    INPUT
         - df: a pandas dataframe that has the information about the users
         in the 'id' column (this is a mandatory column)
         - feature_dict: a dictionary where the keys correspond to the ids,
         and the values correspond to attributes and their values
    OUTPUT
         - df
    Returns an expanded dataframe that has as new columns coming from the
    feature dict
    '''
    df['twt_favorite_count'] = \
        df.id.apply(lambda user_id: feature_dict[user_id]['favorite_count'])
    df['num_hashtags'] = \
        df.id.apply(lambda user_id: feature_dict[user_id]['num_hashtags'])
    df['iphone_source'] = \
        df.id.apply(lambda user_id: feature_dict[user_id]['iphone_source'])
    df['num_mentions'] = \
        df.id.apply(lambda user_id: feature_dict[user_id]['num_mentions'])
    return df


def check_if_many_relative_followers_to_friends(df):
    '''
    INPUT
         - df: a pandas dataframe joined already with the feature dict
    OUTPUT
         - df: a pandas dataframe that has the

    return df which has a new feature: 2*followers_friends
    '''
    boolean_list = []
    for followers, friends in zip(df.followers_count.values,
                                  df.friends_count.values):
        if 2*int(followers) >= int(friends):
            boolean_list.append(1)
        else:
            boolean_list.append(0)
    df['followers_friends'] = boolean_list
    return df


def process_df(df):
    '''
    INPUT
         - df: a pandas dataframe joined already with the feature dict
    OUTPUT
         - df: a pandas dataframe

    returns the pandas dataframe with processed features
    '''
    df['has_30_followers'] = \
        df.followers_count.apply(lambda followers:
                                 1 if int(followers) > 30 else 0)
    df['geo_localized'] = \
        df.geo_enabled.apply(lambda geo: 1 if geo == '1' else 0)
    df['favorited_by_another'] = \
        df.twt_favorite_count.apply(lambda favcnt: 1 if favcnt > 0 else 0)
    df['has_hashtagged'] = \
        df.num_hashtags.apply(lambda hashtag: 1 if hashtag > 0 else 0)
    df['used_iphone'] = \
        df.iphone_source.apply(lambda iphone: 1 if iphone > 0 else 0)
    df['has_mentions'] = \
        df.num_mentions.apply(lambda mentions: 1 if mentions > 0 else 0)
    df = check_if_many_relative_followers_to_friends(df)
    df = drop_unnecessary_columns(df)
    return df


def drop_unnecessary_columns(df):
    '''
    INPUT
         - df: a featurized pandas dataframe
    OUTPUT
         - df: a pandas dataframe

    returns the pandas dataframe with the unnecessary features dropped
    '''
    df.drop(['geo_enabled', 'followers_count', 'friends_count',
             'statuses_count', 'listed_count', 'favourites_count',
             'created_at', 'file', 'twt_favorite_count', 'num_hashtags',
             'iphone_source', 'num_mentions'], axis=1, inplace=True)
    return df


def create_processed_dataframe():
    '''
    INPUT
         - none
    OUTPUT
         - df: pandas df

    returns a processed pandas dataframe with the features for modelling
    '''
    column_list = ['id', 'geo_enabled', 'followers_count',
                   'friends_count', 'statuses_count', 'listed_count',
                   'favourites_count', 'created_at']
    df = extract_columns_from_multiple_csvs(column_list,
                                            human_users +
                                            fake_users)
    feature_dict = \
        extract_features_from_tweet_csv_files(human_tweets+fake_tweets)
    # users_with_userdata = set(df.id)
    users_who_tweeted = set(feature_dict.keys())
    # users_without_tweets = users_with_userdata - users_who_tweeted
    dfusers_who_tweeted = df[df.id.isin(users_who_tweeted)]
    #  dfusers_no_tweets = df[df.id.isin(users_without_tweets)]
    df = combine_user_info_with_feature_dict(dfusers_who_tweeted, feature_dict)
    df = process_df(df)
    return df


if __name__ == "__main__":
    df = create_processed_dataframe()
