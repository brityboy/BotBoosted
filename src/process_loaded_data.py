from load_train_data import get_first_row_of_all_csv_files_in_a_list
from load_train_data import get_intersection_columns_for_different_csv_files
from load_train_data import extract_columns_from_multiple_csvs

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
    '''


if __name__ == "__main__":
    column_list = ['screen_name', 'id', 'geo_enabled', 'followers_count',
                   'friends_count', 'statuses_count', 'listed_count',
                   'favourites_count']
    compiled_df = extract_columns_from_multiple_csvs(column_list,
                                                     human_users +
                                                     fake_users)
