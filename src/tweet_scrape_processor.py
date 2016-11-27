import dill as pickle
import pandas as pd
import numpy as np
import time
from datetime import datetime


"""
This module is responsible for processing each json object into the different
features necessary for the prediction model module so that the random forest
ensemble can make predictions on newly downloaded tweets, or tweets store in
a mongo database, so long as they are in the form of json objects in a list
"""


def process_tweet(tweet):
    """
    Args:
        tweet (json): single tweet object downloaded from twitter's api
    Returns:
        vectorized_tweet (2d numpy array): vectorized tweet in the format
        needed for predictions and for further processing, as is needed
        by the first and earlier version of making predictions
    """
    profile_use_background_image = \
        tweet['user']['profile_use_background_image']
    geo_enabled = tweet['user']['geo_enabled']
    verified = tweet['user']['verified']
    followers_count = tweet['user']['followers_count']
    default_profile_image = tweet['user']['default_profile_image']
    listed_count = tweet['user']['listed_count']
    statuses_count = tweet['user']['statuses_count']
    friends_count = tweet['user']['friends_count']
    favourites_count = tweet['user']['favourites_count']
    favorite_count = tweet['favorite_count']
    num_hashtags = len(tweet['entities']['hashtags'])
    num_mentions = len(tweet['entities']['user_mentions'])
    retweet_count = tweet['retweet_count']
    tweet_date = convert_created_time_to_datetime(tweet['created_at'])
    account_creation_date = \
        convert_created_time_to_datetime(tweet['user']['created_at'])
    time_difference = tweet_date - account_creation_date
    account_age = time_difference.days+1
    followers_friends = 1 if 2*followers_count > friends_count else 0
    has_30_followers = 1 if followers_count >= 30 else 0
    favorited_by_another = 1 if favourites_count > 0 else 0
    has_hashtagged = 1 if num_hashtags > 0 else 0
    has_mentions = 1 if num_mentions > 0 else 0
    user_id = tweet['user']['id']
    text = tweet['text']
    screen_name = tweet['user']['screen_name']
    user_vector = np.array([profile_use_background_image, geo_enabled,
                            verified, followers_count, default_profile_image,
                            listed_count, statuses_count, friends_count,
                            favourites_count, favorite_count, num_hashtags,
                            num_mentions, retweet_count, account_age,
                            followers_friends, has_30_followers,
                            favorited_by_another, has_hashtagged,
                            has_mentions])
    text_vector = np.array([user_id, text, screen_name])
    return np.hstack((user_vector, text_vector))


def process_tweet_v2(tweet):
    """
    Args:
        tweet (json): single tweet object downloaded from twitter's api
    Returns:
        vectorized_tweet (2d numpy array): the necessary row format needed
            for predictions that takes into account the additional features
            that look at behavior versus network information (i.e. tweets per
            follower, likes per friend, etc)
    """
    profile_use_background_image = \
        tweet['user']['profile_use_background_image']
    geo_enabled = tweet['user']['geo_enabled']
    verified = tweet['user']['verified']
    followers_count = tweet['user']['followers_count']
    default_profile_image = tweet['user']['default_profile_image']
    listed_count = tweet['user']['listed_count']
    statuses_count = tweet['user']['statuses_count']
    friends_count = tweet['user']['friends_count']
    favourites_count = tweet['user']['favourites_count']
    favorite_count = tweet['favorite_count']
    num_hashtags = len(tweet['entities']['hashtags'])
    num_mentions = len(tweet['entities']['user_mentions'])
    retweet_count = tweet['retweet_count']
    tweet_date = convert_created_time_to_datetime(tweet['created_at'])
    account_creation_date = \
        convert_created_time_to_datetime(tweet['user']['created_at'])
    time_difference = tweet_date - account_creation_date
    account_age = time_difference.days+1
    followers_friends = 1 if 2*followers_count > friends_count else 0
    has_30_followers = 1 if followers_count >= 30 else 0
    favorited_by_another = 1 if favourites_count > 0 else 0
    has_hashtagged = 1 if num_hashtags > 0 else 0
    has_mentions = 1 if num_mentions > 0 else 0
    user_id = tweet['user']['id']
    text = tweet['text']
    screen_name = tweet['user']['screen_name']
    tweets_followers \
        = -999 if followers_count == 0 else statuses_count / followers_count
    tweets_friends \
        = -999 if friends_count == 0 else statuses_count / friends_count
    likes_followers \
        = -999 if followers_count == 0 else favourites_count / followers_count
    likes_friends \
        = -999 if friends_count == 0 else favourites_count / friends_count
    user_vector = np.array([profile_use_background_image, geo_enabled,
                            verified, followers_count, default_profile_image,
                            listed_count, statuses_count, friends_count,
                            favourites_count, favorite_count, num_hashtags,
                            num_mentions, retweet_count, account_age,
                            followers_friends, has_30_followers,
                            favorited_by_another, has_hashtagged,
                            has_mentions, tweets_followers, tweets_friends,
                            likes_followers, likes_friends])
    text_vector = np.array([user_id, text, screen_name])
    return np.hstack((user_vector, text_vector))


def convert_created_time_to_datetime(datestring):
    """
    Args:
        datestring (str): a string object either as a date or
         a unix timestamp
    Returns:
        a pandas datetime object
    """
    if len(datestring) == 30:
        return pd.to_datetime(datestring)
    else:
        return pd.to_datetime(datetime.fromtimestamp(int(datestring[:10])))


if __name__ == "__main__":
    start = time.time()
    with open('data/test_tweet_scrape.pkl', 'r+') as f:
        tweet_list = pickle.load(f)
    print("load pkl file: ", time.time() - start)
    start = time.time()
    processed_tweets = np.array([process_tweet_v2(tweet)
                                 for tweet in tweet_list])
    print("process tweets tweets: ", time.time() - start)
