import dill as pickle
from lightweight_classifier import *
import pandas as pd
import numpy as np


def process_tweet(tweet):
    '''
    INPUT
         - tweet: json object from twitter's api
    OUTPUT
         - np.array for predictions

    Returns a json from twitter's api into the necessary row format needed
    for predictions
    '''
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
    return np.array([profile_use_background_image, geo_enabled, verified,
                     followers_count, default_profile_image, listed_count,
                     statuses_count, friends_count, favourites_count,
                     favorite_count, num_hashtags, num_mentions, retweet_count,
                     account_age, followers_friends, has_30_followers,
                     favorited_by_another, has_hashtagged, has_mentions])


if __name__ == "__main__":
    df = pd.read_csv('data/training_user_tweet_data.csv')
    with open('data/test_tweet_scrape.pkl', 'r') as f:
        tweet_list = pickle.load(f)
    tweet = tweet_list[0]
    tweetarray = process_tweet(tweet)
    print(tweetarray)
    tweets_for_prediction = [process_tweet(tweet) for tweet in tweet_list]
