import dill as pickle
# from lightweight_classifier import convert_created_time_to_datetime
import pandas as pd
import numpy as np
from tweet_text_processor import split_list
import multiprocessing as mp
import time
import pandas as pd
from datetime import datetime


def process_tweet(tweet):
    '''
    INPUT
         - tweet: json object from twitter's api
    OUTPUT
         - user_vector = np.array for predictions
         - text_vector = np.array with tweet_information

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


def convert_created_time_to_datetime(datestring):
    '''
    INPUT
         - datestring: a string object either as a date or
         a unix timestamp
    OUTPUT
         - a datetime object

    Returns a pandas datetime object
    '''
    if len(datestring) == 30:
        return pd.to_datetime(datestring)
    else:
        return pd.to_datetime(datetime.fromtimestamp(int(datestring[:10])))


def multiprocess_process_tweet(tweet_list):
    '''
    INPUT
         - tweet_list: this is a list of the documents to be tweet tokenized
    OUTPUT
         - list

    Return a list of processed tweets done with multiprocessing
    '''
    n_processes = mp.cpu_count()
    p = mp.Pool(n_processes)
    split_docs = np.array(split_list(tweet_list, n_processes))
    processed_tweets = p.map(process_tweet, split_docs)
    return [item for row in processed_tweets for item in row]

if __name__ == "__main__":
    # df = pd.read_csv('data/training_user_tweet_data.csv')
    start = time.time()
    with open('data/test_tweet_scrape.pkl', 'r+') as f:
        tweet_list = pickle.load(f)
    print("load pkl file: ", time.time() - start)
    # tweet = tweet_list[0]
    # tweetarray = process_tweet(tweet)
    # print(tweetarray)
    start = time.time()
    # tweets_for_prediction = multiprocess_process_tweet(tweet_list)
    processed_tweets = np.array([process_tweet(tweet) for tweet in tweet_list])
    print("process tweets tweets: ", time.time() - start)
