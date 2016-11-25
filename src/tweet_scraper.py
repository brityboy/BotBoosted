import tweepy
import sys
import jsonpickle
import os
import numpy as np
from unidecode import unidecode
import time
from datetime import datetime
import json
import dill as pickle
import tweet_scrape_processor as tsp


def access_credentials():
    '''
    INPUT
         - none
    OUTPUT
         - API_KEY: string
         - API_SECRET: string
         - ACCESS_TOKEN: string
         - ACCESS_TOKEN_SECRET: string

    returns the access keys necessary to use the API
    '''
    with open('data/credentials.json') as f:
        data = json.load(f)
    API_KEY = data['API_KEY']
    API_SECRET = data['API_SECRET']
    ACCESS_TOKEN = data['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = data['ACCESS_TOKEN_SECRET']
    return API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET


def download_tweets_to_list(searchQuery, tweetsPerQry,
                            maxTweets, api, verbose=False):
    '''
    this function downloads tweet info into two lists
    INPUT
        searchQuery = this is the search query, it should follow
        the twitter api query structure
        tweetsPerQry = this is how many tweets you download
        per query to the website
        maxTweets = this is the maximum number of tweets you want to download
        api = this is your api object, you create this with:
            API_KEY, API_SECRET,
            ACCESS_TOKEN, ACCESS_TOKEN_SECRET = access_credentials()
            auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
            api = tweepy.API(auth, wait_on_rate_limit=True,
                             wait_on_rate_limit_notify=True)
    OUTPUT
         - tweet_list - the list of the actual tweets
         - tweet_array_list - the processed arrays for prediction purposes
    '''
    tweet_list = []
    # tweet_array_list = []
    sinceId = None
    max_id = -1L
    tweetCount = 0
    if verbose:
        print("Downloading max {0} tweets".format(maxTweets))
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if not sinceId:
                    new_tweets = api.search(q=searchQuery,
                                            count=tweetsPerQry)
                else:
                    new_tweets = api.search(q=searchQuery,
                                            count=tweetsPerQry,
                                            since_id=sinceId)
            else:
                if not sinceId:
                    new_tweets = api.search(q=searchQuery,
                                            count=tweetsPerQry,
                                            max_id=str(max_id - 1))
                else:
                    new_tweets = api.search(q=searchQuery,
                                            count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            since_id=sinceId)
            if not new_tweets:
                if verbose:
                    print("No more tweets found")
                break
            for tweet in new_tweets:
                tweet_list.append(tweet._json)
            tweetCount += len(new_tweets)
            if verbose:
                print("Downloaded {0} tweets".format(tweetCount))
            src = '/search/tweets'
            rem = 'remaining'
            if verbose:
                print api.rate_limit_status()['resources']['search'][src][rem]
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            if verbose:
                # print("some error : " + str(e))
                print e
                print type(e)
                print e.__dict__
                print e.reason
                print type(e.reason)
            return tweet_list
    print("Downloaded {0} tweets".format(tweetCount))
    return tweet_list


def download_tweets_given_search_query(searchQuery, verbose=False):
    '''
    INPUT
         - a searchQuery
    OUTPUT
         - tweet_list - list of json tweet objects
    Returns a tweet_list - which is a list of tweets about a topic
    '''
    API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET = \
        access_credentials()
    auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)
    maxTweets = 15000  # Some arbitrary large number
    tweetsPerQry = 100
    tweet_list = download_tweets_to_list(searchQuery,
                                         tweetsPerQry,
                                         maxTweets,
                                         api, verbose=verbose)
    return tweet_list

if __name__ == "__main__":
    tweet_list = download_tweets_given_search_query('make america great again',
                                                    verbose=True)
