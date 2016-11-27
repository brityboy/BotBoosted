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


"""
This module downloads the tweets of a given query into a list. In order to
use this module, the user must have a credential.json file in the data
directory. The credientials.json file must be formatted as follows:
    {
      "API_KEY": "xxxxxxxxxx",
      "API_SECRET": "xxxxxxxxxx",
      "ACCESS_TOKEN": "172722928-xxxxxxxxxx",
      "ACCESS_TOKEN_SECRET": "xxxxxxxxxx"
    }

To use this, the user only needs to use one function inside this module, named
"download_tweets_given_search_query"
"""


def access_credentials():
    """
    Args:
        None
    Returns:
        API_KEY (str): the API_KEY from credentials.json
        API_SECRET (str): the API_SECRET from credentials.json
        ACCESS_TOKEN (str): the ACCESS_TOKEN from credentials.json
        ACCESS_TOKEN_SECRET (str): the ACCESS_TOKEN_SECRET
        from credentials.json
    """
    with open('data/credentials.json') as f:
        data = json.load(f)
    API_KEY = data['API_KEY']
    API_SECRET = data['API_SECRET']
    ACCESS_TOKEN = data['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = data['ACCESS_TOKEN_SECRET']
    return API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET


def download_tweets_to_list(searchQuery, tweetsPerQry,
                            maxTweets, api, verbose=False):
    """
    This code was adopted and modified from Bhaskar Karambelkar, the link
    to his blog post is below
    http://www.karambelkar.info/2015/01/how-to-use-
    twitters-search-rest-api-most-effectively./

    Args:
        searchQuery (str): this is the search query, it should follow
        the twitter api query structure
        tweetsPerQry (int): this is how many tweets you download
        per query to the website
        maxTweets (int): this is the maximum number of tweets
        you want to download
        api (api object):  this is your api object, you create this with:

        Example:
            API_KEY, API_SECRET,
            ACCESS_TOKEN, ACCESS_TOKEN_SECRET = access_credentials()
            auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
            api = tweepy.API(auth, wait_on_rate_limit=True,
                             wait_on_rate_limit_notify=True)
    Returns:
        tweet_list (list): the list of the actual tweets in json format
        downloaded via tweepy
    """
    tweet_list = []
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
                print e
                print type(e)
                print e.__dict__
                print e.reason
                print type(e.reason)
            return tweet_list
    print("Downloaded {0} tweets".format(tweetCount))
    return tweet_list


def download_tweets_given_search_query(searchQuery, verbose=False):
    """
    Args:
        searchQuery (str): the query, in twitter api seach query format,
        that you wish to download tweets of on twitter
    Returns:
        tweet_list (list): list of json tweet objects
    """
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
