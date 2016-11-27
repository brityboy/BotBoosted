import time
from tweet_text_processor import process_real_and_fake_tweets_w_plots
from tweet_scraper import download_tweets_given_search_query
from lightweight_predictor import make_lightweight_predictions_v2
from pymongo import MongoClient

"""
This module integrates all of the different modules into two main functions:

a) botboosted - this function allows a user to specify a search, and
then this will download the tweets, classify them, and visualize them as
different subtopics, and summarize them with exemplary tweets

b) botboosted_demonstration - this function replicates the previous
function but works on tweets inside a local mongodb rather than tweets that
are downloaded through twitter's api
"""


def botboosted_demonstration(dbname, collection, verbose=True,
                             searchQuery='Your Topic',
                             save=False):
    """
    Args:
        dbname (str): the name of the mongodb to connect to which has
        the tweets to be analyzed
        collection (str): the name of the collection in that db to access
        verbose (boolean): whether to print out all outputs or not
        searchQuery (str): this is a string that represents the contents
        of the collection, to be used in the title of the barplots
        save (boolean): whether to save the prediction information to a
        csv or not
    Returns:
         None, plots barplots and stacked barplots with representative tweets
         for that topic
    """
    client = MongoClient()
    if verbose:
        totalstart = time.time()
        print('loading model...')
        start = time.time()
    if verbose:
        print("loading model took: ", time.time() - start)
        print('getting and processing tweets...')
        start = time.time()
    tweet_list = []
    db = client[dbname]
    tab = db[collection].find()
    for document in tab:
        tweet_list.append(document)
    if verbose:
        print("loading and processing tweet data took: ", time.time() - start)
        print('making predictions...')
        start = time.time()
    predicted_tweets = make_lightweight_predictions_v2(tweet_list)
    if save:
        filename = 'pred_v2_{}.csv'.format(searchQuery.replace(' ', '_'))
        predicted_tweets.to_csv(filename)
    if verbose:
        del tweet_list
        print("making predictions took: ", time.time() - start)
    process_real_and_fake_tweets_w_plots(predicted_tweets, verbose=verbose,
                                         searchQuery=searchQuery)
    if verbose:
        print('\n')
        print("entire thing took: ", time.time() - totalstart)


def botboosted(searchQuery, verbose=False, save=False):
    """
    Args:
        searchQuery (str): this is a twitter api search query to use
        for tweets to be downloaded from twitter and analyzed
        verbose (boolean): whether to print out all outputs or not
        save (boolean): whether to save the prediction information to a
        csv or not
    Returns:
         None, plots barplots and stacked barplots with representative tweets
         for that topic
    """
    if verbose:
        totalstart = time.time()
        print('loading model...')
        start = time.time()
    if verbose:
        print("loading model took: ", time.time() - start)
        print('getting and processing tweets...')
        start = time.time()
    tweet_list = download_tweets_given_search_query(searchQuery,
                                                    verbose=verbose)
    if verbose:
        print("loading and processing tweet data took: ", time.time() - start)
        print('making predictions...')
        start = time.time()
    predicted_tweets = make_lightweight_predictions_v2(tweet_list)
    if save:
        filename = 'pred_v2_{}.csv'.format(searchQuery.replace(' ', '_'))
        predicted_tweets.to_csv(filename)
    if verbose:
        print("making predictions took: ", time.time() - start)
    process_real_and_fake_tweets_w_plots(predicted_tweets, verbose=verbose,
                                         searchQuery=searchQuery)
    if verbose:
        print('\n')
        print("entire thing took: ", time.time() - totalstart)

if __name__ == "__main__":
    botboosted('make money online fast',
               verbose=True,
               save=True)
    # botboosted_demonstration('spammytweets',
    #                          'freeiphone',
    #                          verbose=True,
    #                          searchQuery='win a free iphone',
    #                          save=False)
