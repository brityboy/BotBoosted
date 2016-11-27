import dill as pickle
from tweet_scrape_processor import process_tweet, process_tweet_v2
import numpy as np
import time
import pandas as pd
from unidecode import unidecode

"""
The objective of this module is to load the random forest ensemble in order
to create a dataframe that has the major information about the predicted
tweets. This information includes the user's screen_name, the tweet itself,
and whether the tweet is real or fake

Of the two functions here, make_lightweight_predictions is deprecated
as it uses models that have solely been trained on the original data
provided by Cresci and co. make_lightweight_predictions_v2, on the other
hand, has had some exposure to the spammers today that. While not a lot, (as it
was trained with roughly 100 additional manually labelled spammers that were
not detected by Twitter), it can pick up some signal given search queries on
Twitter today.

These functions simply need a list of twitter json objects, and these
functions will process them, and produce a dataframe with the username,
the tweet, and the prediction, for the next step in the process
which is to process the tweet text data into the meaninful topics
"""


def load_pickled_model(filename):
    """
    Args:
        filename (str): path and name of the model to load from a pkl file
    Returns:
        model (sklearn classifier model): fit already, and ready to predict
    """
    with open(filename, 'r') as f:
        model = pickle.load(f)
    f.close()
    return model


history_model = load_pickled_model('models/account_history_rf_model.pkl')
behavior_model = load_pickled_model('models/behavior_rate_rf_model.pkl')
ensemble_model = load_pickled_model('models/ensemble_rf_model.pkl')
history_model_v2 = load_pickled_model('models/account_history_rf_v2_model.pkl')
behavior_model_v2 = load_pickled_model('models/behavior_rate_rf_v2_model.pkl')
ensemble_model_v2 = load_pickled_model('models/ensemble_rf_v2_model.pkl')


def make_lightweight_predictions(tweet_list):
    """
    Args:
        tweet_list (list): list of json tweet objects downloaded from twitter
    Returns
        predicted_tweets (dataframe): a dataframe with the user id, the
        text content of the tweet, the screen_name of the user, and
        the predicted value where 1 = fake and 0 means human
    """
    start = time.time()
    processed_tweets = np.array([process_tweet(tweet) for tweet in tweet_list])
    tweet_history = processed_tweets[:, :19].astype(float)
    tweets = processed_tweets[:, 19:]
    tweet_behavior = \
        tweet_history/tweet_history[:, 13].reshape(-1, 1)
    print("loading tweets: ", time.time() - start)
    y_pred = history_model.predict_proba(tweet_history)[:, 1]
    y_pred_b = behavior_model.predict_proba(tweet_behavior)[:, 1]
    tweet_ensemble = np.hstack((tweet_history,
                                tweet_behavior,
                                y_pred.reshape(-1, 1),
                                y_pred_b.reshape(-1, 1)))
    pred = ensemble_model.predict(tweet_ensemble)
    predicted_tweets = np.hstack((tweets, pred.reshape(-1, 1)))
    columns = ['id', 'text', 'screen_name', 'pred']
    predicted_tweets = pd.DataFrame(predicted_tweets, columns=columns)
    predicted_tweets.pred = predicted_tweets.pred.apply(float).apply(int)
    predicted_tweets.text = predicted_tweets.text.apply(unidecode)
    return predicted_tweets


def make_lightweight_predictions_v2(tweet_list):
    """
    Args:
        tweet_list (list): list of json tweet objects downloaded from twitter
    Returns
        predicted_tweets (dataframe): a dataframe with the user id, the
        text content of the tweet, the screen_name of the user, and
        the predicted value where 1 = fake and 0 means human
    """
    start = time.time()
    processed_tweets = np.array([process_tweet_v2(tweet)
                                 for tweet in tweet_list])
    tweet_history = processed_tweets[:, :23].astype(float)
    tweets = processed_tweets[:, 23:]
    tweet_behavior = \
        tweet_history/tweet_history[:, 13].reshape(-1, 1)
    print("loading tweets: ", time.time() - start)
    y_pred = history_model_v2.predict_proba(tweet_history)[:, 1]
    y_pred_b = behavior_model_v2.predict_proba(tweet_behavior)[:, 1]
    tweet_ensemble = np.hstack((tweet_history,
                                tweet_behavior,
                                y_pred.reshape(-1, 1),
                                y_pred_b.reshape(-1, 1)))
    pred = ensemble_model_v2.predict(tweet_ensemble)
    predicted_tweets = np.hstack((tweets, pred.reshape(-1, 1)))
    columns = ['id', 'text', 'screen_name', 'pred']
    predicted_tweets = pd.DataFrame(predicted_tweets, columns=columns)
    predicted_tweets.pred = predicted_tweets.pred.apply(float).apply(int)
    predicted_tweets.text = predicted_tweets.text.apply(unidecode)
    return predicted_tweets


if __name__ == "__main__":
    # df = pd.read_csv('data/training_user_tweet_data.csv')
    start = time.time()
    with open('data/test_tweet_scrape.pkl', 'r+') as f:
        tweet_list = pickle.load(f)
    print("load pkl file: ", time.time() - start)
    predicted_tweets = make_lightweight_predictions_v2(tweet_list)
