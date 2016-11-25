import dill as pickle
from tweet_scrape_processor import process_tweet, process_tweet_v2
from prediction_model import load_pickled_model
import numpy as np
import time
import pandas as pd
from unidecode import unidecode

history_model = load_pickled_model('models/account_history_rf_model.pkl')
behavior_model = load_pickled_model('models/behavior_rate_rf_model.pkl')
ensemble_model = load_pickled_model('models/ensemble_rf_model.pkl')
history_model_v2 = load_pickled_model('models/account_history_rf_v2_model.pkl')
behavior_model_v2 = load_pickled_model('models/behavior_rate_rf_v2_model.pkl')
ensemble_model_v2 = load_pickled_model('models/ensemble_rf_v2_model.pkl')


def make_lightweight_predictions(tweet_list):
    '''
    INPUT
         - tweet_list, list of json objects
    OUTPUT
         - pred, array

    Returns a prediction for each tweet as to whether it is real or fake
    '''
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
    '''
    INPUT
         - tweet_list, list of json objects
    OUTPUT
         - pred, array

    Returns a prediction for each tweet as to whether it is real or fake
    '''
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
