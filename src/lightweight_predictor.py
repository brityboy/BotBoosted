import dill as pickle
from tweet_scrape_processor import process_tweet
from prediction_model import load_pickled_model
import numpy as np
import time

history_model = load_pickled_model('models/account_history_rf_model.pkl')
behavior_model = load_pickled_model('models/behavior_rate_rf_model.pkl')
ensemble_model = load_pickled_model('models/ensemble_rf_model.pkl')

if __name__ == "__main__":
    # df = pd.read_csv('data/training_user_tweet_data.csv')
    start = time.time()
    with open('data/test_tweet_scrape.pkl', 'r+') as f:
        tweet_list = pickle.load(f)
    print("load pkl file: ", time.time() - start)
    start = time.time()
    tweet_history = np.array([process_tweet(tweet) for tweet in tweet_list])
    tweet_behavior = \
        tweet_history/np.array(map(float, tweet_history[:, 13])).reshape(-1, 1)
    print("loading tweets: ", time.time() - start)
    y_pred = history_model.predict_proba(tweet_history)[:, 1]
    y_pred_b = behavior_model.predict_proba(tweet_behavior)[:, 1]
    tweet_ensemble = np.hstack((tweet_history,
                                tweet_behavior,
                                y_pred.reshape(-1, 1),
                                y_pred_b.reshape(-1, 1)))
    pred = ensemble_model.predict(tweet_ensemble)
