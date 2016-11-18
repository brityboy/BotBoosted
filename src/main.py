import time
import dill as pickle
from prediction_model import load_pickled_model
from prediction_model import load_processed_csv_for_predictions
from prediction_model import create_dictionary_with_id_and_predictions
from load_mongo_tweet_data import load_pred_dict_from_pickle, get_tweets
from tweet_text_processor import process_real_and_fake_tweets
from tweet_scraper import download_tweets_given_search_query
from tweet_scrape_processor import process_tweet
from lightweight_predictor import make_lightweight_predictions
from pymongo import MongoClient


def demonstration_on_loaded_data_using_original_model():
    '''
    INPUT
         - nothing
    OUTPUT
         - processes the entire pipeline on data from a local using
           the tweet history (Class A and Class B features)
    Returns none
    '''
    totalstart = time.time()
    print('loading model...')
    start = time.time()
    model = load_pickled_model('models/voting_ensemble_model.pkl')
    print("loading model took: ", time.time() - start)
    print('loading user info for predictions...')
    start = time.time()
    user_id_array, X = \
        load_processed_csv_for_predictions('data/clintonmillion.csv')
    print("loading data took: ", time.time() - start)
    print('making predictions...')
    start = time.time()
    create_dictionary_with_id_and_predictions(model, user_id_array, X,
                                              'clintonmillion_pred')
    pred_dict = load_pred_dict_from_pickle('data/clintonmillion_pred_dict.pkl')
    print("making predictions took: ", time.time() - start)
    del model
    print('loading tweets...')
    start = time.time()
    df = get_tweets('clintonmillion', 'topictweets', pred_dict)
    df_classified_users = df[df.id.isin(pred_dict)]
    df_classified_users['pred'] = \
        df_classified_users.id.apply(lambda _id: pred_dict[_id])
    print("loading tweets took: ", time.time() - start)
    # del pred_dict
    del df
    process_real_and_fake_tweets(df_classified_users)
    print('\n')
    print("entire thing took: ", time.time() - totalstart)


def botboosted_demonstration(searchQuery):
    '''
    INPUT
         - searchQuery - string
    OUTPUT
         - entire pipeline
    Returns none
    '''
    client = MongoClient()
    totalstart = time.time()
    print('loading model...')
    start = time.time()
    history_model = load_pickled_model('models/account_history_rf_model.pkl')
    behavior_model = load_pickled_model('models/behavior_rate_rf_model.pkl')
    ensemble_model = load_pickled_model('models/ensemble_rf_model.pkl')
    print("loading model took: ", time.time() - start)
    print('getting and processing tweets...')
    start = time.time()
    tweet_list = []
    if searchQuery == 'hillary clinton email':
        db = client['clintonmillion']
        tab = db['topictweets'].find()
        for document in tab:
            tweet_list.append(document)
    elif searchQuery == 'donald trump sexual assault':
        db = client['trumpmillion']
        tab = db['topictweets'].find()
        for document in tab:
            tweet_list.append(document)
    else:
        print('that search query is not in the sample datasets')
        return None
    print("loading and processing tweet data took: ", time.time() - start)
    print('making predictions...')
    start = time.time()
    predicted_tweets = make_lightweight_predictions(tweet_list)
    print("making predictions took: ", time.time() - start)
    process_real_and_fake_tweets(predicted_tweets)
    print('\n')
    print("entire thing took: ", time.time() - totalstart)


def botboosted(searchQuery):
    '''
    INPUT
         - searchQuery - string
    OUTPUT
         - entire pipeline
    Returns none
    '''
    totalstart = time.time()
    print('loading model...')
    start = time.time()
    history_model = load_pickled_model('models/account_history_rf_model.pkl')
    behavior_model = load_pickled_model('models/behavior_rate_rf_model.pkl')
    ensemble_model = load_pickled_model('models/ensemble_rf_model.pkl')
    print("loading model took: ", time.time() - start)
    print('getting and processing tweets...')
    start = time.time()
    tweet_list = download_tweets_given_search_query(searchQuery)
    print("loading and processing tweet data took: ", time.time() - start)
    print('making predictions...')
    start = time.time()
    predicted_tweets = make_lightweight_predictions(tweet_list)
    print("making predictions took: ", time.time() - start)
    process_real_and_fake_tweets(predicted_tweets)
    print('\n')
    print("entire thing took: ", time.time() - totalstart)


if __name__ == "__main__":
    # botboosted('hillary clinton')
    botboosted_demonstration('hillary clinton email')
