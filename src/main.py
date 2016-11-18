import time
import dill as pickle
from prediction_model import *
from load_mongo_tweet_data import *
from tweet_text_processor import *


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
    del pred_dict
    del df
    process_real_and_fake_tweets(df_classified_users)
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
    history_model = load_pickled_model('models/account_history_rf.pkl')
    behavior_model = load_pickled_model('models/behavior_rate_rf.pkl')
    ensemble_model = load_pickled_model('models/ensemble_rf.pkl')
    print("loading model took: ", time.time() - start)
    print('getting and processing tweets...')
    start = time.time()
    tweet_list = download_tweets_given_search_query(searchQuery)
    # user_id_array, X = \
    #     load_processed_csv_for_predictions('data/clintonmillion.csv')
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
    del pred_dict
    del df
    process_real_and_fake_tweets(df_classified_users)
    print('\n')
    print("entire thing took: ", time.time() - totalstart)


if __name__ == "__main__":
    botboosted('marcos burial')
