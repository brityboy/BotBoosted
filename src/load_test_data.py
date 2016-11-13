from unidecode import unidecode
from pymongo import MongoClient
from collections import defaultdict
import pandas as pd
from process_loaded_data import *
import dill as pickle


def get_username_list_from_mongo(dbname, collectionname):
    '''
    INPUT
         - dbname: this is a name of a mongo database
         - collectionname: this is a name of a collection within the mongo db
    OUTPUT
         - list
    Returns the list of usernames who tweeted about that topic
    '''
    username_list = set()
    client = MongoClient()
    db = client[dbname]
    tab = db[collectionname].find()
    for document in tab:
        username_list.add(unidecode(document['user']['screen_name']))
    return list(username_list)


def extract_user_information_from_mongo(dbname, collectionname):
    '''
    INPUT
         - dbname: this is a name of a mongo database
         - collectionname: this is a name of a collection within the mongo db
         this is MADE to run for the topictweets collection
    OUTPUT
         - dataframe
    targets to extract right now are:
    a) has_30_followers 1 0
    b) geo_localized 1 0
    c) followers_friends
    '''
    user_id_list = []
    has_30_followers_list = []
    geo_localized_list = []
    followers_friends_list = []
    client = MongoClient()
    db = client[dbname]
    tab = db[collectionname].find()
    for document in tab:
        user_id = str(document['user']['id'])
        followers_count = document['user']['followers_count']
        friends_count = document['user']['friends_count']
        geo_enabled = document['user']['geo_enabled']
        user_id_list.append(user_id)
        has_30_followers_list.append(1) if followers_count >= 0 else \
            has_30_followers_list.append(0)
        geo_localized_list.append(1) if geo_enabled is True else \
            geo_localized_list.append(0)
        followers_friends_list.append(1) if 2 * followers_count >= \
            friends_count else followers_friends_list.append(0)
    result = pd.DataFrame(columns=['id', 'has_30_followers', 'geo_localized',
                                   'followers_friends'])
    result['id'] = user_id_list
    result['has_30_followers'] = has_30_followers_list
    result['geo_localized'] = geo_localized_list
    result['followers_friends'] = followers_friends_list
    return result


def extract_feature_information_from_mongo(dbname, collectionname):
    '''
    INPUT
         - dbname: this is a name of a mongo database
         - collectionname: this is a name of a collection within the mongo db
         this is made to run for the timeline tweets
    OUTPUT
         - dictionary

    returns a nested dictionary where the key is the username, the values are
    the features, and then the values of the feature keys are the values
    This approach will be taken and multiprocessing and threading
    will be used if possible in order to speed up the extraction from both
    the csv files and the mongo databases
    targets to extract right now are:
    a) has been included in another user's favorites (favorite_count)
    get total number of times a users tweet has been
    favorited (sum favorite_count)
    mongo - favorite_count
    b) has used a hashtag in the tweet
    mongo - len(document['entities']['hashtags'])
    c) has logged into twitter via iphone (source)
    mongo - source
    d) has mentioned another user
    mongo - len(document['entities']['user_mentions'])
    '''
    result = defaultdict(defaultdict)
    client = MongoClient()
    db = client[dbname]
    tab = db[collectionname].find()
    for document in tab:
        user_id = str(document['user']['id'])
        favorite_count = document['favorite_count']
        num_hashtags = len(document['entities']['hashtags'])
        iphone_source = 1 if 'iPhone' in document['source'] else 0
        num_mentions = len(document['entities']['user_mentions'])
        if 'favorite_count' not in result[user_id]:
            result[user_id]['favorite_count'] = 0
        if 'num_hashtags' not in result[user_id]:
            result[user_id]['num_hashtags'] = 0
        if 'iphone_source' not in result[user_id]:
            result[user_id]['iphone_source'] = 0
        if 'num_mentions' not in result[user_id]:
            result[user_id]['num_mentions'] = 0
        result[user_id]['favorite_count'] += favorite_count
        result[user_id]['num_hashtags'] += num_hashtags
        result[user_id]['iphone_source'] += iphone_source
        result[user_id]['num_mentions'] += num_mentions
    return result


def process_feature_information_for_modelling(df, feature_dict):
    '''
    INPUT
         - df: pandas dataframe where user info has been matched with
         the features dict
    OUTPUT
         - df

    Returns the modified features from the feature dict into the dataframe
    '''
    df = combine_user_info_with_feature_dict(df, feature_dict)
    df['favorited_by_another'] = \
        df.twt_favorite_count.apply(lambda favcnt: 1 if favcnt > 0 else 0)
    df['has_hashtagged'] = \
        df.num_hashtags.apply(lambda hashtag: 1 if hashtag > 0 else 0)
    df['used_iphone'] = \
        df.iphone_source.apply(lambda iphone: 1 if iphone > 0 else 0)
    df['has_mentions'] = \
        df.num_mentions.apply(lambda mentions: 1 if mentions > 0 else 0)
    return df


def drop_unnecessary_columns_from_test_data(df):
    '''
    INPUT
         - df: pandas dataframe where the features have already been built
    OUTPUT
         - df
    Returns a dataframe where the unnecessary features have already been
    dropped
    '''

    df.drop(['twt_favorite_count', 'num_hashtags', 'iphone_source',
             'num_mentions'], axis=1, inplace=True)
    df = df[['id', 'has_30_followers', 'geo_localized', 'favorited_by_another',
             'has_hashtagged', 'used_iphone', 'has_mentions',
             'followers_friends']]
    return df


def create_processed_dataframe_from_mongo(dbname):
    '''
    INPUT
         - dbname: this is the name of the mongo database where the
         information will be extracted from
    OUTPUT
         - df

    Returns a dataframe that has everything needed in order to do modelling
    '''
    df = extract_user_information_from_mongo(dbname, 'topictweets')
    # df = pd.read_csv('data/clinton_df.csv')
    # df.id = df.id.apply(str)
    feature_dict = extract_feature_information_from_mongo(dbname,
                                                          'timelinetweets')
    # with open('data/clinton_tweets_dict.pkl', 'r') as f:
    #     feature_dict = pickle.load(f)
    df = df.drop_duplicates(subset='id', keep='last')
    users_who_tweeted = set(feature_dict.keys())
    dfusers_who_tweeted = df[df.id.isin(users_who_tweeted)]
    # subset the initial user dataframe to have ONLY the users who tweeted
    df = combine_user_info_with_feature_dict(dfusers_who_tweeted, feature_dict)
    df = process_feature_information_for_modelling(df, feature_dict)
    df = drop_unnecessary_columns_from_test_data(df)
    return df


def write_dict_to_pkl(dict_object, dict_name):
    '''
    INPUT
         - dict_name: str, this is the name of the dictionary
    OUTPUT
         - saves the model to a pkl file
    Returns None
    '''
    with open('data/{}_dict.pkl'.format(dict_name), 'w+') as f:
        pickle.dump(dict_object, f)


if __name__ == "__main__":
    df = create_processed_dataframe_from_mongo('trumpmillion')
    df.to_csv('data/trumpmillion.csv', index=None)
    # dbname = 'clintonmillion'
    # df = extract_user_information_from_mongo(dbname, 'topictweets')
    # feature_dict = extract_feature_information_from_mongo(dbname,
    #                                                       'timelinetweets')
