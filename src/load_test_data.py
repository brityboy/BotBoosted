from unidecode import unidecode
from pymongo import MongoClient
from timeit import Timer
from collections import defaultdict
import pandas as pd


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
        if 'favorite_count' not in result[user_id]:
            result[user_id]['favorite_count'] = 0
        if 'num_hashtags' not in result[user_id]:
            result[user_id]['num_hashtags'] = 0
        if 'iphone_source' not in result[user_id]:
            result[user_id]['iphone_source'] = 0
        if 'num_mentions' not in result[user_id]:
            result[user_id]['num_mentions'] = 0
        if row[14] in ['0', 'NULL']:
            result[user_id]['favorite_count'] += 0
        else:
            result[user_id]['favorite_count'] += int(row[14])
        if row[15] in ['0', 'NULL']:
            result[user_id]['num_hashtags'] += 0
        else:
            result[user_id]['num_hashtags'] += int(row[15])
        if 'iPhone' in row[3]:
            result[user_id]['iphone_source'] += 1
        else:
            result[user_id]['iphone_source'] += 0
        if row[17] in ['0', 'NULL']:
            result[user_id]['num_mentions'] += 0
        else:
            result[user_id]['num_mentions'] += int(row[17])
    pass


if __name__ == "__main__":
    # t = Timer(lambda: extract_user_information_from_mongo
            #   ('clintonmillion', 'topictweets'))
    # print("Completed sequential in %s seconds." % t.timeit(1))
    df = extract_user_information_from_mongo('clintonmillion', 'topictweets')
