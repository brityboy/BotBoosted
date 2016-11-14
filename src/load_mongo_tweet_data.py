import dill as pickle
from pymongo import MongoClient
from unidecode import unidecode
import pandas as pd


def load_pred_dict_from_pickle(filename):
    '''
    INPUT
         - filename: for the location of the pkl file
    OUTPUT
         - dictionary

    Returns a dictionary where the keys are the twitter id's and the values
    are the prediction for whether it is fake or not
    '''
    with open(filename, 'r') as f:
        pred_dict = pickle.load(f)
    return pred_dict


def get_tweets(dbname, collectionname, pred_dict):
    '''
    INPUT
         - dbname: name of the mongo db to connect to
         - collectionname: name of table inside db (but this is really for
         the topictweets db)
         - pred_dict: dictionary which has keys = user id's and values = pred
         for whether they are fake or not
    OUTPUT
        - df
    Returns a dataframe which has the user_id, the tweet, and whether it is
    fake or not
    '''
    user_id_list = []
    textlist = []
    username_list = []
    client = MongoClient()
    db = client[dbname]
    tab = db[collectionname].find()
    for document in tab:
        user_id_list.append(str(document['user']['id']))
        username_list.append(unidecode(document['user']['screen_name']))
        textlist.append(unidecode(document['text']))
    df = pd.DataFrame(user_id_list, columns=['id'])
    df['text'] = textlist
    df['screen_name'] = username_list
    return df


if __name__ == "__main__":
    pred_dict = load_pred_dict_from_pickle('data/trumpmillion_pred_dict.pkl')
    df = get_tweets('trumpmillion', 'topictweets', pred_dict)
    df_classified_users = df[df.id.isin(pred_dict)]
    df_classified_users['pred'] = \
        df_classified_users.id.apply(lambda _id: pred_dict[_id])
    df_classified_users.to_csv('data/trumptweets.csv', index=None)
