from unidecode import unidecode
from pymongo import MongoClient


def get_username_list_from_mongo(dbname, collectionname):
    username_list = []
    client = MongoClient()
    db = client[dbname]
    tab = db[collectionname].find()
    for document in tab:
        username_list.append(unidecode(document['user']['screen_name']))
    return list(set(username_list))


def get_tweets(dbname, collectionname):
    textlist = []
    client = MongoClient()
    db = client[dbname]
    tab = db[collectionname].find()
    for document in tab:
        textlist.append(unidecode(document['text']))
    return textlist


if __name__ == "__main__":
