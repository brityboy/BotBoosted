import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from process_loaded_data import check_if_many_relative_followers_to_friends
from datetime import datetime
from pymongo import MongoClient
from tweet_scrape_processor import process_tweet
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from dill import pickle

"""
This module is used to create the random forest ensemble that will classify
a user as real or fake based on a single tweet and the user data embedded
in that json object

Previous research in the area of classifying twitter users as real or fake
has done so by using class A (lightweight) and class B (costlier) features.
Lightweight features include everything that you can get from a single tweet
(total tweets, follower, likes, account creation date) as these are embedded
in the json object that one can get when downloading a tweet via twitter's
API. Costlier features include a user's tweet history,
meaning the tweets themselves.

The contribution to the research community of this lightweight classifier is
a classification method that relies solely on class A features. The approach
is as follows: a) create features from user's account history (total likes,
total tweets, total followers, total friends, etc) b) create features that
express relative volume (total likes divided by total number of followers,
total tweets divided by total number of friends, etc) as it was observed that
some accounts have hundreds and thousands of tweets but very few people in
their network c) create features that express behavior rate (total likes per
day, total tweets per day, total likes per friends per day) as the account
creation date is available in the json object and it was observed that fake
accounts do "machine gun" tweeting where they tweet very frequently in a small
period of time. These set of features was added in order to also make the
model less naive to new users

No features took the content or the words of the tweet into account (i.e. NLP
based prediction) as the premise is that a human is always behind the message
being artificially propagated. The behavior captured by the tweet was taken
into account by looking at hashtag usage, mentions, whether the tweet was
favorited by another person, etc.

The classification model is a random forest ensemble made up of three random
forest models.

Random Forest 1 (RF1) takes in account history features and relative volume
features
Random Forest 2 (RF2) takes in behavior rate features that look at account
history features per day and relative volume features per day
Random Forest 3 (RF3) takes in the predicted probabilities of Random Forest
1 and Random Forest 2, along with all of these models features, and then
makes the final prediction.
The final Random Forest is able to balance out the work of the previous ones
by understanding the user patterns along the two major facets: account
history and account behavior rate.

The ten fold cross validated accuracy of RF1 is 97%, RF2 has 95%, and RF3
has 98%. Previous research using this dataset achieved these kinds of scores
as well. However, they did so with class A and class B features. The
contribution of this work is that this kind of performance was attained
using only class A features.

To run this, just run the function:
    create_ensemble_model()

Todo:
    * train the model with more samples from today's set of Twitter's
    false negatives so that the model can understand the patterns of
    the spammers of today
"""


def view_feature_importances(df, model):
    """
    Args:
        df (pandas dataframe): dataframe which has the original data
        model (sklearn model): this is the sklearn classification model that
        has already been fit (work with tree based models)
    Returns:
        nothing, this just prints the feature importances in descending order
    """
    columns = df.columns
    features = model.feature_importances_
    featimps = []
    for column, feature in zip(columns, features):
        featimps.append([column, feature])
    print(pd.DataFrame(featimps, columns=['Features',
                       'Importances']).sort_values(by='Importances',
                                                   ascending=False))


def evaluate_model(model, X_train, y_train):
    """
    Args:
        model (sklearn classification model): this model from sklearn that
        will be used to fit the data and to see the 10 fold cross val score of
        X_train (2d numpy array): this is the feature matrix
        y_train (1d numpy array): this is the array of targets
    Returns:
        prints information about the model's accuracy using 10
         fold cross validation
        model (sklearn classification model): the model that has already been
        fit to the data
    """
    print(np.mean(cross_val_score(model, X_train, y_train,
                                  cv=10, n_jobs=-1, verbose=10)))
    model.fit(X_train, y_train)
    return model


def write_model_to_pkl(model, model_name):
    """
    Args:
        model_name (str): this is the name of the model
        model (sklearn fit model): the sklearn classification model
        that will be saved to a pkl file
    Returns:
        nothing, saves the model to a pkl file
    """
    with open('models/{}_model.pkl'.format(model_name), 'w+') as f:
        pickle.dump(model, f)


def view_classification_report(model, X_test, y_test):
    """
    Args
        model (sklearn classification model): this model from sklearn that
        will has already been fit
        X_test (2d numpy array): this is the feature matrix
        y_test (1d numpy array): this is the array of targets
    Returns
        nothing, this is just a wrapper for the classification report
    """
    print(classification_report(y_test, model.predict(X_test)))


def gridsearch(paramgrid, model, X_train, y_train):
    """
    Args:
        paramgrid (dictionary): a dictionary of lists where the keys are the
        model's tunable parameters and the values are a list of the
        different parameter values to search over
        X_train (2d numpy array): this is the feature matrix
        y_train (1d numpy array): this is the array of targets
    Returns:
        best_model (sklearn classifier): a fit sklearn classifier with the
        best parameters from the gridsearch
        gridsearch (gridsearch object): the gridsearch object that has
        already been fit
    """
    gridsearch = GridSearchCV(model,
                              paramgrid,
                              n_jobs=-1,
                              verbose=10,
                              cv=10)
    gridsearch.fit(X_train, y_train)
    best_model = gridsearch.best_estimator_
    print('these are the parameters of the best model')
    print(best_model)
    print('\nthese is the best score')
    print(gridsearch.best_score_)
    return best_model, gridsearch


def balance_classes(sm, X, y):
    """
    Args:
        sm (imblearn class): this is an imbalance learn oversampling or
        undersampling class
        X (2d numpy array): this is the feature matrix
        y (1d numpy array): this is the array of the targets
    Returns:
        X (2d numpy array): this is the balanced feature matrix
        y (1d numpy array): this is the corresponding balanced target array
    Returns X and y after being fit with the resampling method
    """
    X, y = sm.fit_sample(X, y)
    return X, y


def load_all_training_data():
    """
    Args:
         - none
    Returns:
        df (pandas dataframe): the training dataframe with the ff
        things done to it:
            a) protected accounts dropped
            b) irrelevant columns removed
    """
    df = pd.read_csv('data/all_user_data.csv')
    df = df.query('protected != 1')
    df.drop(['profile_image_url_https',
             'profile_sidebar_fill_color',
             'profile_text_color',
             'profile_background_color',
             'profile_link_color',
             'profile_image_url',
             'profile_background_image_url_https',
             'profile_banner_url',
             'profile_background_image_url',
             'profile_background_tile',
             'profile_sidebar_border_color',
             'default_profile',
             'file',
             'time_zone',
             'screen_name',
             'utc_offset',
             'protected'], axis=1, inplace=True)
    return df


def get_most_recent_tweets_per_user():
    """
    Args:
        none
    Returns
        df (pandas dataframe): Returns a dataframe with only one tweet per
        row which is the MOST recent tweet recorded for that user_id
    """
    tweetdf = pd.read_csv('data/training_tweets.csv')
    tweetdf.timestamp = pd.to_datetime(tweetdf.timestamp)
    index = tweetdf.groupby('user_id').apply(lambda x: np.argmax(x.timestamp))
    tweetdf = tweetdf.loc[index.values]
    tweetdf.reset_index().drop('Unnamed: 0', axis=1, inplace=True)
    tweetdf.drop('Unnamed: 0', axis=1, inplace=True)
    return tweetdf


def load_master_training_df():
    """
    Args:
        none
    Returns
        df (pandas dataframe): Returns dataframe combining most recent tweet
        info with user info. notes on the columns:
            updated - when the account was last updated
            geo_enabled - if the account is geo enabled
            description - text which has the user input self-description
            verified - if the account is verified or not
            followers_count - number of followers
            location - string, location
            default_profile_image - binary, yes or no
            listed_count - how many times the account was listed
            statuses count - number of tweets posted
            friends_count - number of accounts the user is following
            name - user specified user name
            lang - user specified user language (CANNOT BE USED)
            favourites_count - number of items favourited
            url - user specified url
            created_at - date the account was created
            user_id - twitter assigned user id (unique in the twittersphere)
            favorite_count - times the tweet was favorited
            num_hashtags - number of hashtags used in the tweet
            text - the tweet contents
            source - the device used to upload the tweet
            num_mentions - number of users mentioned in the tweet
            timestamp - timestamp of the tweet
            geo - if the tweet was geo localized or not
            place - user specified place of the tweet
            retweet_count - number of times the tweet was retweeted
    """
    df = load_all_training_data()
    tweetdf = get_most_recent_tweets_per_user()
    users_who_tweeted = set(tweetdf.user_id.apply(int))
    df = df[df.id.isin(users_who_tweeted)]
    df['user_id'] = df['id']
    df = pd.merge(df, tweetdf, on='user_id')
    df.drop(['id',
             'label_x',
             'reply_count',
             'file'], axis=1, inplace=True)
    df.updated = pd.to_datetime(df.updated)
    df.created_at = df.created_at.apply(convert_created_time_to_datetime)
    account_age = df.timestamp - df.created_at
    account_age = map(get_account_age_in_days, account_age.values)
    df['account_age'] = account_age
    return df


def get_account_age_in_days(numpy_time_difference):
    """
    Args
        numpy_time_difference (numpy timedelta): a numpy timedelta object
        that is the difference between the user's account creation date
        and the date of their most recent tweet
    Return
        account_age (int)
    """
    return int(numpy_time_difference/1000000000/60/60/24)+1


def convert_created_time_to_datetime(datestring):
    """
    Args:
        datestring (str): a string object either as a date or
         a unix timestamp
    Returns
        datetime_object (pandas datetime object): the converted string as
        a datetime object
    """
    if len(datestring) == 30:
        return pd.to_datetime(datestring)
    else:
        return pd.to_datetime(datetime.fromtimestamp(int(datestring[:10])))


def feature_engineering(df):
    """
    Args:
        df (pandas dataframe): the initial pandas dataframe with the user
        and tweet information
    Returns
        df (pandas dataframe): the processed dataframe with the features
        needed for the model
    Returns - features needed for the model
    """
    df = check_if_many_relative_followers_to_friends(df)
    df['has_30_followers'] = \
        df.followers_count.apply(lambda x: 1 if x >= 30 else 0)
    df['favorited_by_another'] = \
        df.favorite_count.apply(lambda favcnt: 1 if favcnt > 0 else 0)
    df['has_hashtagged'] = \
        df.num_hashtags.apply(lambda hashtag: 1 if hashtag > 0 else 0)
    df['has_mentions'] = \
        df.num_mentions.apply(lambda mentions: 1 if mentions > 0 else 0)
    df = df.fillna(-999)
    return df


def get_and_process_mongo_tweets(dbname, collection):
    """
    Args:
        dbname (str): the name of the mongo db to connect to
        collection (str): the name of the table inside that mongodb
    Returns:
        tweet_history (list): this is the list of processed tweets
    """
    client = MongoClient()
    tweet_list = []
    db = client[dbname]
    tab = db[collection].find()
    for document in tab:
        tweet_list.append(document)
    processed_tweets = np.array([process_tweet(tweet) for tweet in tweet_list])
    tweet_history = processed_tweets[:, :19].astype(float)
    return tweet_history


def drop_unnecessary_features(df):
    """
    Args:
        df (pandas dataframe): the initial pandas dataframe
    Returns
        df (pandas dataframe): a dataframe where the unnecessary
        features have been dropped
    """
    df.drop(['updated', 'description', 'location', 'name', 'lang', 'url',
             'user_id', 'text', 'source', 'timestamp', 'created_at', 'geo',
             'place'], axis=1, inplace=True)
    return df


def behavior_network_ratio_feature_creation(df):
    """
    Args:
        df (pandas dataframe): initial dataframe
    Returns:
        df (pandas dataframe): the dataframe with additional features,
        where -999 is applied if an inf or a nan will appear, to denote
        missing values
    """
    df['tweets_followers'] = df.statuses_count / df.followers_count
    df['tweets_friends'] = df.statuses_count / df.friends_count
    df['likes_followers'] = df.favourites_count / df.followers_count
    df['likes_friends'] = df.favourites_count / df.friends_count
    df.tweets_followers = \
        df.tweets_followers.apply(lambda tf: -999 if tf == np.inf else tf)
    df.tweets_friends = \
        df.tweets_friends.apply(lambda tf: -999 if tf == np.inf else tf)
    df.likes_followers = \
        df.likes_followers.apply(lambda lf: -999 if lf == np.inf else lf)
    df.likes_friends = \
        df.likes_friends.apply(lambda lf: -999 if lf == np.inf else lf)
    df = df.fillna(-999)
    return df


def create_ensemble_model():
    """
    Args:
        none
    Returns
        nothing, this prints out the performance of the model and then
        saves it to a pkl file, it takes in the training user tweet data,
        the miley cyrus data, and the celebrity users data, as additional
        information to train the model to sense spam in today's times
    """
    print('this is the portion that checks absolute user behavior values')
    df = pd.read_csv('data/training_user_tweet_data.csv')
    cyrusdf = pd.read_csv('data/mileycyrususers.csv')
    celebdf = pd.read_csv('data/celebrityusers.csv')
    df = behavior_network_ratio_feature_creation(df)
    cyrusdf = behavior_network_ratio_feature_creation(cyrusdf)
    celebdf = behavior_network_ratio_feature_creation(celebdf)
    print('this is the portion that checks absolute user behavior values')
    y = df.pop('label_y')
    y = y.values
    X = df.values
    ycyrus = cyrusdf.pop('label_y')
    ycyrus = ycyrus.values
    yceleb = celebdf.pop('label_y')
    yceleb = yceleb.values
    cyrusX = cyrusdf.values
    celebX = celebdf.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    cyX_train, cyX_test, cyy_train, cyy_test = \
        train_test_split(cyrusX, ycyrus, test_size=.2)
    ceX_train, ceX_test, cey_train, cey_test = \
        train_test_split(celebX, yceleb, test_size=.2)
    X_train_b, y_train_b = balance_classes(RandomUnderSampler(),
                                           X_train, y_train)
    X_test_b, y_test_b = balance_classes(RandomUnderSampler(),
                                         X_test, y_test)
    X_train_b = np.vstack((X_train_b, cyX_train, ceX_train))
    y_train_b = np.hstack((y_train_b, cyy_train, cey_train))
    X_test_b = np.vstack((X_test_b, cyX_test, ceX_test))
    y_test_b = np.hstack((y_test_b, cyy_test, cey_test))
    weights = 1
    X_train_bw = X_train_b * weights
    paramgrid = {'n_estimators': [200],
                 'max_features': ['auto'],
                 'criterion': ['entropy'],
                 'min_samples_split': [10],
                 'min_samples_leaf': [8],
                 'max_depth': [30],
                 'bootstrap': [True]}
    model = RandomForestClassifier(n_jobs=-1)
    model, gs = gridsearch(paramgrid, model, X_train_bw, y_train_b)
    print("\nthis is the model performance on the training data\n")
    view_classification_report(model, X_train_bw, y_train_b)
    view_classification_report(model, X_test_b*weights, y_test_b)
    print(confusion_matrix(y_train_b, model.predict(X_train_bw)))
    print("this is the model performance on the test data\n")
    print(confusion_matrix(y_test_b, model.predict(X_test_b*weights)))
    print("this is the model performance on different split ratios\n")
    print("\nthese are the model feature importances\n")
    view_feature_importances(df, model)
    y_pred = model.predict_proba(X_train_bw)[:, 1]
    print('this is the portion that checks user behavior rate values')
    X_train_bwr = X_train_bw/X_train_bw[:, 13].reshape(-1, 1)
    weights = 1
    X_train_bwr = X_train_bwr * weights
    paramgrid = {'n_estimators': [200],
                 'max_features': ['auto'],
                 'criterion': ['entropy'],
                 'min_samples_split': [16],
                 'min_samples_leaf': [18],
                 'max_depth': [30],
                 'bootstrap': [True]}
    modelb = RandomForestClassifier(n_jobs=-1)
    modelb, gs = gridsearch(paramgrid, modelb, X_train_bwr, y_train_b)
    print("\nthis is the model performance on the training data\n")
    view_classification_report(modelb, X_train_bwr, y_train_b)
    print(confusion_matrix(y_train_b, modelb.predict(X_train_bwr)))
    print("this is the model performance on the test data\n")
    X_test_br = X_test_b * weights/X_test_b[:, 13].reshape(-1, 1)
    view_classification_report(modelb, X_test_br, y_test_b)
    print(confusion_matrix(y_test_b, modelb.predict(X_test_br)))
    print("\nthese are the model feature importances\n")
    view_feature_importances(df, modelb)
    y_pred_b = modelb.predict_proba(X_train_bwr)[:, 1]
    print('this is the portion that ensembles these two facets')
    ensemble_X = np.hstack((X_train_bw, X_train_bwr,
                            y_pred.reshape(-1, 1), y_pred_b.reshape(-1, 1)))
    model_ens = RandomForestClassifier(n_jobs=-1)
    paramgrid = {'n_estimators': [500],
                 'max_features': ['auto'],
                 'criterion': ['entropy'],
                 'min_samples_split': [16],
                 'min_samples_leaf': [11],
                 'max_depth': [20],
                 'bootstrap': [True]}
    model_ens, gs = gridsearch(paramgrid, model_ens, ensemble_X, y_train_b)
    print("\nthis is the model performance on the training data\n")
    view_classification_report(model_ens, ensemble_X, y_train_b)
    print(confusion_matrix(y_train_b, model_ens.predict(ensemble_X)))
    print("this is the model performance on the test data\n")
    y_pred_test = model.predict_proba(X_test_b)[:, 1]
    X_test_br = X_test_b/X_test_b[:, 13].reshape(-1, 1)
    y_pred_test_b = modelb.predict_proba(X_test_br)[:, 1]
    ensemble_X_test = np.hstack((X_test_b, X_test_br,
                                 y_pred_test.reshape(-1, 1),
                                 y_pred_test_b.reshape(-1, 1)))
    view_classification_report(model_ens, ensemble_X_test, y_test_b)
    print(confusion_matrix(y_test_b, model_ens.predict(ensemble_X_test)))
    columns = \
        list(df.columns)+[column+'_rate' for
                          column in df.columns] + \
        ['pred_model_1', 'pred_model_2']
    ensdf = pd.DataFrame(ensemble_X, columns=columns)
    view_feature_importances(ensdf, model_ens)
    print('evaluating the model on the new kind of spam')
    newX = np.vstack((cyX_test, ceX_test))
    newXbr = newX/newX[:, 13].reshape(-1, 1)
    newy = np.hstack((cyy_test, cey_test))
    newy_pred = model.predict(newX)
    newy_pred_b = modelb.predict(newXbr)
    newXens = np.hstack((newX, newXbr,
                         newy_pred.reshape(-1, 1),
                         newy_pred_b.reshape(-1, 1)))
    print(confusion_matrix(newy, model_ens.predict(newXens)))
    print('fitting to all and writing to pkl')
    y_all = np.hstack((y_train_b, y_test_b))
    behavior_X = np.vstack((X_train_bw, X_test_b))
    behavior_rate_X = np.vstack((X_train_bwr, X_test_br))
    ensemble_X = np.vstack((ensemble_X, ensemble_X_test))
    model.fit(behavior_X, y_all)
    modelb.fit(behavior_rate_X, y_all)
    model_ens.fit(ensemble_X, y_all)
    write_model_to_pkl(model, 'account_history_rf_v2')
    write_model_to_pkl(modelb, 'behavior_rate_rf_v2')
    write_model_to_pkl(model_ens, 'ensemble_rf_v2')


if __name__ == "__main__":
    df = pd.read_csv('data/training_user_tweet_data.csv')
    cyrusdf = pd.read_csv('data/mileycyrususers.csv')
    celebdf = pd.read_csv('data/celebrityusers.csv')
    df = behavior_network_ratio_feature_creation(df)
    cyrusdf = behavior_network_ratio_feature_creation(cyrusdf)
    celebdf = behavior_network_ratio_feature_creation(celebdf)
    print('this is the portion that checks absolute user behavior values')
    y = df.pop('label_y')
    y = y.values
    X = df.values
    ycyrus = cyrusdf.pop('label_y')
    ycyrus = ycyrus.values
    yceleb = celebdf.pop('label_y')
    yceleb = yceleb.values
    cyrusX = cyrusdf.values
    celebX = celebdf.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    cyX_train, cyX_test, cyy_train, cyy_test = \
        train_test_split(cyrusX, ycyrus, test_size=.2)
    ceX_train, ceX_test, cey_train, cey_test = \
        train_test_split(celebX, yceleb, test_size=.2)
    X_train_b, y_train_b = balance_classes(RandomUnderSampler(),
                                           X_train, y_train)
    X_test_b, y_test_b = balance_classes(RandomUnderSampler(),
                                         X_test, y_test)
    X_train_b = np.vstack((X_train_b, cyX_train, ceX_train))
    y_train_b = np.hstack((y_train_b, cyy_train, cey_train))
    X_test_b = np.vstack((X_test_b, cyX_test, ceX_test))
    y_test_b = np.hstack((y_test_b, cyy_test, cey_test))
    weights = 1
    X_train_bw = X_train_b * weights
    paramgrid = {'n_estimators': [200],
                 'max_features': ['auto'],
                 'criterion': ['entropy'],
                 'min_samples_split': [10],
                 'min_samples_leaf': [8],
                 'max_depth': [30],
                 'bootstrap': [True]}
    model = RandomForestClassifier(n_jobs=-1)
    model, gs = gridsearch(paramgrid, model, X_train_bw, y_train_b)
    print("\nthis is the model performance on the training data\n")
    view_classification_report(model, X_train_bw, y_train_b)
    print(confusion_matrix(y_train_b, model.predict(X_train_bw)))
    print("this is the model performance on the test data\n")
    view_classification_report(model, X_test_b*weights, y_test_b)
    print(confusion_matrix(y_test_b, model.predict(X_test_b*weights)))
    print("this is the model performance on different split ratios\n")
    print("\nthese are the model feature importances\n")
    view_feature_importances(df, model)
    y_pred = model.predict_proba(X_train_bw)[:, 1]
    print('this is the portion that checks user behavior rate values')
    X_train_bwr = X_train_bw/X_train_bw[:, 13].reshape(-1, 1)
    weights = 1
    X_train_bwr = X_train_bwr * weights
    paramgrid = {'n_estimators': [200],
                 'max_features': ['auto'],
                 'criterion': ['entropy'],
                 'min_samples_split': [16],
                 'min_samples_leaf': [18],
                 'max_depth': [30],
                 'bootstrap': [True]}
    modelb = RandomForestClassifier(n_jobs=-1)
    modelb, gs = gridsearch(paramgrid, modelb, X_train_bwr, y_train_b)
    print("\nthis is the model performance on the training data\n")
    view_classification_report(modelb, X_train_bwr, y_train_b)
    print(confusion_matrix(y_train_b, modelb.predict(X_train_bwr)))
    print("this is the model performance on the test data\n")
    X_test_br = X_test_b * weights/X_test_b[:, 13].reshape(-1, 1)
    view_classification_report(modelb, X_test_br, y_test_b)
    print(confusion_matrix(y_test_b, modelb.predict(X_test_br)))
    print("\nthese are the model feature importances\n")
    view_feature_importances(df, modelb)
    y_pred_b = modelb.predict_proba(X_train_bwr)[:, 1]
    print('this is the portion that ensembles these two facets')
    ensemble_X = np.hstack((X_train_bw, X_train_bwr,
                            y_pred.reshape(-1, 1), y_pred_b.reshape(-1, 1)))
    model_ens = RandomForestClassifier(n_jobs=-1)
    paramgrid = {'n_estimators': [500],
                 'max_features': ['auto'],
                 'criterion': ['entropy'],
                 'min_samples_split': [16],
                 'min_samples_leaf': [11],
                 'max_depth': [20],
                 'bootstrap': [True]}
    model_ens, gs = gridsearch(paramgrid, model_ens, ensemble_X, y_train_b)
    print("\nthis is the model performance on the training data\n")
    view_classification_report(model_ens, ensemble_X, y_train_b)
    print(confusion_matrix(y_train_b, model_ens.predict(ensemble_X)))
    print("this is the model performance on the test data\n")
    y_pred_test = model.predict_proba(X_test_b)[:, 1]
    X_test_br = X_test_b/X_test_b[:, 13].reshape(-1, 1)
    y_pred_test_b = modelb.predict_proba(X_test_br)[:, 1]
    ensemble_X_test = np.hstack((X_test_b, X_test_br,
                                 y_pred_test.reshape(-1, 1),
                                 y_pred_test_b.reshape(-1, 1)))
    view_classification_report(model_ens, ensemble_X_test, y_test_b)
    print(confusion_matrix(y_test_b, model_ens.predict(ensemble_X_test)))
    columns = \
        list(df.columns)+[column+'_rate' for
                          column in df.columns] + \
        ['pred_model_1', 'pred_model_2']
    ensdf = pd.DataFrame(ensemble_X, columns=columns)
    view_feature_importances(ensdf, model_ens)
    print('evaluating the model on the new kind of spam')
    newX = np.vstack((cyX_test, ceX_test))
    newXbr = newX/newX[:, 13].reshape(-1, 1)
    newy = np.hstack((cyy_test, cey_test))
    newy_pred = model.predict(newX)
    newy_pred_b = modelb.predict(newXbr)
    newXens = np.hstack((newX, newXbr,
                         newy_pred.reshape(-1, 1),
                         newy_pred_b.reshape(-1, 1)))
    print(confusion_matrix(newy, model_ens.predict(newXens)))
    # print('fitting to all and writing to pkl')
    # y_all = np.hstack((y_train_b, y_test_b))
    # behavior_X = np.vstack((X_train_bw, X_test_b))
    # behavior_rate_X = np.vstack((X_train_bwr, X_test_br))
    # ensemble_X = np.vstack((ensemble_X, ensemble_X_test))
    # model.fit(behavior_X, y_all)
    # modelb.fit(behavior_rate_X, y_all)
    # model_ens.fit(ensemble_X, y_all)
    # write_model_to_pkl(model, 'account_history_rf_v2')
    # write_model_to_pkl(modelb, 'behavior_rate_rf_v2')
    # write_model_to_pkl(model_ens, 'ensemble_rf_v2')
