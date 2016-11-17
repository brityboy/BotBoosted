import numpy as np
import pandas as pd
from classification_model import *


def load_all_training_data():
    '''
    INPUT
         - none
    OUTPUT
         - df
    Returns the training dataframe with the ff things done to it:
    a) irrelevant columns removed
    b) protected accounts dropped
    c) specific column nan's filled with 0
    '''
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
    # df.fillna(-99999, inplace=True)
    return df


def explore_df_contents(df):
    '''
    INPUT
         - df
    OUTPUT
         - prints the head of each dataframe column

    Returns nothing
    '''
    for column in df:
        print('\n')
        print(column)
        print(df[column].head())


def get_most_recent_tweets_per_user():
    '''
    INPUT
         - none
    OUTPUT
         - df

    Returns a dataframe with only one tweet per row which is the MOST recent
    tweet recorded for that user_id
    '''
    tweetdf = pd.read_csv('data/training_tweets.csv')
    user_and_max_date = tweetdf[['user_id',
                                 'timestamp']].groupby('user_id').max()
    tweetdf.timestamp = pd.to_datetime(tweetdf.timestamp)
    index = tweetdf.groupby('user_id').apply(lambda x: np.argmax(x.timestamp))
    tweetdf = tweetdf.loc[index.values]
    tweetdf.reset_index().drop('Unnamed: 0', axis=1, inplace=True)
    tweetdf.drop('Unnamed: 0', axis=1, inplace=True)
    return tweetdf


def load_master_training_df():
    '''
    INPUT
         - none
    OUTPUT
         - df

    Returns dataframe combining most recent tweet info with user info
    notes on the columns:
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
    lang - user specified user language
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

    '''
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
    # df.created_at = pd.to_datetime(df.created_at)
    # df['type_created_at'] = df.created_at.apply(type)
    return df

if __name__ == "__main__":
    df = load_master_training_df()
    df = df[['profile_use_background_image', 'geo_enabled', 'verified',
             'followers_count', 'default_profile_image', 'listed_count',
             'statuses_count', 'friends_count', 'favourites_count',
             'favorite_count', 'num_hashtags', 'num_mentions',
             'retweet_count', 'label_y']]
    df = df.fillna(-999)
    y = df.pop('label_y')
    y = y.values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    X_train_b, y_train_b = balance_classes(RandomUnderSampler(),
                                           X_train, y_train)
    X_test_b, y_test_b = balance_classes(RandomUnderSampler(),
                                         X_test, y_test)
    weights = get_igr_attribute_weights(X_train_b, y_train_b, df)
    X_train_bw = X_train_b * weights
    paramgrid = {'n_estimators': [200],
                 'max_features': ['auto'],
                 'criterion': ['gini', 'entropy'],
                 'min_samples_split': [15, 16, 17, 18, 19, 20, 21, 22, 23],
                 'min_samples_leaf': [5, 6, 7, 8],
                 'max_depth': [12, 13, 14, 15, 16, 17],
                 'bootstrap': [True]}
    model = RandomForestClassifier(n_jobs=-1)
    # model = evaluate_model(model, X_train_bw, y_train_b)
    model, gridsearch = gridsearch(paramgrid, model, X_train_bw, y_train_b)
    print("\nthis is the model performance on the training data\n")
    view_classification_report(model, X_train_b, y_train_b)
    confusion_matrix(y_train_b, model.predict(X_train_b))
    print("this is the model performance on the test data\n")
    view_classification_report(model, X_test_b, y_test_b)
    confusion_matrix(y_test_b, model.predict(X_test_b))
    print("this is the model performance on different split ratios\n")
    etcb = Eval(model, .05, .5, .05, 10)
    etcb.evaluate_data(X_test_b, y_test_b)
    etcb.plot_performance()
    print("\nthese are the model feature importances\n")
    view_feature_importances(df, model)
