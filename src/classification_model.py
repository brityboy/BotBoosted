from information_gain_ratio import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from process_loaded_data import *
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.cluster import DBSCAN
import numpy as np


def evaluate_model(model, X_train, y_train):
    '''
    INPUT
         - model: this is a classification model from sklearn
         - X_train: 2d array of the features
         - y_train: 1d array of the target
    OUTPUT
         - information about the model's accuracy using 10 fold cross validation
         - model: the fit model
    Returns the model
    '''
    print(np.mean(cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1, verbose=10)))
    model.fit(X_train, y_train)
    return model


def balance_classes(sm, X, y):
    '''
    INPUT
         - sm: imblearn oversampling/undersampling method
         - X: 2d array of features
         - y: 1d array of targets
    OUTPUT
         - X (balanced feature set)
         - y (balanced feature set)
    Returns X and y after being fit with the resampling method
    '''
    X, y = sm.fit_sample(X, y)
    return X, y


def cluster_based_oversampling_with_smote(metric, X_train, y_train, sm):
    '''
    INPUT
         - metric: this is the kind of distance formula DBSCAN will use
         - X_train: 2d array of features
         - y_train: 1d array of targets
         - sm: imblearn oversampling/undersampling method
    OUTPUT
         - X_train (stratified and balanced feature set)
         - y_train (stratified and balanced feature set)
    Clusters X_train using dbscan with jaccard similarity, then oversamples
    or undersamples
    '''
    db = DBSCAN(metric=metric)
    db.fit(X_train)
    X_list = []
    y_list = []
    for cluster in np.unique(db.labels_):
        X_samp = X_train[db.labels_ == cluster]
        y_samp = y_train[db.labels_ == cluster]
        X_samp, y_samp = balance_classes(sm, X_train, y_train)
        X_list.append(X_samp)
        y_list.append(y_samp)
    return np.vstack(X_list), np.hstack(y_list)


def view_classification_report(model, X_test, y_test):
    '''
    INPUT
         - model: an sklearn classifier model that has already been fit
         - X_test: 2d array of the features
         - y_test: 1d array of the target
    OUTPUT
         - information on the classifiaction performance of the model
    Returns none
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def check_classifier_with_different_test_subsamples(model, X_test, y_test):
    '''
    INPUT
         - model: classifier model fit with sklearn
         - X_test: 2d array of the features
         - y_test: 1d array of the target
    OUTPUT
         - classification report for different classifier output splits
    returns nothing
    '''
    stepsets = np.arange(start=0.05, stop=1, step=.05)
    pass


if __name__ == "__main__":
    df = create_processed_dataframe()
    user_id_array = df.pop('id')
    y = df.pop('label')
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    model = RandomForestClassifier()
    model = evaluate_model(model, X_train, y_train)
    view_classification_report(model, X_test, y_test)
