from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import dill as pickle
import pandas as pd
from evaltestcvbs import EvalTestCVBS as Eval
import information_gain_ratio as igr
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC


def evaluate_model(model, X_train, y_train):
    '''
    INPUT
         - model: this is a classification model from sklearn
         - X_train: 2d array of the features
         - y_train: 1d array of the target
    OUTPUT
         - information about the model's accuracy using 10
         fold cross validation
         - model: the fit model
    Returns the model
    '''
    print(np.mean(cross_val_score(model, X_train, y_train,
                                  cv=10, n_jobs=-1, verbose=10)))
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
    print(classification_report(y_test, model.predict(X_test)))


def write_model_to_pkl(model, model_name):
    '''
    INPUT
         - model_name: str, this is the name of the model
         - model: the sklearn classification model that will be saved
    OUTPUT
         - saves the model to a pkl file
    Returns None
    '''
    with open('models/{}_model.pkl'.format(model_name), 'w+') as f:
        pickle.dump(model, f)


def view_feature_importances(df, model):
    '''
    INPUT
         - df: dataframe which has the original data
         - model: this is the sklearn classification model that has
         already been fit (work with tree based models)
    OUTPUT
         - prints the feature importances in descending order
    Returns nothing
    '''
    columns = df.columns
    features = model.feature_importances_
    featimps = []
    for column, feature in zip(columns, features):
        featimps.append([column, feature])
    print(pd.DataFrame(featimps, columns=['Features',
                       'Importances']).sort_values(by='Importances',
                                                   ascending=False))


def gridsearch(paramgrid, model, X_train, y_train):
    '''
    INPUT
         - paramgrid: dictionary of lists containing parmeters and
         hypermarameters
         - X_train: 2d array of features
         - y_train: 1d array of class labels
    OUTPUT
         - best_model: a fit sklearn classifier with the best parameters
         - the gridsearch object

    Performs grid search cross validation and
    returns the best model and the gridsearch object
    '''
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


def get_igr_attribute_weights(X_train_b, y_train_b, df):
    '''
    INPUT
         - X_train_b: 2d array of features from balanced class values
         - y_train b: 1d array of balanced y values
         - df: original dataframe from which data was loaded
    OUTPUT
         - numpy array
    Returns an array of the different attribute weights
    '''
    bdf = pd.DataFrame(X_train_b, columns=df.columns)
    weights = []
    for attribute in bdf.columns:
        weights.append(igr.information_gain_ratio_categorical(attribute,
                                                              bdf,
                                                              y_train_b))
    return np.array(weights)

if __name__ == "__main__":
    df = pd.read_csv('data/training_df.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    user_id_array = df.pop('id')
    y = df.pop('label')
    y = y.values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    X_train_b, y_train_b = balance_classes(RandomUnderSampler(),
                                           X_train, y_train)
    X_test_b, y_test_b = balance_classes(RandomUnderSampler(),
                                         X_test, y_test)
    weights = get_igr_attribute_weights(X_train_b, y_train_b, df)
    X_train_bw = X_train_b * weights
    # paramgrid = {'n_estimators': [1000],
    #              'loss': ['exponential'],
    #              'max_features': ['auto'],
    #              'min_samples_split': [22],
    #              'min_samples_leaf': [5],
    #              'max_depth': [3],
    #              'subsample': [.5]}
    # paramgrid = {'n_estimators': [200],
    #              'max_features': ['auto'],
    #              'criterion': ['gini', 'entropy'],
    #              'min_samples_split': [15, 16, 17, 18, 19, 20, 21, 22, 23],
    #              'min_samples_leaf': [5, 6, 7, 8],
    #              'max_depth': [12, 13, 14, 15, 16, 17],
    #              'bootstrap': [True]}
    # paramgrid = {'kernel': ['rbf'],
    #              'gamma': [.01, 'auto', 1.0, 5.0, 10.0, 11, 12, 13],
    #              'C': [.001, .01, .1, 1, 5]}
    # model = SVC(probability=True)
    model = RandomForestClassifier(n_jobs=-1)
    # model = GradientBoostingClassifier()
    # model, gridsearch = gridsearch(paramgrid, model, X_train_bw, y_train_b)
    model = evaluate_model(model, X_train_bw, y_train_b)
    print("\nthis is the model performance on the training data\n")
    view_classification_report(model, X_train_b, y_train_b)
    confusion_matrix(y_train_b, model.predict(X_train_b))
    print("this is the model performance on the test data\n")
    view_classification_report(model, X_test_b, y_test_b)
    confusion_matrix(y_test_b, model.predict(X_test_b))
    print("this is the model performance on different split ratios\n")
    etcb = Eval(model, .05, .5, .05, 100)
    etcb.evaluate_data(X_test_b, y_test_b)
    etcb.plot_performance()
    # print("\nthese are the model feature importances\n")
    # view_feature_importances(df, model)
    print(model)
    # write_model_to_pkl(model, 'tuned_gboostc')
