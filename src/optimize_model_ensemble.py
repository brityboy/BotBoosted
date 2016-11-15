from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import dill as pickle
from sklearn.cross_validation import train_test_split
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from classification_model import *


def plot_roc_curve(model, X, y, modelname):
    '''
    INPUT
         - model - a pre-fit model
         - X - 2d array of x values
         - y - 1d array of targets
         - modelname - name for the label of the plot
    OUTPUT
         - plots the roc curve of a model
    Return none
    '''
    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:, 1])
    plt.plot(fpr, tpr, label=modelname)


if __name__ == "__main__":
    with open('models/tuned_random_forest_model.pkl') as rf:
        rf_model = pickle.load(rf)
    with open('models/tuned_svm_sigmoid_model.pkl') as rbf_svc:
        rbfsvc_model = pickle.load(rbf_svc)
    with open('models/vanilla_gaussian_nb_model.pkl') as gnb:
        gnb_model = pickle.load(gnb)
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
    plot_roc_curve(rf_model, X_test_b, y_test_b, 'RandomForest')
    plot_roc_curve(rbfsvc_model, X_test_b, y_test_b, 'SVM_sigmoid')
    plot_roc_curve(gnb_model, X_test_b, y_test_b, 'GaussianNB')
    plt.legend(loc='best')
    plt.show()
