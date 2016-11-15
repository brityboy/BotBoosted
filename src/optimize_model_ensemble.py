from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import dill as pickle
from sklearn.cross_validation import train_test_split
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from classification_model import *
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from evaltestcvbs import EvalTestCVBS as Eval


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


def plot_multiple_roc_curves(model_tuple_list, X_test_b, y_test_b):
    '''
    INPUT
         - model_tuple_list - list of tuples of the models (name, model)
         - X_test_b - 2d array of features
         - y_test_b - 1d array of targets
    OUTPUT
         - plots and shows multiple roc curves

    Returns none
    '''
    for model_tuple in model_tuple_list:
        name, model = model_tuple
        plot_roc_curve(model, X_test_b, y_test_b, name)
    plt.legend(loc='best')
    plt.xlabel('False Negative Rate = 1-Recall')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison of Different Models')
    plt.show()


def load_models_and_model_list():
    '''
    INPUT
         - none
    OUTPUT
         - list of tuples where the tuple has (model_name, model)
         - model_list: list of the different models
         - model_name_list: list of model names for label printing
    Returns the list of model information loaded from the pkl files
    '''
    with open('models/tuned_random_forest_model.pkl') as rf:
        rf_model = pickle.load(rf)
    # with open('models/tuned_svm_sigmoid_model.pkl') as sigmoid:
    #     sigmoidsvc_model = pickle.load(sigmoid)
    with open('models/vanilla_gaussian_nb_model.pkl') as gnb:
        gnb_model = pickle.load(gnb)
    with open('models/tuned_svm_rbf_model.pkl') as rbf:
        rbfsvc_model = pickle.load(rbf)
    with open('models/vanilla_gboostc_model.pkl') as gbc:
        gbc_model = pickle.load(gbc)
    with open('models/tuned_gboostc_model.pkl') as tgbc:
        tgbc_model = pickle.load(tgbc)
    model_list = [rf_model, gnb_model, rbfsvc_model,
                  gbc_model, tgbc_model]
    model_names = ['Tuned Random Forest', 'GaussianNB',
                   'SVC RBF', 'Vanilla Gradient Boosting',
                   'Tuned Gradient Boosting']
    return [(model_name,
             model) for model_name, model in zip(model_names, model_list)]


def retrain_models(model_tuple_list, X_train_b, y_train_b):
    '''
    INPUT
         - model_list: list of preconfigured models
         - X_train_b: 2d array of features
         - y_train_b: 1d array of labels
    OUTPUT
         - model_list: models that are re_fit to the training data

    Returns trained models
    '''
    trained_model_list = []
    for model_tuple in model_tuple_list:
        name, model = model_tuple
        model.fit(X_train_b, y_train_b)
        trained_model_list.append((name, model))
    return trained_model_list


def create_voting_classifier_ensemble(model_tuple_list):
    '''
    INPUT
         - model tuple list: list of model tuples (name, model)
    OUTPUT
         - a fit ensemble

    Return fit voting ensemble
    '''
    ensemble = VotingClassifier(model_tuple_list, voting='soft')
    ensemble.fit(X_train_b, y_train_b)
    return ensemble


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
    model_tuple_list = load_models_and_model_list()
    model_tuple_list = retrain_models(model_tuple_list, X_train_b, y_train_b)
    ensemble = create_voting_classifier_ensemble(model_tuple_list)
    plot_multiple_roc_curves(model_tuple_list+[('ensemble', ensemble)], X_test_b, y_test_b)
    print("this is the model performance on different split ratios\n")
    etcb = Eval(ensemble, .05, .95, .05, 10)
    etcb.evaluate_data(X_test_b, y_test_b)
    etcb.plot_performance()
    ensemble.fit(np.vstack((X_train_b, X_test_b)), np.hstack((y_train_b, y_test_b)))
    write_model_to_pkl(ensemble, 'voting_ensemble')
