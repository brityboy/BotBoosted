import pandas as pd
import dill as pickle
from load_test_data import *


def load_pickled_model(filename):
    '''
    INPUT
         - filename: str, path and name of the file
    OUTPUT
         - model: sklearn classifier model, fit already

    Returns the unpickled model
    '''
    with open(filename, 'r') as f:
        model = pickle.load(f)
    f.close()
    return model


def load_processed_csv_for_predictions(filename):
    '''
    INPUT
         - filename: str, path and name of the processed csv file to make
         predictions on
    OUTPUT
         - X: this is a 2d array of the features
         - user_id_array: 1d array of the features

    Returns
    '''
    df = pd.read_csv(filename)
    df.id = df.id.apply(str)
    user_id_array = df.pop('id')
    X = df.values
    return user_id_array, X


def create_dictionary_with_id_and_predictions(model, user_id_array, X,
                                              dict_filename):
    '''
    INPUT
         - model: this is the loaded sklearn model
         - user_id_array: single column dataframe of the user ids
         - X: this is the feature list for prediction
    OUTPUT
         - pickles a dictionary object where the keys are the id and the values
         are the prediction on it

    Returns none
    '''
    y_pred = model.predict(X)
    user_id_array = pd.DataFrame(user_id_array, columns=['id'])
    user_id_array['pred'] = y_pred
    user_id_array.index = user_id_array.id
    user_id_array.drop('id', axis=1, inplace=True)
    pred_dict = user_id_array.to_dict()['pred']
    write_dict_to_pkl(pred_dict, dict_filename)


if __name__ == "__main__":
    model = load_pickled_model('models/voting_ensemble_model.pkl')
    user_id_array, X = \
        load_processed_csv_for_predictions('data/trumpmillion.csv')
    create_dictionary_with_id_and_predictions(model, user_id_array, X,
                                              'trumpmillion_pred')
