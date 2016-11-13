import pandas as pd
import dill as pickle


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


if __name__ == "__main__":
    model = load_pickled_model('models/vanilla_random_forest_model.pkl')
    user_id_array, X = \
        load_processed_csv_for_predictions('data/clintonmillion.csv')
    user_id_df = create_dataframe_with_id_and_predictions(user_id_array, X)
