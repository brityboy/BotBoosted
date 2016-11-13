from information_gain_ratio import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from process_loaded_data import *
from sklearn.metrics import classification_report


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


if __name__ == "__main__":
    df = create_processed_dataframe()
    user_id_array = df.pop('id')
    y = df.pop('label')
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    model = RandomForestClassifier()
    model = evaluate_model(model, X_train, y_train)
    view_classification_report(model, X_test, y_test)
