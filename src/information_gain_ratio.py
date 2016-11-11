import numpy as np
import pandas as pd
from collections import Counter
import fraud_detection as fd

def entropy(y):
    '''
    INPUT:
        - y: 1d numpy array
    OUTPUT:
        - float

    Return the entropy of the array y.
    '''
    unique = set(y)
    count = Counter(y)
    ent = 0
    for val in unique:
        p = count[val]/float(len(y))
        ent += p * np.log2(p)
    return -1 * ent


def gini(y):
    '''
    INPUT:
        - y: 1d numpy array
    OUTPUT:
        - float

    Return the gini impurity of the array y.
    '''
    unique = np.unique(y)
    count = Counter(y)
    ent = 0
    for val in unique:
        p = count[val]/float(len(y))
        ent += p**2
    return 1 - ent


def make_split(X, y, split_index, split_value):
    '''
    INPUT:
        - X: 2d numpy array
        - y: 1d numpy array
        - split_index: int (index of feature)
        - split_value: int/float/bool/str (value of feature)
    OUTPUT:
        - X1: 2d numpy array (feature matrix for subset 1)
        - y1: 1d numpy array (labels for subset 1)
        - X2: 2d numpy array (feature matrix for subset 2)
        - y2: 1d numpy array (labels for subset 2)

    Return the two subsets of the dataset achieved by the given feature and
    value to split on.

    Call the method like this:
    >>> X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)

    X1, y1 is a subset of the data.
    X2, y2 is the other subset of the data.
    '''
    if type(split_value) != float and type(split_value) != int:
        X1 = X[X[:, split_index] != split_value]
        y1 = y[X[:, split_index] != split_value]
        X2 = X[X[:, split_index] == split_value]
        y2 = y[X[:, split_index] == split_value]
    else:
        X1 = X[X[:, split_index] <= split_value]
        y1 = y[X[:, split_index] <= split_value]
        X2 = X[X[:, split_index] > split_value]
        y2 = y[X[:, split_index] > split_value]
    return X1, y1, X2, y2


def information_gain(y, y1, y2, impurity_criterion):
    '''
    INPUT:
        - y: 1d numpy array
        - y1: 1d numpy array (labels for subset 1)
        - y2: 1d numpy array (labels for subset 2)
    OUTPUT:
        - float

    Return the information gain of making the given split.

    Use self.impurity_criterion(y) rather than calling _entropy or _gini
    directly.
    '''
    return impurity_criterion(y) - (float(len(y1))/len(y) * impurity_criterion(y1) + float(len(y2))/len(y) * impurity_criterion(y2))


def choose_split_index(X, y):
    '''
    INPUT:
        - X: 2d numpy array
        - y: 1d numpy array
    OUTPUT:
        - index: int (index of feature)
        - value: int/float/bool/str (value of feature)
        - splits: (2d array, 1d array, 2d array, 1d array)

    Determine which feature and value to split on. Return the index and
    value of the optimal split along with the split of the dataset.

    Return None, None, None if there is no split which improves information
    gain.

    Call the method like this:
    >>> index, value, splits = self._choose_split_index(X, y)
    >>> X1, y1, X2, y2 = splits
    '''
    info_gain = 0
    split_index = 0
    split_value = 0
    for i in xrange(X.shape[1]):
        for j in np.unique(X[:, i]):
            X1, y1, X2, y2 = make_split(X, y, i, j)
            current_info_gain = information_gain(y, y1, y2, impurity_criterion)
            if current_info_gain >= info_gain:
                info_gain = current_info_gain
                split_index = i
                split_value = j
    if info_gain <= error_threshold:
        print info_gain
        return None, None, None
    else:
        return split_index, split_value, make_split(X, y, split_index, split_value)


def load_contraceptive_data():
    df = pd.read_csv('data/cmc.data.txt', header=None)
    df.columns = ['wife_age', 'wife_educ', 'hus_educ', 'num_kids', 'wife_rel',
                  'wife_work_status', 'hus_job', 'living_std', 'media_expo', 'label']
    y = df.pop('label')
    y = np.array(y)
    X = df.values
    return df, X, y


def information_gain_by_attribute_categorical(attribute, df, y):
    attribute_value_array = df[attribute].values
    possible_attribute_values = np.unique(attribute_value_array)
    attribute_info_gain = 0
    numerator_values = Counter(attribute_value_array)
    for possible_attribute_value in possible_attribute_values:
        value_info_gain = 0
        subset_of_y_values = y[attribute_value_array == possible_attribute_value]
        y_outcomes = np.unique(subset_of_y_values)
        for y_outcome in y_outcomes:
            value_info_gain += float(len(subset_of_y_values[subset_of_y_values == y_outcome]))/len(subset_of_y_values) * np.log2(float(len(subset_of_y_values[subset_of_y_values == y_outcome]))/len(subset_of_y_values))
        value_info_gain = -1 * value_info_gain
        attribute_info_gain += float(numerator_values[possible_attribute_value])/len(y) * value_info_gain
    return entropy(y) - attribute_info_gain
    # return attribute_info_gain


def potential_information_by_attribute_categorical(attribute, df, y):
    attribute_value_array = df[attribute].values
    possible_attribute_values = np.unique(attribute_value_array)
    potential_information = 0
    for possible_attribute_value in possible_attribute_values:
        subset_of_y = y[attribute_value_array == possible_attribute_value]
        potential_information += (float(len(subset_of_y))/len(y)) * np.log2(float(len(subset_of_y))/len(y))
    return -1 * potential_information


def information_gain_ratio_categorical(attribute, df, y):
    information_gain = information_gain_by_attribute_categorical(attribute, df, y)
    potential_information = potential_information_by_attribute_categorical(attribute, df, y)
    return float(information_gain)/potential_information


def information_gain_by_attribute_continuous(attribute, df, y):
    attribute_value_array = df[attribute].values
    attribute_value_array = np.delete(attribute_value_array, 2)
    attribute_info_gain = 0
    numerator_values = Counter(attribute_value_array)
    for possible_attribute_value in possible_attribute_values:
        value_info_gain = 0
        subset_of_y_values = y[attribute_value_array == possible_attribute_value]
        y_outcomes = np.unique(subset_of_y_values)
        for y_outcome in y_outcomes:
            value_info_gain += float(len(subset_of_y_values[subset_of_y_values == y_outcome]))/len(subset_of_y_values) * np.log2(float(len(subset_of_y_values[subset_of_y_values == y_outcome]))/len(subset_of_y_values))
        value_info_gain = -1 * value_info_gain
        attribute_info_gain += float(numerator_values[possible_attribute_value])/len(y) * value_info_gain
    return entropy(y) - attribute_info_gain
    # return attribute_info_gain


def potential_information_by_attribute_continuous(attribute, df, y):
    attribute_value_array = df[attribute].values
    possible_attribute_values = np.unique(attribute_value_array)
    potential_information = 0
    for possible_attribute_value in possible_attribute_values:
        subset_of_y = y[attribute_value_array == possible_attribute_value]
        potential_information += (float(len(subset_of_y))/len(y)) * np.log2(float(len(subset_of_y))/len(y))
    return -1 * potential_information


def information_gain_ratio_continuous(attribute, df, y):
    information_gain = information_gain_by_attribute_continuous(attribute, df, y)
    potential_information = potential_information_by_attribute_continuous(attribute, df, y)
    return float(information_gain)/potential_information


def load_play_golf():
    df = pd.read_csv('data/playgolf.csv')
    df.columns = [c.lower() for c in df.columns]
    y = df.pop('result')
    y = y.values
    X = df.values
    return df, X, y


def load_labor_negotiations_data():
    df = pd.read_csv('data/labor-neg.data.txt', header=None)
    df.columns = ['dur', 'wage1', 'wage2', 'wage3', 'cola', 'hours', 'pension',
                  'stby_pay', 'shift_diff', 'educ_allw', 'holidays',
                  'vacation', 'lngtrm_disabil', 'dntl_ins', 'bereavement',
                  'empl_hpln', 'target']
    y = df.pop('target')
    y = y.values
    X = df.values
    return df, X, y

if __name__ == "__main__":
    # df, X, y = load_contraceptive_data()
    # df, X, y = load_play_golf()
    df = fd.load_data('data/creditcardfraud.zip')
    y = fd.get_y(df, 'label')
    df, X = fd.get_x(df)
    impurity_criterion = entropy
    error_threshold = 0
    print('information_gain')
    for attribute in df.columns:
        print attribute, information_gain_by_attribute_categorical(attribute, df, y)
    print('')
    print('split_information_gain')
    for attribute in df.columns:
        print attribute, potential_information_by_attribute_categorical(attribute, df, y)
    print('')
    print('information_gain_ratio')
    for attribute in df.columns:
        print attribute, information_gain_ratio_categorical(attribute, df, y)
    # index, value, splits = choose_split_index(X, y)
    # X1, y1, X2, y2 = splits
