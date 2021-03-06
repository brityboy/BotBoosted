import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations


'''
these functions are the general functions used by all information gain ratio
functions
'''


def is_categorical(x):
    '''
    INPUT
         - single data point x
    OUTPUT
         - boolean
    returns true if x is categorical else false
    '''
    return isinstance(x, str) or isinstance(x, bool) or isinstance(x, unicode)


def check_if_categorical(attribute, df):
    '''
    INPUT:
         - attribute: the feature inside the dataframe to check
         - df: the DataFrame itself
    OUTPUT:
         - boolean
    Returns True if feature in df is categorical else False
    '''
    check_if_categorical = np.vectorize(is_categorical)
    if np.mean(check_if_categorical(df[attribute].values)) == 1:
        return True
    else:
        return False


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


def information_gain(y, y1, y2, impurity_criterion):
    '''
    INPUT:
        - y: 1d numpy array
        - y1: 1d numpy array (labels for subset 1)
        - y2: 1d numpy array (labels for subset 2)
    OUTPUT:
        - float
    Return the information gain of making the given split.
    '''
    return impurity_criterion(y) - \
        (float(len(y1))/len(y) * impurity_criterion(y1) +
            float(len(y2))/len(y) * impurity_criterion(y2))


'''
these are the helper functions for the continuous version of information gain
ratio
'''


def multiple_information_gain(y, y_list, impurity_criterion):
    '''
    INPUT:
        - y: 1d numpy array
        - y_list: list of y values [y1, y2, y3]
        - impurity_criterion: either gini or entropy
    OUTPUT:
        - float
    Return the information gain of making the given split.
    '''
    aggregate_entropy = 0
    for y_vals in y_list:
        aggregate_entropy += float(len(y_vals))/len(y) * \
            impurity_criterion(y_vals)
    return impurity_criterion(y) - aggregate_entropy


def determine_optimal_continuous_split_values(attribute, df, y):
    '''
    INPUT
         - attribute: str, feature to check
         - df: pandas dataframe of features
         - y: 1d array, target
    OUTPUT
         - max_split: tuple of best values to split on
         - info_gain_array: numpy array of all information gains
         - possible_splits: list of all possible split values
    Returns tuple of split values that optimize information gain (min 1 max 3)
    '''
    attribute_value_array = df[attribute].values
    split_values = np.unique(sorted(attribute_value_array))[:-1]
    possible_splits = list(combinations(split_values, 1))
    max_info_gain = 0
    for split in possible_splits:
        X_list, y_list = make_multiple_split(attribute_value_array, y, split)
        if multiple_information_gain(y, y_list, entropy) > max_info_gain:
            max_info_gain = multiple_information_gain(y, y_list, entropy)
            max_split = split
    return max_split


def determine_optimal_continuous_split_values(attribute, df, y):
    '''
    INPUT
         - attribute: str, feature to check
         - df: pandas dataframe of features
         - y: 1d array, target
    OUTPUT
         - max_split: tuple of best values to split on
    Returns tuple of split values that optimize information gain (min 1 max 3)
    '''
    attribute_value_array = df[attribute].values
    split_values = np.unique(sorted(attribute_value_array))[:-1]
    # possible_splits = list(combinations(split_values, 1))
    max_info_gain = 0
    for split in combinations(split_values, 1):
        X_list, y_list = make_multiple_split(attribute_value_array, y, split)
        if multiple_information_gain(y, y_list, entropy) > max_info_gain:
            max_info_gain = multiple_information_gain(y, y_list, entropy)
            max_split = split
    return max_split


def split_list(doc_list, n_groups):
    '''
    INPUT
         - doc_list - is a list of documents to be split up
         - n_groups - is the number of groups to split the doc_list into
    OUTPUT
         - list
    Returns a list of len n_groups which seeks to evenly split up the original
    list into continuous sub_lists
    '''
    avg = len(doc_list) / float(n_groups)
    split_lists = []
    last = 0.0
    while last < len(doc_list):
        split_lists.append(doc_list[int(last):int(last + avg)])
        last += avg
    return split_lists


def potential_attribute_information_gain_continuous(X_list):
    '''
    INPUT
         - X_list: list of optimally split attribute values
    OUTPUT
         - float
    Returns the potential information gain for a continuous split variable
    using ross quinlan's information gain ratio formula in C4.5
    '''
    potential_information_gain = 0
    n_X = sum([len(subset_of_X) for subset_of_X in X_list])
    for X_values in X_list:
        subset_ratio = float(len(X_values))/n_X
        potential_information_gain += subset_ratio * np.log2(subset_ratio)
    return -1 * potential_information_gain


def make_multiple_split(X, y, split_value):
    '''
    INPUT:
        - X: 2d numpy array
        - y: 1d numpy array
        - split_value: single integers or tuples
    OUTPUT:
        - X1: 2d numpy array (feature matrix for subset 1)
        - X2: 2d numpy array (feature matrix for subset 2)
        - X3: 2d numpy array (feature matrix for subset 3)
        - X4: 2d numpy array (feature matrix for subset 4)
        - y1: 1d numpy array (labels for subset 1)
        - y2: 1d numpy array (labels for subset 2)
        - y3: 1d numpy array (labels for subset 3)
        - y4: 1d numpy array (labels for subset 4)


    Return the multiple subsets of the dataset achieved by the given feature
    and value to split on. --> two lists (one for X, one for y)
    '''
    if len(split_value) == 1:
        split_value = split_value[0]
        X1 = X[X <= split_value]
        y1 = y[X <= split_value]
        X2 = X[X > split_value]
        y2 = y[X > split_value]
        return [X1, X2], [y1, y2]
    if len(split_value) == 2:
        lower, upper = split_value
        X1 = X[X <= lower]
        y1 = y[X <= lower]
        X2 = X[(X > lower) & (X <= upper)]
        y2 = y[(X > lower) & (X <= upper)]
        X3 = X[X > upper]
        y3 = y[X > upper]
        return [X1, X2, X3], [y1, y2, y3]
    if len(split_value) == 3:
        lower, mid, upper = split_value
        X1 = X[X <= lower]
        y1 = y[X <= lower]
        X2 = X[(X > lower) & (X <= mid)]
        y2 = y[(X > lower) & (X <= mid)]
        X3 = X[(X > mid) & (X <= upper)]
        y3 = y[(X > mid) & (X <= upper)]
        X4 = X[X > upper]
        y4 = y[X > upper]
        return [X1, X2, X3, X4], [y1, y2, y3, y4]


def information_gain_ratio_continuous(attribute, df, y):
    '''
    INPUT
         - attribute: str, feature to check
         - df: pandas dataframe of features
         - y: 1d array, target
    OUTPUT
         - float
    Returns the information gain ratio accdg to Quinlan's C4.5
    '''
    max_split = determine_optimal_continuous_split_values(attribute, df, y)
    X_list, y_list = make_multiple_split(df[attribute].values, y, max_split)
    ig = multiple_information_gain(y, y_list, entropy)
    pig = potential_attribute_information_gain_continuous(X_list)
    return ig/pig


'''
these functions below compute for information gain ratio for continuous
variables and work in numpy, thus could potentially be much faster than
the pandas version
'''


def information_gain_ratio_continuous_1d(X, y):
    '''
    INPUT
         - X: continuous feature, 1d array
         - y: 1d array, target
    OUTPUT
         - float
    Returns the information gain ratio accdg to Quinlan's C4.5
    '''
    max_split = determine_optimal_continuous_split_values_1d(X, y)
    X_list, y_list = make_multiple_split(X, y, max_split)
    ig = multiple_information_gain(y, y_list, entropy)
    pig = potential_attribute_information_gain_continuous(X_list)
    return ig/pig


def determine_optimal_continuous_split_values_1d(X, y):
    '''
    INPUT
         - X: continuous feature, 1d array
         - y: 1d array, target
    OUTPUT
         - max_split: tuple of best values to split on
    Returns tuple of split values that optimize information gain (min 1 max 3)
    '''
    attribute_value_array = X
    split_values = np.unique(sorted(attribute_value_array))[:-1]
    max_info_gain = 0
    for split in combinations(split_values, 1):
        X_list, y_list = make_multiple_split(attribute_value_array, y, split)
        if multiple_information_gain(y, y_list, entropy) > max_info_gain:
            max_info_gain = multiple_information_gain(y, y_list, entropy)
            max_split = split
    return max_split


'''
these are the categorical functions that work 100 percent correctly accdg to
ross quinlan's information gain ratio formulas from C4.5
'''


def information_gain_by_attribute_categorical(attribute, df, y):
    '''
    INPUT
         - attribute: string, column in the dataframe that IS categorical
         - df: dataframe of features
         - y: 1d array of targets
    OUTPUT
         - float
    Return the information gain for a specific attribute
    '''
    attribute_value_array = df[attribute].values
    possible_attribute_values = np.unique(attribute_value_array)
    attribute_info_gain = 0
    numerator_values = Counter(attribute_value_array)
    for possible_attribute_value in possible_attribute_values:
        value_info_gain = 0
        subset_of_y_values = \
            y[attribute_value_array == possible_attribute_value]
        y_outcomes = np.unique(subset_of_y_values)
        for y_outcome in y_outcomes:
            y_num_value = len(subset_of_y_values
                              [subset_of_y_values == y_outcome])
            value_info_gain += \
                float(y_num_value)/len(subset_of_y_values) \
                * np.log2(float(y_num_value)/len(subset_of_y_values))
        attribute_info_gain += \
            float(numerator_values[possible_attribute_value])/len(y) * \
            -1 * value_info_gain
    return entropy(y) - attribute_info_gain


def potential_information_by_attribute_categorical(attribute, df, y):
    '''
    INPUT
         - attribute: str, feature to check
         - df: pandas dataframe of features
         - y: 1d array, target
    OUTPUT
         - float
    Returns the potential information gain accdg to Quinlan's C4.5
    '''
    attribute_value_array = df[attribute].values
    possible_attribute_values = np.unique(attribute_value_array)
    potential_information = 0
    for possible_attribute_value in possible_attribute_values:
        subset_of_y = y[attribute_value_array == possible_attribute_value]
        potential_information += \
            (float(len(subset_of_y))/len(y)) \
            * np.log2(float(len(subset_of_y))/len(y))
    return -1 * potential_information


def information_gain_ratio_categorical(attribute, df, y):
    '''
    INPUT
         - attribute: str, feature to check
         - df: pandas dataframe of features
         - y: 1d array, target
    OUTPUT
         - float
    Returns the information gain ratio accdg to Quinlan's C4.5
    '''
    information_gain = \
        information_gain_by_attribute_categorical(attribute, df, y)
    potential_information = \
        potential_information_by_attribute_categorical(attribute, df, y)
    return float(information_gain)/potential_information


'''
this function computes for information gain ratio, checks first if it is
categorical or continuous, and then calls the appropriate functions
currently works for dataframes only
'''


def information_gain_ratio(attribute, df, y):
    '''
    INPUT
         - attribute: str, feature to check
         - df: pandas dataframe of features
         - y: 1d array, target
    OUTPUT
         - float
    Returns the information gain ratio accdg to Quinlan's C4.5, and checks
    if the feature is continuous or categorical so as to appropriately
    compute for the information gain ratio
    '''
    if check_if_categorical(attribute, df):
        return information_gain_ratio_categorical(attribute, df, y)
    else:
        return information_gain_ratio_continuous(attribute, df, y)


'''
these functions load toy data to test the functions on
'''


def load_play_golf():
    '''
    INPUT
         - none
    OUTPUT
         - df
         - X
         - y
    Return the df, X features, y values for the playgold.csv toy dataset
    '''
    df = pd.read_csv('data/playgolf.csv')
    df.columns = [c.lower() for c in df.columns]
    y = df.pop('result')
    y = y.values
    X = df.values
    return df, X, y


def load_labor_negotiations_data():
    '''
    INPUT
         - none
    OUTPUT
         - df
         - X
         - y
    Return the df, X features, y values for the labor-neg.data.txt dataset
    '''
    df = pd.read_csv('data/labor-neg.data.txt', header=None)
    df.columns = ['dur', 'wage1', 'wage2', 'wage3', 'cola', 'hours', 'pension',
                  'stby_pay', 'shift_diff', 'educ_allw', 'holidays',
                  'vacation', 'lngtrm_disabil', 'dntl_ins', 'bereavement',
                  'empl_hpln', 'target']
    y = df.pop('target')
    y = y.values
    X = df.values
    return df, X, y


def load_contraceptive_data():
    '''
    INPUT
         - none
    OUTPUT
         - df
         - X
         - y
    Return the df, X features, y values for the cmc.data.txt dataset
    '''
    df = pd.read_csv('data/cmc.data.txt', header=None)
    df.columns = ['wife_age', 'wife_educ', 'hus_educ', 'num_kids', 'wife_rel',
                  'wife_work_status', 'hus_job', 'living_std',
                  'media_expo', 'label']
    y = df.pop('label')
    y = np.array(y)
    X = df.values
    return df, X, y


if __name__ == "__main__":
    df, X, y = load_play_golf()
    print('information_gain')
    for attribute in df.columns:
        print(attribute,
              information_gain_by_attribute_categorical(attribute, df, y))
    print('')
    print('split_information_gain')
    for attribute in df.columns:
        print(attribute,
              potential_information_by_attribute_categorical(attribute, df, y))
    print('')
    print('information_gain_ratio')
    for attribute in df.columns:
        print(attribute, information_gain_ratio_categorical(attribute, df, y))
    print('\ntest information gain for temperature')
    print(information_gain_ratio_continuous('humidity', df, y))
    print(information_gain_ratio_continuous_1d(df['humidity'].values, y))
