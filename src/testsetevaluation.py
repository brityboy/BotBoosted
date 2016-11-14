import numpy as np
import pandas as pd
# from classification_model import *
from prediction_model import *
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt


class Eval(object):
    '''
    this is a class that does random sampling from test data to solidify
    one's understanding of a model's performance, and is best used for
    classifiers where the natural distribution of the class distribution
    is unknown

    this class is currently made such that
    1 is the anomaly/fraud to detect
    there are more 1 cases than 0 cases
    the natural distribution (ratio) of 1:0 is unknown and needs to be tested
    '''
    def __init__(self, model, r_min, r_max, r_step, n_jobs):
        '''
        INPUT
             - model: this is an sklearn classification
             model that has already been fit
             - r_min: this is the smallest ratio percentage split to test
             - r_max: this is the largest ratio percentage split to test
             and this SHOULD be lower than the current distribution in the data
             - r_step: this is the size of the step to take in checking
             different ratio percentages between r_min and r_max
             - n_jobs: this is the number of times that a sample will be drawn
             and fit in order to get average classification performance values
        this initializes the class, and this class assumes that the 1 class
        is an anomaly or fraud and it is larger than 0 (currently)
        '''
        self.model = model
        self.r_min = r_min
        self.r_max = r_max
        self.r_step = r_step
        self.n_jobs = n_jobs

    def evaluate_data(self, X_test, y_test):
        '''
        INPUT
             - X_test: 2d array of features
             - y_test: 1d array of corresponding values
        OUTPUT
             - percent_range: 1d array
             - avg_precision_0: list
             - avg_recall_0: list
             - avg_precision_1: list
             - avg_recall_1: list

        Returns the percent range over which the n_jobs evaluation was done,
        and this also has the classification report information in four
        lists for these different percent values over the percent range
        '''
        X_pos = X_test[y_test == 1]
        X_neg = X_test[y_test == 0]
        y_pos = y_test[y_test == 1]
        y_neg = y_test[y_test == 0]
        n_pos_total = y_pos.shape[0]
        n_neg_total = y_neg.shape[0]
        self.percent_range = np.arange(self.r_min, self.r_max, self.r_step)
        self.avg_precision_0 = []
        self.avg_recall_0 = []
        self.avg_precision_1 = []
        self.avg_recall_1 = []
        self.std_precision_0 = []
        self.std_recall_0 = []
        self.std_precision_1 = []
        self.std_recall_1 = []
        print('total of {} prediction jobs'.
              format(self.percent_range.shape[0]*self.n_jobs))
        for percent in self.percent_range:
            print('currently evaluating split at {} percent'.format(percent))
            n_draw = int((n_neg_total*percent)/(1-percent))
            precision_0 = []
            recall_0 = []
            precision_1 = []
            recall_1 = []
            for i in xrange(self.n_jobs):
                # print('currently doing job {} for {} percent'.
                #       format(i+1, percent))
                drawn_index = np.random.choice(n_pos_total, n_draw,
                                               replace=False)
                bootstrapped_index = np.random.choice(n_neg_total, n_neg_total,
                                                      replace=True)
                drawn_X = X_pos[drawn_index]
                test_X = np.vstack((X_neg[bootstrapped_index], drawn_X))
                test_y = np.hstack((y_neg, np.ones(n_draw)))
                y_pred = self.model.predict(test_X)
                precision_0.append(precision_score(test_y, y_pred,
                                                   pos_label=0))
                recall_0.append(recall_score(test_y, y_pred, pos_label=0))
                precision_1.append(precision_score(test_y, y_pred,
                                                   pos_label=1))
                recall_1.append(recall_score(test_y, y_pred, pos_label=1))
            avg_prec_0 = np.mean(precision_0)
            avg_rec_0 = np.mean(recall_0)
            avg_prec_1 = np.mean(precision_1)
            avg_rec_1 = np.mean(recall_1)
            print('the avg precision for class 0 is {}'.format(avg_prec_0))
            print('the avg recall for class 0 is {}'.format(avg_rec_0))
            print('the avg precision for class 1 is {}'.format(avg_prec_1))
            print('the avg recall for class 1 is {}'.format(avg_rec_1))
            self.avg_precision_0.append(avg_prec_0)
            self.avg_recall_0.append(avg_rec_0)
            self.avg_precision_1.append(avg_prec_1)
            self.avg_recall_1.append(avg_rec_1)
            self.std_precision_0.append(np.std(precision_0, ddof=1))
            self.std_recall_0.append(np.std(recall_0, ddof=1))
            self.std_precision_1.append(np.std(precision_1, ddof=1))
            self.std_recall_1.append(np.std(recall_1, ddof=1))

        return self.percent_range, self.avg_precision_0, self.avg_recall_0, \
            self.avg_precision_1, self.avg_recall_1

    def plot_performance(self):
        '''
        INPUT
             - None
        OUTPUT
             - plots the performance of the classifier
        returns none
        '''
        plt.plot(self.percent_range, self.avg_precision_0, label='precision_0')
        plt.plot(self.percent_range, self.avg_recall_0, label='recall_0')
        plt.plot(self.percent_range, self.avg_precision_1, label='precision_1')
        plt.plot(self.percent_range, self.avg_recall_1, label='recall_1')
        plt.legend(loc='best')
        plt.title('Model Precision and Recall at Different Split Percentages')
        plt.xlabel('Split Percentages')
        plt.ylabel('Performance Score')
        plt.show()

# if __name__ == "__main__":
#     model = load_pickled_model('models/vanilla_random_forest_model.pkl')
#     df = pd.read_csv('data/training_df.csv')
#     df.drop('Unnamed: 0', axis=1, inplace=True)
#     user_id_array = df.pop('id')
#     y = df.pop('label')
#     y = y.values
#     X = df.values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
#     etcb = Eval(model, .05, .7, .05, 10)
#     percent_range, avg_precision_0, avg_recall_0, avg_precision_1, \
#         avg_recall_1 = etcb.evaluate_data(X_test, y_test)
#     etcb.plot_performance()
