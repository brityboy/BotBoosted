import numpy as np
import pandas as pd
import csv
import codecs


def load_data_into_dataframe(filename):
    df = pd.read_csv(path)
    return df


def open_text_file(filename):
    text_list = []
    with open(filename) as f:
        for line in f:
            text_list.append(line.replace('\n', ''))
    return np.array(text_list)

def open_csv_file(filename):
    text_list = []
    with open(filename, 'r') as csvfile:
        # opencsvfile = csv.reader(codecs.open(filename, 'rU', 'utf-16'))
        opencsvfile = csv.reader(x.replace('\0', '').replace('\n', '') for x in csvfile)
            for row in opencsvfile:
                text_list.append(row)
    return text_list

def brute_open_text_file_as_df(filename):
    text_list = []
    with open(filename, 'r') as f:
        for line in f:
            if len(line.split(',')) == 25:
                text_list.append(line.replace('\n', '').split(','))
    columns = text_list[0]
    df = pd.DataFrame(text_list[1:], columns=columns)
    return df


if __name__ == "__main__":
    users = pd.read_csv('data/datasets_full.csv/genuine_accounts.csv/users.csv')
    # tweets = pd.read_csv()
    text_list = open_csv_file('data/datasets_full.csv/genuine_accounts.csv/tweets.csv')
