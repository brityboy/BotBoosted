# from __future__ import division
import numpy as np
import pandas as pd
import twokenize as tw
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
# from itertools import combinations
# from dit.divergences import jensen_shannon_divergence
# import dit
from dill import pickle
from sklearn.decomposition import NMF
# from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import matplotlib.pyplot as plt
from information_gain_ratio import *
from unidecode import unidecode
import multiprocessing as mp
import time
# from scipy.spatial.distance import cosine
import warnings
import operator as op
from paretonmf import ParetoNMF


def fix_the_sequence_of_repeated_characters(word):
    '''
    INPUT
         - word: this is the word being inspected for multiple letters
    OUTPUT
         - str
    Returns a cleaned up string where the sequence of letters repeating
    more than a few times have been removed
    '''
    word = word.lower()
    letters = set(word)
    for letter in letters:
        strings = re.findall(letter+'{3,}', word)
        strings.sort(key=len, reverse=True)
        if strings:
            for string in strings:
                word = word.replace(string, letter * 2)
    return word


def tokenize_tweet(text):
    '''
    INPUT
         - text string for the tweettes
    OUTPUT
         - list
    Returns a list of the different tokens inside the tweet
    WHERE the ff things have been DONE which are twitter specific
     - hashtags are removed (words beginning with #)
     - mentions (words beginning with @) are removed
     - url's are replaced with the word 'url'
    '''
    token_list = []
    text = text.replace('\n', ' ')
    for token in text.split():
        url = tw.Url_RE.search(token)
        mentions = tw.Entity_RE.search(token)
        time = tw.Time_RE.search(token)
        numnum = tw.NumNum_RE.search(token)
        numcomma = tw.NumberWithCommas_RE.search(token)
        separators = tw.Separators_RE.search(token)
        emoticon = tw.emoticons.Emoticon_RE.search(token)
        if url:
            token_list.append('_url_')
        elif token[0] == '#':
            token_list.append('_hash_')
            pass
        elif time:
            pass
        elif separators:
            pass
        elif numcomma:
            pass
        elif numnum:
            pass
        elif token[0] == '@':
            token_list.append('_user_')
        elif mentions:
            token_list.append('_user_')
        elif token == 'RT':
            pass
        elif token == 'Retweeted':
            pass
        elif type(token) == int:
            token_list.append('_number_')
        elif emoticon:
            token_list.append(tw.emoticons.analyze_tweet(token).lower())
        else:
            replace_punctuation = \
                string.maketrans(string.punctuation,
                                 ' '*len(string.punctuation))
            token = token.translate(replace_punctuation)
            token = fix_the_sequence_of_repeated_characters(token)
            token_list.append(token)
    return ' '.join(token_list)


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


def multiprocess_tokenize_tweet(documents):
    '''
    INPUT
         - documents: this is a list of the documents to be tweet tokenized
    OUTPUT
         - list

    Return a list of tokenized tweets done with multiprocessing
    '''
    n_processes = mp.cpu_count()
    p = mp.Pool(n_processes)
    split_docs = split_list(documents, n_processes)
    tokenized_tweets = p.map(tokenize_tweet_list, split_docs)
    return [item for row in tokenized_tweets for item in row]


def tokenize_tweet_list(split_docs):
    '''
    INPUT
         - split_docs: list of tweets to be tokenized
    OUTPUT
         - list of tokenized tweets

    Returns a list of sequentially tokenized tweets
    '''
    return [tokenize_tweet(text) for text in split_docs]


def replace_infrequent_words_with_tkn(tokenized_tweets, n_words):
    '''
    INPUT
         - tokenized_tweets - list of tokenized tweets that went through
         the tweet tokenizer function
         - n_words - word count frequency cut off such that if frequency
         is n_words and below, then the word will be replaced
    OUTPUT
         - list of tokenized tweets where words that occur n_words times or less
         are replaced with the word '_unk_'
    Returns tokenized_tweets, a list of cleaned up tweets
    '''
    processed_tweets = []
    string_tweets = ' '.join(tokenized_tweets)
    word_count_dict = Counter(string_tweets.split())
    infreq_word_dict = \
        {token: freq for (token, freq) in word_count_dict.items() if freq <= 4}
    infreq_words = set(infreq_word_dict.keys())
    for tweet in tokenized_tweets:
        processed_tweets.append(' '.join(['_tkn_' if token in infreq_words
                                          else token for token
                                          in tweet.split()]))
    return processed_tweets


def tfidf_vectorizer(documents):
    '''
    INPUT
         - list of documents
    OUTPUT
         - tfidf: text vectorizer object
         - tfidf_matrix: sparse matrix of word counts

    Processes the documents corpus using a tfidf vectorizer
    '''
    documents = replace_infrequent_words_with_tkn(documents, 4)
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(documents)
    return tfidf, tfidf_matrix


def compute_for_doc_importance(tfidf, matrix, topic_label):
    '''
    INPUT
         - matrix - the sparse matrix (document, word) information
         - topic_label - the topic into which this document falls
    OUTPUT
         - igr_list - list

    Computes for the information gain ratio of each word with the topic as
    the label given the following procedure, bag_of_words

    this is the sequential method for doing this
    '''
    igr_list = []
    bag_of_words = map(unidecode, tfidf.get_feature_names())
    n_words = len(bag_of_words)
    topic_label = np.array(map(str, topic_label))
    word_df = pd.DataFrame(matrix.todense(), columns=bag_of_words)
    for i, word in enumerate(word_df):
        print('we are on word {} out of {} words'.format(i, n_words))
        igr_list.append(information_gain_ratio_continuous(word, word_df,
                                                          topic_label))
    del word_df
    return igr_list, bag_of_words


def compute_doc_importance_parallel(tfidf, matrix, topic_label):
    '''
    INPUT
         - matrix - the sparse matrix (document, word) information
         - topic_label - the topic into which this document falls
    OUTPUT
         - igr_list - list

    Computes for the information gain ratio of each word with the topic as
    the label given the following procedure, bag_of_words

    This procedure uses multiprocessing and threading in order to
    '''
    igr_list = []
    bag_of_words = map(unidecode, tfidf.vocabulary_.keys())
    n_words = len(bag_of_words)
    topic_label = np.array(map(str, topic_label))
    word_df = pd.DataFrame(matrix.todense(), columns=bag_of_words)
    n_processes = mp.cpu_count()
    pool = mp.Pool(processes=n_processes)
    info_tuple_list = [(word, word_df, topic_label) for word in bag_of_words]
    split_info_tuple_list = split_list(info_tuple_list, n_processes)
    start = time.time()
    results = pool.map(tuple_igr_computation, split_info_tuple_list)
    print("multiprocessing igr computation: ", time.time() - start)
    return igr_list, bag_of_words


def tuple_igr_computation(info_tuple):
    '''
    INPUT
         - info_tuple containing (attribute, dataframe, and y)
    OUTPUT
         - information_gain_ratio: float
    Returns
    '''
    attribute, df, y = info_tuple
    print(attribute)
    try:
        return information_gain_ratio_continuous(attribute, df, y)
    except:
        return attribute


def remove_nan_tweets_from_df(df):
    '''
    INPUT
         - df: pandas df with tweets in the text column
    OUTPUT
         - pandas df

    Returns a dataframe where text rows that are not string are removed
    '''
    df['istext'] = df.text.apply(lambda x: 1 if type(x) == str else 0)
    df = df.query('istext == 1')
    return df


def get_most_important_tweets_and_words_per_topic(tfidf, H, tfidf_matrix,
                                                  topic_label, df):
    '''
    INPUT
         - tfidf: this is the tfidf object
         - H: matrix, this is the topic matrix from NMF
         - tfidf_matrix: this is the tfidf matrix
         - topic_label: this is a list that has the topic label for each doc
         - df: this dataframe has all the tweets
    OUTPUT

    Returns the most important tweets per topic by getting the average tfidf
    of the words in the sentence
    '''
    bag_of_words = np.array(map(unidecode, tfidf.get_feature_names()))
    topic_label = np.array(topic_label)
    ntweets = topic_label.shape[0]
    tfidfsum = np.sum(tfidf_matrix, axis=1)
    wordcount = np.apply_along_axis(lambda x: np.sum(x > 0), axis=1,
                                    arr=tfidf_matrix.todense())
    avg_sent_imp = tfidfsum/wordcount.reshape(-1, 1)
    avg_sent_imp = np.asarray(avg_sent_imp).flatten()
    tweetarray = df.text.values
    for i, unique_topic in enumerate(np.unique(topic_label)):
        subset_tweet_array = tweetarray[topic_label == unique_topic]
        subset_sent_importance = avg_sent_imp[topic_label == unique_topic]
        nsubtweets = subset_sent_importance.shape[0]
        print('\n')
        print('topic #{}'.format(i+1))
        print('this is the exemplary tweet from this topic')
        # print(subset_tweet_array[np.argmax(subset_sent_importance)])
        print('these are 5 unique example tweets from this topic')
        print(np.unique(subset_tweet_array[np.argsort(subset_sent_importance)[::-1]])[:5])
        print('\n')
        print('these are the top words from this topic')
        print(bag_of_words[np.argsort(H[i])[::-1]][:10])
        subset_percent = round(float(nsubtweets)/ntweets*100, 2)
        print('{} percent of tweets are in this topic'.format(subset_percent))
    pass


def extract_tweets_from_dataframe(df):
    '''
    INPUT
         - df - dataframe
    OUTPUT
         - prints the top tweets from a dataframe given the topic-count

    Return nothing
    '''
    print('tokenizing tweets...')
    documents = [document for document in
                 df.text.values if type(document) == str]
    start = time.time()
    tokenized_tweets = multiprocess_tokenize_tweet(documents)
    print("tokenizing the tweets took: ", time.time() - start)
    print('creating the tfidf_matrix...')
    start = time.time()
    tfidf, tfidf_matrix = tfidf_vectorizer(tokenized_tweets)
    print("vectorizing took: ", time.time() - start)
    print('extracting topics...')
    start = time.time()
    pnmf = ParetoNMF(noise_pct=.20, step=1, pnmf_verbose=True)
    pnmf.evaluate(tfidf_matrix)
    W = pnmf.nmf.transform(tfidf_matrix)
    H = pnmf.nmf.components_
    topic_label = np.apply_along_axis(func1d=np.argmax,
                                      axis=1, arr=W)
    print("extracted {} topics: "
          .format(pnmf.topic_count), time.time() - start)
    print('fetching important tweets...')
    start = time.time()
    get_most_important_tweets_and_words_per_topic(tfidf, H, tfidf_matrix,
                                                  topic_label, df)
    print("fetching took: ", time.time() - start)
    del W
    del H
    del tfidf

def process_real_and_fake_tweets(df):
    '''
    INPUT
         - dataframe
    OUTPUT
         - prints out the top tweets for an arbitrary topic number
           tentatively set to half a percent of the sample size

    Returns none
    '''
    print('we are going to process {} tweets'.format(df.shape[0]))
    fakedf = df.query('pred == 1')
    realdf = df.query('pred == 0')
    print('there are {} fake tweets in this query'.format(fakedf.shape[0]))
    print('there are {} real tweets in this query'.format(realdf.shape[0]))
    if fakedf.shape[0] > 0:
        extract_tweets_from_dataframe(fakedf)
    del fakedf
    if realdf.shape[0] > 0:
        extract_tweets_from_dataframe(realdf)


if __name__ == "__main__":
    df = pd.read_csv('data/trumptweets.csv')
    process_real_and_fake_tweets(df)
