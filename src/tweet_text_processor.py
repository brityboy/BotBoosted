from __future__ import division
import numpy as np
import pandas as pd
import twokenize as tw
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from itertools import combinations
from dit.divergences import jensen_shannon_divergence
import dit
from dill import pickle
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from information_gain_ratio import *
from unidecode import unidecode
import multiprocessing as mp
import threading
import time
from scipy.spatial.distance import cosine

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
                word = word.replace(string, letter*2)
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
            token_list.append('url')
        elif token[0] == '#':
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
            pass
        elif mentions:
            token_list.append('user_mention')
        elif token == 'RT':
            pass
        elif token == 'Retweeted':
            pass
        elif type(token) == int:
            pass
        elif emoticon:
            token_list.append(tw.emoticons.analyze_tweet(token).lower())
        else:
            token = token.translate(None, string.punctuation)
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


def word_count_vectorizer(documents):
    '''
    INPUT
         - list of documents
    OUTPUT
         - vectorizer: text vectorizer object
         - word_count_matrix: sparse matrix of word counts

    Processes the documents corpus using a word count vectorizer
    '''
    vect = CountVectorizer(stop_words='english')
    word_counts_matrix = vect.fit_transform(documents)
    return vect, word_counts_matrix


def tfidf_vectorizer(documents):
    '''
    INPUT
         - list of documents
    OUTPUT
         - tfidf: text vectorizer object
         - tfidf_matrix: sparse matrix of word counts

    Processes the documents corpus using a tfidf vectorizer
    '''
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)
    return tfidf, tfidf_matrix


def fit_LDA(matrix, n_topics):
    '''
    INPUT
         - matrix: takes in a sparse word count matrix
    OUTPUT
         - topics: matrix
         - topic_label: list
    Returns the topic matrix for the specified number of topics requested
    rows pertain to the different words, the columns pertain to the different
    topics; and a list that has the label for that topic
    '''
    topic_label = []
    lda = LatentDirichletAllocation(n_topics=n_topics, n_jobs=-1)
    topics = lda.fit_transform(matrix)
    for document in topics:
        topic_label.append(np.argmax(document))
    return topics, topic_label


def compute_LDA_inter_topic_JS_distance(topics):
    '''
    INPUT
         - topics: matrix of (documents, topics)
    OUTPUT
         - the list of the inter_topic distance
         THERE IS SOMETHING WRONG WITH THIS BECAUSE IT SHOULD BE THE
         INTER DOCUMENT DISTANCE AND NOT THE INTER TOPIC DISTANCE, FUCK
         (MAYBE NMF INSTEAD FOR ALL PRACTICAL PURPOSES)
    Returns the average jensen shannon divergence between pair-wise topics
    by going through each row of the matrix
    '''
    n_topics = topics.shape[1]
    doc_distance_list = []
    for pair in combinations(range(n_topics), 2):
        item1, item2 = pair
        vector1 = dit.ScalarDistribution(topics[:, item1])
        vector2 = dit.ScalarDistribution(topics[:, item2])
        doc_distance_list.append(jensen_shannon_divergence([vector1, vector2]))
    return doc_distance_list


def fit_nmf(matrix, n_topics):
    '''
    INPUT
         - matrix (tfidf) representation of the documents
         - n_topics: number of topics to extract
    OUTPUT
         - W matrix: these are the documents
         - H matrix: these are the latent topics
         - nmf: this is the fit nmf object
         - topic_label: list of topic labels
    Returns the W matrix, the H matrix, the fit nmf object, and the
    label for each of the topics
    '''
    topic_label = []
    nmf = NMF(n_components=n_topics)
    W = nmf.fit_transform(matrix)
    H = nmf.components_
    for document in W:
        topic_label.append(np.argmax(document))
    return W, H, nmf, topic_label


def get_inter_nmf_topic_distance(H, topic_label, weighted=False):
    '''
    INPUT
         - the H matrix generated by NMF
         - weighted = true/false if the averages are weighted by how many
           documents are in that topic
         - topic_label: list of topic labels to get the weight of each document
    OUTPUT
         - avg_inter_topic_distance: float
         - doc_distance_list: list
    Returns the average inter topic distance by computing for the average
    of the pairwise distances of all the topics AND the list from which
    this average was generated
    '''
    # topic_weight_dict = Counter(topic_label)
    n_topics = H.shape[0]
    H = H/np.sum(H, axis=1).reshape(-1, 1)
    doc_distance_list = []
    for pair in combinations(range(n_topics), 2):
        item1, item2 = pair
        # doc_distance_list.append(cosine(H[item1], H[item2]))
        # doc_distance_list.append(np.linalg.norm(H[item1]-H[item2]))
        vector1 = dit.ScalarDistribution(H[item1])
        vector2 = dit.ScalarDistribution(H[item2])
        doc_distance_list.append(jensen_shannon_divergence([vector1, vector2]))
    avg_inter_topic_distance = np.mean(doc_distance_list)
    return avg_inter_topic_distance, doc_distance_list


def explore_nmf_topic_range(n_min, n_max, matrix):
    '''
    INPUT
         - n_min: the minimum number of topics to check
         - n_max: the maximum number of topics to check
         - matrix: the tfidf matrix
    OUTPUT
         - max_topic_count: int
         - dist_list: list

    Returns the max_topic_count with the best separation of topics
    and the dist_list which has the average inter-topic distance for
    each topic count specified
    '''
    dist_list = []
    topic_range = range(n_min, n_max+1)
    for n_topics in topic_range:
        print('currently building NMF with {}'.format(n_topics))
        W, H, nmf, topic_label = fit_nmf(matrix, n_topics)
        print('computing inter_topic_distance for {}'.format(n_topics))
        avg_itd, _ = get_inter_nmf_topic_distance(H, topic_label)
        dist_list.append(avg_itd)
    max_topic_count = topic_range[np.argmax(np.array(dist_list))]
    return max_topic_count, dist_list, topic_range


def plot_the_max_topic_count(max_topic_count, dist_list, topic_range):
    '''
    INPUT
         - the dist_list from the explore_nmf_topic_range function
    OUTPUT
         - plots the inter topic distance and shows the
    Returns nothing
    '''
    plt.plot(topic_range, dist_list, label='average inter topic distance')
    plt.title('Topic Count to Choose is {}'.format(max_topic_count))
    plt.xlabel('Number of Topics')
    plt.ylabel('Average Inter-Topic Distance')
    plt.axvline(x=max_topic_count, color='k', linestyle='--',
                label='n_topics for max inter_topic_dist')
    plt.legend(loc='best')
    plt.show()


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


def explore_important_words(tfidf, n_words):
    '''
    INPUT
         - tfidf: the tfidf object
         - n_words: n number of words to retrieve per topic
    OUTPUT
         - print the top n_words in each topic

    Returns none
    '''
    vocab = np.array(map(str, tfidf.get_feature_names()))
    for row in H:
        print(vocab[np.argsort(row)[-n_words:]])


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
    print "multiprocessing igr computation: ", time.time() - start
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
    print attribute
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


def get_most_importance_sentences_per_topic(tfidf, matrix, topic_label):


if __name__ == "__main__":
    df = pd.read_csv('data/trumptweets.csv')
    print('tokenizing tweets')
    documents = [document for document in
                 df.text.values if type(document) == str]
    start = time.time()
    tokenized_tweets = multiprocess_tokenize_tweet(documents)
    print "multiprocess tokenizing: ", time.time() - start
    # tokenized_tweets = [tokenize_tweet(document) for document in documents]
    # text = df.text.values[49723]
    # text = documents[32]
    # for i, document in enumerate(documents):
    #     print(i, tokenize_tweet(document))
    # s = "TTTCCGACTTTTTGACTTACGAAAAAA"
    # print(fix_the_sequence_of_repeated_characters(s))
    # print tw.emoticons.analyze_tweet(':)')
    # vectorizer, word_counts_matrix = word_count_vectorizer(documents)
    # topics = fit_LDA(word_counts_matrix, 10)
    # with open('data/lda_sample.pkl', 'w+') as f:
    #     pickle.dump(topics, f)
    print('creating the tfidf_matrix')
    start = time.time()
    tfidf, tfidf_matrix = tfidf_vectorizer(tokenized_tweets)
    print "tfidf vectorizing: ", time.time() - start
    # print('exploring the nmf topic range')
    # start = time.time()
    # max_topic_count, dist_list, \
    #     topic_range = explore_nmf_topic_range(2, 20, tfidf_matrix)
    # print "nmf exploration: ", time.time() - start
    # plot_the_max_topic_count(max_topic_count, dist_list, topic_range)
    W, H, nmf, topic_label = fit_nmf(tfidf_matrix, 17)
