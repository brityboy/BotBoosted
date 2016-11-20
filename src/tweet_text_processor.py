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
# from information_gain_ratio import *
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
            token_list.append('_url_')
        elif token[0] == '#':
            # token_list.append(token[1:])
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
            # token_list.append('user_mention')
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
            # token = token.translate(None, string.punctuation)
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
        processed_tweets.append(' '.join(['_tkn_' if token in infreq_words else token for token in tweet.split()]))
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


def get_most_importance_tweets_and_words_per_topic(tfidf, H, tfidf_matrix,
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


def get_most_importance_tweets_per_topic(tfidf_matrix,
                                         topic_label, df):
    '''
    INPUT
         - tfidf_matrix: this is the tfidf matrix
         - topic_label: this is a list that has the topic label for each doc
         - df: this dataframe has all the tweets
    OUTPUT

    Returns the most important tweets per topic by getting the average tfidf
    of the words in the sentence
    '''
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
        print(subset_tweet_array[np.argmax(subset_sent_importance)])
        subset_percent = round(float(nsubtweets)/ntweets*100, 2)
        print('{} percent of tweets are in this topic'.format(subset_percent))
    pass


def get_intra_topic_similarity_in_w_matrix(W, metric):
    '''
    INPUT
         - W matrix from NMF
    OUTPUT
         - float: average jensen shannon divergence within topics

    Returns the average intra-topic similarity using jensen shannon divergence
    of a W matrix
    '''
    # W = W/np.sum(W, axis=1).reshape(-1, 1)
    topic_labels = np.argmax(W, axis=1)
    unique_topics = np.unique(topic_labels)
    for unique_topic in unique_topics:
        subset = W[topic_labels == unique_topic]
        total = np.sum(np.tril(pairwise_distances(subset,
                                                  metric=metric,
                                                  n_jobs=-1)))
        average = total/float(ncr(subset.shape[1], 2))
        print(average)


def check_runtime(n_topics, metric):
    '''
    INPUT
         - n_topics
         - metric
    OUTPUT
         - prints out the content and the time for n_topics taken from
         a W matrix extracted via NMF and a distance metrics for pairwise
         computations

    Returns none
    '''
    start = time.time()
    W, H, nmf, topic_label = fit_nmf(tfidf_matrix, n_topics)
    get_intra_topic_similarity_in_w_matrix(W, metric)
    print("distance computation time: ", time.time() - start)


def jsd(x, y):
    '''
    INPUT
         - x: np array distribution
         - y: np array distribution
    OUTPUT
         - float
    Returns the JS Divergence
    taken from # @author: jonathanfriedman
    as seen on http://stats.stackexchange.com/questions/29578/jensen-shannon\
    -divergence-calculation-for-3-prob-distributions-is-this-ok
    '''
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    x = np.array(x)
    y = np.array(y)
    d1 = x*np.log2(2*x/(x+y))
    d2 = y*np.log2(2*y/(x+y))
    d1[np.isnan(d1)] = 0
    d2[np.isnan(d2)] = 0
    d = 0.5*np.sum(d1+d2)
    return d


def ncr(n, r):
    '''
    INPUT
         - n, int
         - r, int
         (these are for n choose r)
    OUTPUT
         - int
    Returns the computation of combinations n choose r
    taken from http://stackoverflow.com/questions/4941753/is-\
    there-a-math-ncr-function-in-python
    '''
    r = min(r, n-r)
    if r == 0:
        return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom


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
    # documents = [document for document in
    #              df.text.values if len(document.split()) > 1]
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
    # W, H, nmf, topic_label = fit_nmf(tfidf_matrix, topic_count)
    # print "extracted {} topics: ".format(topic_count), time.time() - start
    print("extracted {} topics: "
          .format(pnmf.topic_count), time.time() - start)
    print('fetching important tweets...')
    start = time.time()
    # get_most_importance_tweets_per_topic(tfidf_matrix,
    #                                      topic_label, df)
    get_most_importance_tweets_and_words_per_topic(tfidf, H, tfidf_matrix,
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
    # faketopics = int(fakedf.shape[0]*.005)
    # realtopics = int(realdf.shape[0]*.005)
    # extract_tweets_from_dataframe(fakedf,
    #                               20 if faketopics > 20 else faketopics)
    # extract_tweets_from_dataframe(realdf,
    #                               20 if realtopics > 20 else realtopics)
    if fakedf.shape[0] > 0:
        extract_tweets_from_dataframe(fakedf)
    del fakedf
    if realdf.shape[0] > 0:
        extract_tweets_from_dataframe(realdf)


if __name__ == "__main__":
    df = pd.read_csv('data/clintontweets.csv')
    process_real_and_fake_tweets(df)
    # print('exploring the nmf topic range')
    # start = time.time()
    # max_topic_count, dist_list, \
    #     topic_range = explore_nmf_topic_range(2, 20, tfidf_matrix)
    # print "nmf exploration: ", time.time() - start
    # plot_the_max_topic_count(max_topic_count, dist_list, topic_range)
    # get_intra_topic_similarity_in_w_matrix(W, 'euclidean')
