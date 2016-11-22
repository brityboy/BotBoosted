import numpy as np
import pandas as pd
import twokenize as tw
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from unidecode import unidecode
import multiprocessing as mp
import time
from paretonmf import ParetoNMF
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
# from corpus_explorer import visualize_topics, visualize_tweets
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB


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
            for char in strings:
                word = word.replace(char, letter * 2)
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
        elif re.search('\d+', token):
            token_list.append('_num_')
        elif token[0] == '@':
            token_list.append('_user_')
        elif mentions:
            token_list.append('_user_')
        elif token == 'RT':
            pass
        elif token == 'Retweeted':
            pass
        elif type(token) == int:
            token_list.append('_num_')
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
         - list of tokenized tweets where words that occur
         n_words times or less
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


def count_vectorizer(documents):
    '''
    INPUT
         - list of documents
    OUTPUT
         - tfidf: text vectorizer object
         - tfidf_matrix: sparse matrix of word counts

    Processes the documents corpus using a tfidf vectorizer
    '''
    documents = replace_infrequent_words_with_tkn(documents, 4)
    countvec = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    tf_matrix = countvec.fit_transform(documents)
    return countvec, tf_matrix


def compute_for_word_importance(tfidf_matrix, topic_label):
    '''
    INPUT
         - tfidf_matrix - sparse matrix, the tfidf matrix
           of the tokenized tweets
         - topic_label - the topic label assigned
    OUTPUT

    Returns the importance of each word as its feature importance from a
    random forest
    '''
    sparse_tfidf = sparse.csr_matrix(tfidf_matrix)
    model = RandomForestClassifier(n_jobs=-1)
    model.fit(sparse_tfidf, topic_label)
    return model.feature_importances_


def compute_for_word_log_prob(tf_matrix, topic_label):
    '''
    INPUT
         - tfidf_matrix - sparse matrix, the tfidf matrix
           of the tokenized tweets
         - topic_label - the topic label assigned
    OUTPUT

    Returns the importance of each word as its feature importance from a
    random forest
    '''
    mb = MultinomialNB()
    mb.fit(tf_matrix, topic_label)
    return np.exp(mb.coef_+1)


def compute_for_word_importance_lightweight(H):
    '''
    INPUT
         - H topic matrix
    OUTPUT

    Returns the importance of each word as its feature importance from a
    random forest
    '''
    model = RandomForestClassifier(n_jobs=-1)
    model.fit(H, range(H.shape[0]))
    return model.feature_importances_


def get_most_important_tweets_and_words_per_topic(tfidf, H, W, tfidf_matrix,
                                                  topic_label,
                                                  word_importance, documents,
                                                  verbose=False,
                                                  detailed=False):
    '''
    INPUT
         - tfidf: this is the tfidf object
         - H: matrix, this is the topic matrix from NMF
         - tfidf_matrix: this is the tfidf matrix
         - topic_label: this is a list that has the topic label for each doc
         - df: this dataframe has all the tweets
         - documents: this is the list of documents in the df that were
        filtered already
         - verbose: to have the function print out its contnets
         - detailed: to have the function compute for sentence importance using
         the tfidf values and not just the W matrix values
    OUTPUT
         - tweet_dict: dictionary that has two keys:
            a)
    Returns the most important tweets per topic by getting the average tfidf
    of the words in the sentence
    '''
    tweet_dict = defaultdict(dict)
    bag_of_words = np.array(map(unidecode, tfidf.get_feature_names()))
    topic_label = np.array(topic_label)
    ntweets = topic_label.shape[0]
    # the lines below run the computations on the tfidf_matrix.todense(),
    # which while MORE ACCURATE, is a very costly operation
    # (suited for large machines, uncomment the lines below to access these)
    if detailed:
        # sentimportance = \
        #     tfidf_matrix.todense() * word_importance.reshape(-1, 1)
        sparse_tfidf = sparse.csr_matrix(tfidf_matrix)
        sentimportance = sparse_tfidf.dot(word_importance)
        # wordcount = np.sum(sparse_tfidf > 0, axis=1)
        # wordcount = np.apply_along_axis(lambda x: np.sum(x > 0), axis=1,
        #                                 arr=tfidf_matrix.todense())
        # avg_sent_imp = sentimportance/wordcount.reshape(-1, 1)
        # avg_sent_imp = np.asarray(avg_sent_imp).flatten()
        # avg_sent_imp = sentimportance/np.asarray(wordcount).T[0]
        avg_sent_imp = np.array(sentimportance)
    else:
        # the operations below are suboptimal
        # but will work on smaller instances
        # because they rely on the decomposed W matrix
        avg_sent_imp = np.mean(W, axis=1)
    tweetarray = np.array(documents)
    for i, unique_topic in enumerate(np.unique(topic_label)):
        subset_tweet_array = tweetarray[topic_label == unique_topic]
        subset_sent_importance = avg_sent_imp[topic_label == unique_topic]
        nsubtweets = subset_sent_importance.shape[0]
        exemplary_tweet = subset_tweet_array[np.argmax(subset_sent_importance)]
        tweet_dict['exemplary_tweet'][i] = exemplary_tweet
        top_ten_words = \
            bag_of_words[np.argsort(word_importance*H[i])[::-1]][:5]
        tweet_dict['top_words'][i] = top_ten_words
        subset_pct = round(float(nsubtweets)/ntweets*100, 2)
        if verbose:
            print('\n')
            print('topic #{}'.format(i+1))
            print('this is the exemplary tweet from this topic')
            print(exemplary_tweet)
            # print('these are 5 unique example tweets from this topic')
            # print(subset_tweet_array[np.argsort(subset_sent_importance)[::-1]])[:5]
            print('\n')
            print('these are the top words from this topic')
            print(top_ten_words)
            print('{} percent of tweets are in this topic'.format(subset_pct))
    return tweet_dict


def extract_tweets_from_dataframe(df, verbose=False):
    '''
    INPUT
         - df - dataframe
    OUTPUT
         - prints the top tweets from a dataframe given the topic-count

    Return nothing
    '''
    df['length'] = df.text.apply(len)
    df = df.query('length > 1')
    if verbose:
        print('tokenizing tweets...')
        start = time.time()
    documents = [document for document in
                 df.text.values if type(document) == str]
    tokenized_tweets = multiprocess_tokenize_tweet(documents)
    if verbose:
        print("tokenizing the tweets took: ", time.time() - start)
        print('creating the tfidf_matrix...')
        start = time.time()
    tfidf, tfidf_matrix = tfidf_vectorizer(tokenized_tweets)
    if verbose:
        print("vectorizing took: ", time.time() - start)
        print('extracting topics...')
        start = time.time()
    pnmf = ParetoNMF(noise_pct=.20, step=1, pnmf_verbose=True)
    pnmf.evaluate(tfidf_matrix)
    W = pnmf.nmf.transform(tfidf_matrix)
    H = pnmf.nmf.components_
    topic_label = np.apply_along_axis(func1d=np.argmax,
                                      axis=1, arr=W)
    if verbose:
        print("extracted {} topics: "
              .format(pnmf.topic_count), time.time() - start)
        print('determining important words...')
        start = time.time()
    word_importance = compute_for_word_importance(tfidf_matrix, topic_label)
    # word_importance = compute_for_word_importance_lightweight(H)
    if verbose:
        print("word importance computations took: ", time.time() - start)
        print('fetching important tweets...')
        start = time.time()
    tweet_dict = get_most_important_tweets_and_words_per_topic(tfidf, H, W,
                                                               tfidf_matrix,
                                                               topic_label,
                                                               word_importance,
                                                               documents,
                                                               verbose=verbose)
    if verbose:
        print("fetching took: ", time.time() - start)
    del W
    del H
    del tfidf
    del pnmf
    return tweet_dict


def process_real_and_fake_tweets(df, verbose=False):
    '''
    INPUT
         - dataframe - must have the screen_name, the text, and the
         pred value for a user so that it can be processed for
         real and fake tweet exploration
         - verbose: set this to true for it to print the output
    OUTPUT
         -

    Returns none
    '''
    if verbose:
        print('we are going to process {} tweets'.format(df.shape[0]))
    fakedf = df.query('pred == 1')
    realdf = df.query('pred == 0')
    if verbose:
        print('there are {} fake tweets in this query'.format(fakedf.shape[0]))
        print('there are {} real tweets in this query'.format(realdf.shape[0]))
    if fakedf.shape[0] > 0:
        if verbose:
            print('processing the fake tweets')
        extract_tweets_from_dataframe(fakedf, verbose=verbose)
    del fakedf
    if realdf.shape[0] > 0:
        if verbose:
            print('processing the real tweets')
        extract_tweets_from_dataframe(realdf, verbose=verbose)


if __name__ == "__main__":
    df = pd.read_csv('data/clinton_predicted_tweets_v2.csv')
    # df = pd.read_csv('data/trumptweets.csv')
    # process_real_and_fake_tweets(df, verbose=True)
    extract_tweets_from_dataframe(df, verbose=True)
