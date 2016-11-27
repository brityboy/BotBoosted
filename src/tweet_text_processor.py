import numpy as np
import pandas as pd
import twokenize as tw
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from unidecode import unidecode
import multiprocessing as mp
import time
from paretonmf import ParetoNMF
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from scipy import sparse
from corpus_explorer import plot_topics, plot_all_tweets
import matplotlib.pyplot as plt

"""
These are a series of functions that, at a high level, do the following things:

a) Tokenize Tweets in a Twitter specific way (convert links into "url"),
remove usernames, remove hashtags, correct the spelling of words
(i.e. "goooooooooood" --> "good") for normalization purposes, convert
emoticons into words (i.e. :) --> "happy"), remove punctuation,
remove stopwords

b) Vectorize the Tweet-Tokenized documents into WordCounts or TFIDF for the
extraction of topics via IPNMF

c) Soft cluster each document with their corresponding topic number and then
compute for word importance using a random forest's feature importance where
the features are the tweet's tfidf values and the labels are the soft
clustered topic labels for each tweet

d) Determine most important words/sentences/tweets by multiplying the tf-idf
with the feature importance, as a means of determining the "exemplary tweets"
that make up that topic

e) Create a stacked barplot that shows the distribution of the real and fake
tweets within the different subtopics of the tweet corpus, and a percentage
stacked barplot that shows how much of each subtopic is real and fake

This script handles everything that has already been predicted on, it is the
one that makes the most sense of the predictions by bucketing them into
different topics

Of the different functions here, one function is responsible for
processing the dataframe of predicted tweets into the topics, and then
finally the generated plots. This function is called

"process_real_and_fake_tweets_w_plots"

There is another function here, that instead of barplots, makes scatterplots,
and is currently being worked on and improved to make the plots much
easier to understand, this is called process_real_and_fake_tweets.

The current working and stable version of BotBoosted uses the first
function mentioned, which produces stacked barplots
"""


def blockify_tweet(tweet):
    """
    Args:
        tweet (string): the that holds the content of the tweet
    Returns:
        a blockified tweet (string) such that different words are in multiple
        lines so as to make it easier to read when plotted with matplotlib
    Example:
        string = "this is a very long tweet that is hard to read when plotted"
        blockify_tweet(string)
        >>> "this is a very long
            tweet that is hard
            to read when plotted"
    """
    tweet = tweet.replace('\n', '')
    word_count = len(tweet.split())
    line_length = int(np.sqrt(word_count))
    lines = split_list(tweet.split(), line_length)
    return '\n'.join([' '.join(line) for line in lines])


def fix_the_sequence_of_repeated_characters(word):
    """
    Args:
        word (str): this is a single token being inspected for multiple
        letters
    Returns:
        word (str): this is the cleaned up string where the sequence of
        letters repeating more than a few times have been removed
    Example:
        string = "gooooooooood"
        fix_the_sequence_of_repeated_characters(string)
        >>> "good"
    """
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
    """
    INPUT
        text (str): this is the text string that makes up the entire tweet
    Returns
        text (str): this is the tokenized tweet WHERE the ff things have been
        done which are twitter specific:
         - url's are replaced with "url_"
         - hashtags are replaced with "hash_"
         - url's are replaced with the word 'url'
    """
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
            token_list.append('url_')
        elif token[0] == '#':
            token_list.append('hash_')
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
            token_list.append('num_')
        elif token[0] == '@':
            token_list.append('user_')
        elif mentions:
            token_list.append('user_')
        elif token == 'RT':
            pass
        elif token == 'Retweeted':
            pass
        elif type(token) == int:
            token_list.append('num_')
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
    """
    Args:
        doc_list (list): is a list of documents to be split up
        n_groups (int): is the number of groups to split the doc_list into
    Returns:
        split_lists (list): a list of n_groups which are approximately equal
        in length, as a necessary step prior to multiprocessing
    """
    avg = len(doc_list) / float(n_groups)
    split_lists = []
    last = 0.0
    while last < len(doc_list):
        split_lists.append(doc_list[int(last):int(last + avg)])
        last += avg
    return split_lists


def multiprocess_tokenize_tweet(documents):
    """
    Args:
        documents (list): this is a list of the documents to be tweet tokenized
    Returns:
        tokenized_tweets (list): a list of tokenized tweets
        done with multiprocessing
    """
    n_processes = mp.cpu_count()
    p = mp.Pool(n_processes)
    split_docs = split_list(documents, n_processes)
    tokenized_tweets = p.map(tokenize_tweet_list, split_docs)
    return [item for row in tokenized_tweets for item in row]


def tokenize_tweet_list(split_docs):
    """
    Args:
        split_docs (list): list of tweets to be tokenized
    Returns:
        tokenized_tweets (list): a list of sequentially tokenized tweets
    """
    return [tokenize_tweet(text) for text in split_docs]


def replace_infrequent_words_with_tkn(tokenized_tweets, n_words):
    """
    Args:
        tokenized_tweets (list): list of tokenized tweets that went through
        the tweet tokenizer function
        n_words (int): word count frequency cut off such that if frequency
        is n_words and below, then the word will be replaced with the string
        'tkn_'
    Returns:
        processed_tweets (list): list of tokenized tweets where words that
        occur n_words times or less are replaced with the word 'tkn_'
    """
    processed_tweets = []
    string_tweets = ' '.join(tokenized_tweets)
    word_count_dict = Counter(string_tweets.split())
    infreq_word_dict = \
        {token: freq for (token, freq) in
         word_count_dict.items() if freq <= n_words}
    infreq_words = set(infreq_word_dict.keys())
    for tweet in tokenized_tweets:
        processed_tweets.append(' '.join(['tkn_' if token in infreq_words
                                          else token for token
                                          in tweet.split()]))
    return processed_tweets


def tfidf_vectorizer(documents):
    """
    Args:
        documents (list): list of documents to be vectorized via tfidf
    Returns:
        tfidf (sklearn fit tfidf object): text vectorizer object
        tfidf_matrix (compressed sparse row matrix): csr matrix of
        term frequency and inverse document frequency
    """
    documents = replace_infrequent_words_with_tkn(documents, 4)
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(documents)
    return tfidf, tfidf_matrix


def compute_for_word_importance(tfidf_matrix, topic_label):
    """
    Args:
        tfidf_matrix (compressed sparse row format matrix): tfidf matrix
        of the tokenized tweets
        topic_label (1d numpy array): the topic label assigned
    Returns:
        word_importance (1d numpy array): Returns the importance of each
        word as its feature importance from a random forest
    """
    sparse_tfidf = sparse.csr_matrix(tfidf_matrix)
    model = RandomForestClassifier(n_jobs=-1, n_estimators=100)
    model.fit(sparse_tfidf, topic_label)
    return model.feature_importances_


def get_most_important_tweets_and_words_per_topic(tfidf, H, W, tfidf_matrix,
                                                  topic_label,
                                                  word_importance, documents,
                                                  verbose=False):
    """

    THIS FUNCTION IS DEPRECATED

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
         - tweet_dict: dictionary that has the ff keys:
            a)
    Returns the most important tweets per topic by getting the average tfidf
    of the words in the sentence
    """
    tweet_dict = defaultdict(dict)
    bag_of_words = np.array(map(unidecode, tfidf.get_feature_names()))
    topic_label = np.array(topic_label)
    ntweets = topic_label.shape[0]
    sparse_tfidf = sparse.csr_matrix(tfidf_matrix)
    sentimportance = sparse_tfidf.dot(word_importance)
    tweetarray = np.array(documents)
    for i, unique_topic in enumerate(np.unique(topic_label)):
        subset_tweet_array = tweetarray[topic_label == unique_topic]
        subset_sent_importance = sentimportance[topic_label == unique_topic]
        nsubtweets = subset_sent_importance.shape[0]
        exemplary_tweet = subset_tweet_array[np.argmax(subset_sent_importance)]
        # tweet_dict['exemplary_tweet'][i] = exemplary_tweet
        tweet_dict['exemplary_tweet'][i] = blockify_tweet(exemplary_tweet)
        top_words = \
            bag_of_words[np.argsort(word_importance*H[i])[::-1]][:3]
        tweet_dict['top_words'][i] = ', '.join(top_words)
        subset_pct = round(float(nsubtweets)/ntweets*100, 2)
        tweet_dict['topic_size_pct'][i] = subset_pct
        tweet_dict['topic_size_n'][i] = nsubtweets
        tweet_dict['tweet_subset_sentimportance'][i] = subset_sent_importance
        tweet_dict['topic_tweets'][i] = subset_tweet_array
        if verbose:
            print('\n')
            print('topic #{}'.format(i+1))
            print('this is the exemplary tweet from this topic')
            print(exemplary_tweet)
            print('\n')
            print('these are the top words from this topic')
            print(top_words)
            print('{} percent of tweets are in this topic'.format(subset_pct))
    return tweet_dict


def extract_tweets_from_dataframe(df, verbose=False):
    '''

    THIS FUNCTION IS DEPRECATED

    INPUT
         - df - dataframe
    OUTPUT
         - prints the top tweets from a dataframe given the topic-count
         and plots them on PC1 and PC2 in order to understand the tweets
         inside the topic

    Return nothing
    '''
    df.text = df.text.apply(str)
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
    pnmf = ParetoNMF(noise_pct=.2, step=1, pnmf_verbose=verbose)
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
    plot_topics(H, tweet_dict)
    plot_all_tweets(W, topic_label, tweet_dict)
    del W
    del H
    del tfidf
    del pnmf
    del tweet_dict


def extract_tweets_from_dataframe_for_barplots(df, verbose=False,
                                               searchQuery='Your Topic'):
    """
    Args:
        df (pandas dataframe): dataframe that has the text, the username,
        the predicted value, so that the tweets can be processed into
        subtopics
        verbose (boolean): True if the function will be verbose in its
        reporting of the status of each procedure else False
        searchQuery (string): this is the searched query and its presence
        here is for it to included in the title of the barplot
    Returns:
        nothing, plots the distribution stacked barplot and the
        percentage stacked barplot of the different tweets after running
        the corpus through multiple functions, namely:
        a) process_unique_tweets_through_paretonmf
        b) compute_for_word_importance
        c) get_important_tweets_and_words_for_barplot
        d) compute_real_and_fake_tweets_within_each_topics
        e) make_stacked_barplot
        f) make_stacked_barplot_percentage
    """
    W, H, topic_label, tfidf, tfidf_matrix = \
        process_unique_tweets_through_paretonmf(df, verbose=verbose)
    if verbose:
        print('determining important words...')
        start = time.time()
    word_importance = compute_for_word_importance(tfidf_matrix, topic_label)
    if verbose:
        print("word importance computations took: ", time.time() - start)
        print('fetching important tweets...')
        start = time.time()
    tweet_dict = get_important_tweets_and_words_for_barplot(tfidf, H, W,
                                                            tfidf_matrix,
                                                            topic_label,
                                                            word_importance,
                                                            df,
                                                            verbose=verbose)
    if verbose:
        print("fetching took: ", time.time() - start)
    rf_df = compute_real_and_fake_tweets_within_each_topics(topic_label, df)
    make_stacked_barplot(rf_df, tweet_dict, searchQuery=searchQuery)
    make_stacked_barplot_percentage(rf_df, tweet_dict, searchQuery=searchQuery)
    del W
    del H
    del tfidf
    del tweet_dict


def process_real_and_fake_tweets(df, verbose=False):
    '''

    THIS FUNCTION IS DEPRECATED

    INPUT
         - dataframe - must have the screen_name, the text, and the
         pred value for a user so that it can be processed for
         real and fake tweet exploration
         - verbose: set this to true for it to print the output
    OUTPUT
         -

    Returns none
    '''
    df.text = df.text.apply(str)
    df['length'] = df.text.apply(len)
    df = df.query('length > 1')
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


def compute_real_and_fake_tweets_within_each_topics(topic_label, df):
    """
    Args:
        topic_label (1d numpy array): an array that has the soft cluster
        topic label for each tweets
        df (pandas dataframe): which has the ff columns: user_id, tweet, pred
    Returns:
         rf_df (pandas dataframe): rf_df meaning real fake dataframe, as this
         dataframe has four columns: label, real, fake, and total where
         the values beneath real, fake, and total are the number of tweets in
         those different topic groups, this is a necessary step for the
         bartplot to compute the height for the distribution based
         stacked barplot and the percentage ratios for the percentage
         based barplot
    """
    pred_values = df.pred.values
    unique_topics = np.unique(topic_label)
    total_fake = []
    total_real = []
    rf_df = pd.DataFrame(unique_topics, columns=['label'])
    for topic in unique_topics:
        pred_subset = pred_values[topic_label == topic]
        total_fake.append(np.sum(pred_subset == 1))
        total_real.append(np.sum(pred_subset == 0))
    rf_df['fake'] = total_fake
    rf_df['real'] = total_real
    rf_df['total'] = rf_df.fake + rf_df.real
    return rf_df.sort_values(by='total', ascending=False)


def process_real_and_fake_tweets_w_plots(df, verbose=False,
                                         searchQuery='Your Topic'):
    """
    Args:
        df (pandas dataframe): must have the screen_name, the text, and the
        pred value for a user so that it can be processed for
        real and fake tweet exploration
        verbose (boolean): set this to true for it to print the output
        searchQuery (string): this is an optional item that is used for web
        searched topics so that the query appears in the plot titles
    Returns:
        Returns none, plots a stacked bar plot that shows the different main
        topics, as well as the number of fake and real tweets within each topic
    """
    df.text = df.text.apply(str)
    df['length'] = df.text.apply(len)
    df = df.query('length > 1')
    if verbose:
        print('we are going to process {} tweets'.format(df.shape[0]))
    fake_tweets = np.sum(df.pred.values == 1)
    real_tweets = np.sum(df.pred.values == 0)
    if verbose:
        print('there are {} fake tweets in this query'.format(fake_tweets))
        print('there are {} real tweets in this query'.format(real_tweets))
    extract_tweets_from_dataframe_for_barplots(df, verbose=verbose,
                                               searchQuery=searchQuery)


def get_important_tweets_and_words_for_barplot(tfidf, H, W, tfidf_matrix,
                                               topic_label,
                                               word_importance, df,
                                               verbose=False):
    """
    Args:
        tfidf (fit tfidf object): this is the tfidf object that was already
        fit to the corpus
        H (2d array): this is the topic matrix from NMF
        tfidf_matrix (csr format matrix): this is the tfidf matrix
        topic_label (1d array): this is a 1d array that has the topic label
        for each doc
        df (pandas dataframe): dataframe has the tweet content which includes
        the screen_name, the user_id, the text, and the predicted value
        verbose (boolean): True if to have the function print out the status
        and False otherwise
    Returns:
        tweet_dict (dictionary): dictionary that has the ff keys: value pairs
        a) exemplary_real_tweet:
            keys: topic number, values: string containing the most important
            real tweet
        b) exemplary_fake_tweet:
            keys: topic number, values: string containing the most important
            fake tweet
        c) top_words:
            keys: topic number, values: string containing the most important
            words used in that topic as a string
        d) topic_size_pct:
            keys: topic number, values: float, the size of this topic
            versus all the other topics
        e) topic_size_n:
            keys: topic number, values: the number of tweets in this topic
        f) tweet_subset_sentimportance:
            keys: topic number, values: a list of the tweet importances
            for the tweets in this topic
        g) topic_tweets:
            keys: topic number, values: a list of the tweets in this topic
    """
    tweet_dict = defaultdict(dict)
    bag_of_words = np.array(map(unidecode, tfidf.get_feature_names()))
    topic_label = np.array(topic_label)
    ntweets = topic_label.shape[0]
    sparse_tfidf = sparse.csr_matrix(tfidf_matrix)
    sentimportance = sparse_tfidf.dot(word_importance)
    tweetarray = df.text.values
    for topic in np.unique(topic_label):
        subset_pred = df.pred.values[topic_label == topic]
        subset_tweet_array = tweetarray[topic_label == topic]
        subset_sent_importance = sentimportance[topic_label == topic]
        nsubtweets = subset_sent_importance.shape[0]
        if len(subset_tweet_array[np.argsort
                                  (subset_sent_importance)[::-1]]
                                 [subset_pred == 0]) == 0:
            exemplary_real_tweet = 'no real tweets for this subtopic'
        else:
            exemplary_real_tweet = \
                subset_tweet_array[np.argsort
                                   (subset_sent_importance)
                                   [::-1]][subset_pred == 0][0]
        if len(subset_tweet_array[np.argsort
                                  (subset_sent_importance)[::-1]]
                                 [subset_pred == 1]) == 0:
            exemplary_real_tweet = 'no fake tweets for this subtopic'
        else:
            exemplary_fake_tweet = \
                subset_tweet_array[np.argsort
                                   (subset_sent_importance)
                                   [::-1]][subset_pred == 1][0]
        tweet_dict['exemplary_real_tweet'][topic] = \
            blockify_tweet(exemplary_real_tweet)
        tweet_dict['exemplary_fake_tweet'][topic] = \
            blockify_tweet(exemplary_fake_tweet)
        top_words = \
            bag_of_words[np.argsort(word_importance*H[topic])[::-1]][:5]
        tweet_dict['top_words'][topic] = ', '.join(top_words)
        subset_pct = round(float(nsubtweets)/ntweets*100, 2)
        tweet_dict['topic_size_pct'][topic] = subset_pct
        tweet_dict['topic_size_n'][topic] = nsubtweets
        tweet_dict['tweet_subset_sentimportance'][topic] = \
            subset_sent_importance
        tweet_dict['topic_tweets'][topic] = subset_tweet_array
        if verbose:
            print('\n')
            print('topic #{}'.format(topic+1))
            print('this is the exemplary REAL tweet from this topic')
            print(exemplary_real_tweet)
            print('\n')
            print('this is the exemplary FAKE tweet from this topic')
            print(exemplary_fake_tweet)
            print('\n')
            print('these are the top words from this topic')
            print(top_words)
            print('{} percent of tweets are in this topic'.format(subset_pct))
    return tweet_dict


def make_xtick_labels_with_top_words(rf_df, tweet_dict):
    """
    Args:
        rf_df (pandas dataframe): columns are label, fake, real,
        and the total number of tweets, the values within each column are
        the number of tweets for that topic
        tweet_dict: dictionary that contains important information about the
        extracted tweets
    Returns:
         x_ticks (list): a list that has the stacked top words which will
         pertain to the topic being explained
    """
    x_ticks = []
    labels = rf_df.label.values
    for label in labels:
        x_ticks.append('\n'.join(tweet_dict['top_words'][label].split(', ')))
    return x_ticks


def make_stacked_barplot(rf_df, tweet_dict, searchQuery='Your Topic'):
    """
    Args:
        rf_df (pandas dataframe): columns are label, fake, real,
        and the total number of tweets, the values within each column are
        the number of tweets for that topic
        tweet_dict: dictionary that contains important information about the
        extracted tweets
        searchQuery (str): the query that the userd searched, this is to
        add this to the plot supertitle
    Returns:
        nothing, plots a stacked barplot
    """
    x_ticks = make_xtick_labels_with_top_words(rf_df, tweet_dict)
    total_fake = np.sum(rf_df.fake.values)
    total_real = np.sum(rf_df.real.values)
    N = len(rf_df.label.values)
    fake_tweets = rf_df.fake.values
    real_tweets = rf_df.real.values
    ind = np.arange(N)
    width = 0.35
    p1 = plt.bar(ind, fake_tweets, width, color='.55')
    p2 = plt.bar(ind, real_tweets, width, color='y', bottom=fake_tweets)
    plt.ylabel('Count of Tweets')
    plt.suptitle('Tweets by Topic, by Real/Fake, for: {}'.format(searchQuery),
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.85)
    plt.title('There are {} fake tweets and {} real tweets'.format(total_fake,
                                                                   total_real))
    plt.xticks(ind + width/2., x_ticks)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend((p2[0], p1[0]), ('Real', 'Fake'), fontsize='large')
    plt.grid('off')
    plt.tight_layout()
    plt.show()


def make_stacked_barplot_percentage(rf_df, tweet_dict,
                                    searchQuery='Your Topic'):
    """
    Args:
        rf_df (pandas dataframe): columns are label, fake, real,
        and the total number of tweets, the values within each column are
        the number of tweets for that topic
        tweet_dict: dictionary that contains important information about the
        extracted tweets
        searchQuery (str): the query that the userd searched, this is to
        add this to the plot supertitle
    Returns:
        nothing, plots a stacked barplot
    """
    x_ticks = make_xtick_labels_with_top_words(rf_df, tweet_dict)
    rf_df['fake_pct'] = (rf_df.fake/rf_df.total)*100
    rf_df['real_pct'] = (rf_df.real/rf_df.total)*100
    total_tweets = np.sum(rf_df.total.values)
    total_fake = round(np.sum(rf_df.fake.values)/float(total_tweets), 2)*100
    total_real = round(np.sum(rf_df.real.values)/float(total_tweets), 2)*100
    N = len(rf_df.label.values)
    fake_tweets = rf_df.fake_pct.values
    real_tweets = rf_df.real_pct.values
    ind = np.arange(N)
    width = 0.35
    fig = plt.figure()
    fig.suptitle('Fake:Real Tweets / Topic Percent for {}'.format(searchQuery),
                 fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    title_string = 'On Average, {} Percent are Fake, {} Percent are Real'
    ax.set_title(title_string.format(total_fake, total_real))
    p1 = plt.bar(ind, fake_tweets, width, color='.55')
    p2 = plt.bar(ind, real_tweets, width, color='y', bottom=fake_tweets)
    ax.set_ylabel('Percent of Tweets in a Topic')
    plt.xticks(ind + width/2., x_ticks)
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax.legend((p2[0], p1[0]), ('Real', 'Fake'), fontsize='large')
    ax.grid('off')
    plt.tight_layout()
    rects = ax.patches
    labels = get_fake_and_real_top_tweets(rf_df, tweet_dict)
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height*.9, label,
                ha='center', va='bottom', fontsize=12)
    plt.show()


def get_fake_and_real_top_tweets(rf_df, tweet_dict):
    """
    Args:
        rf_df (pandas dataframe): columns are label, fake, real,
        and the total number of tweets, the values within each column are
        the number of tweets for that topic
        tweet_dict: dictionary that contains important information about the
        extracted tweets
    Returns:
        fake_tweet_list+real_tweet_list (list): that has the fake tweets and
        real tweets in sequence for the plotting of these tweets on top
        of the stacked barplot
    """
    fake_tweet_list = []
    real_tweet_list = []
    for topic in rf_df.label.values:
        fake_tweet_list.append(tweet_dict['exemplary_fake_tweet'][topic])
        real_tweet_list.append(tweet_dict['exemplary_real_tweet'][topic])
    return fake_tweet_list+real_tweet_list


def process_unique_tweets_through_paretonmf(df, verbose=False):
    """
    Processed the unique tweets inside the corpus of tweets such that
    topic extraction models the unique strings rather than modeling the
    corpus that has many identical tweets (due to retweets, etc). After
    determining the number of topics within the unique tweets,
    this function will soft cluster the entire corpus to the topics
    using the nmf object that modeled the unique tweets

    Args:
        df (pandas dataframe): dataframe that has all of the predicted
        tweets for processing (has the user's screen_name, the
        tweet text, and the prediction for whether they are real
        or fake)
        verbose (boolean): True to view the progress inside this function
        else False
    Returns:
        W (2d array): of tweets and the reduced dimension of topics
        H (2d array): of topics and the tokens
        topic_label (1d array): of the different labels for the different
        tweets as document soft clusters
        tfidf (fit tfidf object): the fit object to be used in the next
        processes
        tfidf_matrix (csr format matrix): holding the tfidf values
    """
    if verbose:
        print('tokenizing tweets...')
        start = time.time()
    documents = [document for document
                 in df.text.values if type(document) == str]
    tokenized_tweets = multiprocess_tokenize_tweet(documents)
    unique_tweets = np.unique(tokenized_tweets)
    if verbose:
        print("tokenizing the tweets took: ", time.time() - start)
        print('creating the tfidf_matrix...')
        start = time.time()
    tfidf, tfidf_matrix = tfidf_vectorizer(unique_tweets)
    if verbose:
        print("vectorizing took: ", time.time() - start)
        print('extracting topics...')
        start = time.time()
    pnmf = ParetoNMF(noise_pct='auto', step='auto', pnmf_verbose=verbose)
    pnmf.evaluate(tfidf_matrix)
    if verbose:
        print("extracted {} topics: "
              .format(pnmf.topic_count), time.time() - start)
        print('vectorizing the entire corpus and soft clustering tweets...')
    tfidf_matrix = tfidf.transform(tokenized_tweets)
    W = pnmf.nmf.transform(tfidf_matrix)
    H = pnmf.nmf.components_
    topic_label = np.apply_along_axis(func1d=np.argmax,
                                      axis=1, arr=W)
    if verbose:
        print("vectorizing and soft clustering took: "
              .format(pnmf.topic_count), time.time() - start)
    return W, H, topic_label, tfidf, tfidf_matrix


if __name__ == "__main__":
    df = pd.read_csv('data/clinton_predicted_tweets_v2.csv')
    # df = pd.read_csv('data/trump_predicted_tweets_v2.csv')
    process_real_and_fake_tweets_w_plots(df, verbose=True)
