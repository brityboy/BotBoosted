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
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from corpus_explorer import plot_topics, plot_all_tweets
import matplotlib.pyplot as plt


def blockify_tweet(tweet):
    '''
    INPUT
         - tweet as a string in a single line
    OUTPUT
         - a blockified tweet (such that different words are in multiple lines)
    Returns a bockified tweet as a string
    '''
    tweet = tweet.replace('\n', '')
    word_count = len(tweet.split())
    line_length = int(np.sqrt(word_count))
    lines = split_list(tweet.split(), line_length)
    return '\n'.join([' '.join(line) for line in lines])


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
    model = RandomForestClassifier(n_jobs=-1, n_estimators=100)
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
         - tweet_dict: dictionary that has the ff keys:
            a)
    Returns the most important tweets per topic by getting the average tfidf
    of the words in the sentence
    '''
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
    pnmf = ParetoNMF(noise_pct=.20, step=1, pnmf_verbose=verbose)
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


def extract_tweets_from_dataframe_for_barplots(df, verbose=False):
    '''
    INPUT
         - df - dataframe
    OUTPUT
         - plots

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
    pnmf = ParetoNMF(noise_pct=.20, step=2, pnmf_verbose=verbose)
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
    tweet_dict = get_important_tweets_and_words_for_barplot(tfidf, H, W,
                                                            tfidf_matrix,
                                                            topic_label,
                                                            word_importance,
                                                            df,
                                                            verbose=verbose)
    if verbose:
        print("fetching took: ", time.time() - start)
    rf_df = compute_real_and_fake_tweets_within_each_topics(topic_label, df)
    make_stacked_barplot(rf_df, tweet_dict)
    del W
    del H
    del tfidf
    del pnmf
    del tweet_dict


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


def compute_real_and_fake_tweets_within_each_topics(topic_label, df):
    '''
    INPUT
         - topic_label - an array that has the soft cluster topic label
         for each tweets
         - df - which has the ff columns: user_id, tweet, pred
    OUTPUT
         - dataframe
    Returns a dataframe with four columns: label, real, fake, and total where
    the values beneath real, fake, and total are the number of tweets in those
    topics groups
    '''
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


def process_real_and_fake_tweets_w_plots(df, verbose=False):
    '''
    INPUT
         - dataframe - must have the screen_name, the text, and the
         pred value for a user so that it can be processed for
         real and fake tweet exploration
         - verbose: set this to true for it to print the output
    OUTPUT
         - plots a stacked bar plot that shows the different main topics,
         as well as the number of fake and real tweets within each topic

    Returns none
    '''
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
    extract_tweets_from_dataframe_for_barplots(df, verbose=verbose)


def get_important_tweets_and_words_for_barplot(tfidf, H, W, tfidf_matrix,
                                               topic_label,
                                               word_importance, df,
                                               verbose=False,
                                               detailed=False):
    '''
    INPUT
         - tfidf: this is the tfidf object
         - H: matrix, this is the topic matrix from NMF
         - tfidf_matrix: this is the tfidf matrix
         - topic_label: this is a list that has the topic label for each doc
         - df: this dataframe has all the tweets
         - df: this is the tweet content from the filtered dataframe
         - verbose: to have the function print out its contnets
         - detailed: to have the function compute for sentence importance using
         the tfidf values and not just the W matrix values
    OUTPUT
         - tweet_dict: dictionary that has the ff keys:
            a)
    Returns the most important tweets per topic by getting the average tfidf
    of the words in the sentence
    '''
    tweet_dict = defaultdict(dict)
    bag_of_words = np.array(map(unidecode, tfidf.get_feature_names()))
    topic_label = np.array(topic_label)
    ntweets = topic_label.shape[0]
    sparse_tfidf = sparse.csr_matrix(tfidf_matrix)
    sentimportance = sparse_tfidf.dot(word_importance)
    tweetarray = df.text.values
    for i, unique_topic in enumerate(np.unique(topic_label)):
        subset_pred = df.pred.values[topic_label == unique_topic]
        subset_tweet_array = tweetarray[topic_label == unique_topic]
        subset_sent_importance = sentimportance[topic_label == unique_topic]
        nsubtweets = subset_sent_importance.shape[0]
        exemplary_real_tweet = \
            subset_tweet_array[np.argsort
                               (subset_sent_importance)
                               [::-1]][subset_pred == 0][0]
        exemplary_fake_tweet = \
            subset_tweet_array[np.argsort
                               (subset_sent_importance)
                               [::-1]][subset_pred == 1][0]
        tweet_dict['exemplary_real_tweet'][i] = \
            blockify_tweet(exemplary_real_tweet)
        tweet_dict['exemplary_fake_tweet'][i] = \
            blockify_tweet(exemplary_fake_tweet)
        top_words = \
            bag_of_words[np.argsort(word_importance*H[i])[::-1]][:5]
        tweet_dict['top_words'][i] = ', '.join(top_words)
        subset_pct = round(float(nsubtweets)/ntweets*100, 2)
        tweet_dict['topic_size_pct'][i] = subset_pct
        tweet_dict['topic_size_n'][i] = nsubtweets
        tweet_dict['tweet_subset_sentimportance'][i] = subset_sent_importance
        tweet_dict['topic_tweets'][i] = subset_tweet_array
        if verbose:
            print('\n')
            print('topic #{}'.format(i+1))
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
    '''
    INPUT
         - rf_df: columns are label, fake, real, and the total number of tweets
         - tweet_dict: dictionary that contains important information about the
         extracted tweets
    OUTPUT
         - list
    Returns a list that has the stacked top words which will pertain to the
    topic being explained
    '''
    x_ticks = []
    labels = rf_df.label.values
    for label in labels:
        x_ticks.append('\n'.join(tweet_dict['top_words'][label].split(', ')))
    return x_ticks


def make_stacked_barplot(rf_df, tweet_dict):
    '''
    INPUT
         - rf_df: columns are label, fake, real, and the total number of tweets
         - tweet_dict: dictionary that contains important information about the
         extracted tweets
    OUTPUT
         - plots a stacked barplot
    Returns none
    '''
    x_ticks = make_xtick_labels_with_top_words(rf_df, tweet_dict)
    total_fake = np.sum(rf_df.fake.values)
    total_real = np.sum(rf_df.real.values)
    N = len(rf_df.label.values)
    fake_tweets = rf_df.fake.values
    real_tweets = rf_df.real.values
    ind = np.arange(N)
    width = 0.35
    p1 = plt.bar(ind, fake_tweets, width, color='.4')
    p2 = plt.bar(ind, real_tweets, width, color='y', bottom=fake_tweets)
    plt.ylabel('Count of Tweets')
    plt.suptitle('Breakdown of Tweets by Topic, and by Real/Fake', fontsize=14,
                 fontweight='bold')
    plt.subplots_adjust(top=0.85)
    plt.title('There are {} fake tweets and {} real tweets'.format(total_fake,
                                                                   total_real))
    plt.xticks(ind + width/2., x_ticks)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend((p2[0], p1[0]), ('Real', 'Fake'), fontsize='large')
    plt.grid('off')
    plt.tight_layout()
    plt.show()


def make_stacked_barplot_percentage(rf_df, tweet_dict):
    '''
    INPUT
         - rf_df: columns are label, fake, real, and the total number of tweets
         - tweet_dict: dictionary that contains important information about the
         extracted tweets
    OUTPUT
         - plots a stacked barplot
    Returns none
    '''
    x_ticks = make_xtick_labels_with_top_words(rf_df, tweet_dict)
    rf_df['fake_pct'] = (rf_df.fake/rf_df.total)*100
    rf_df['real_pct'] = (rf_df.real/rf_df.total)*100
    total_tweets = df.shape[0]
    total_fake = round(np.sum(rf_df.fake.values)/float(total_tweets), 2)*100
    total_real = round(np.sum(rf_df.real.values)/float(total_tweets), 2)*100
    N = len(rf_df.label.values)
    fake_tweets = rf_df.fake_pct.values
    real_tweets = rf_df.real_pct.values
    ind = np.arange(N)
    width = 0.35
    p1 = plt.bar(ind, fake_tweets, width, color='.4')
    p2 = plt.bar(ind, real_tweets, width, color='y', bottom=fake_tweets)
    plt.ylabel('Percent of Tweets in a Topic')
    plt.suptitle('Percentage of FAKE/REAL Tweets in Each Topic', fontsize=14,
                 fontweight='bold')
    plt.subplots_adjust(top=0.85)
    title_string = '{} percent of tweets are Fake, {} percent are Real'
    plt.title(title_string.format(total_fake, total_real))
    plt.xticks(ind + width/2., x_ticks)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend((p2[0], p1[0]), ('Real', 'Fake'), fontsize='large')
    plt.grid('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verbose = True
    df = pd.read_csv('data/clinton_predicted_tweets_v2.csv')
    # df = pd.read_csv('data/trump_predicted_tweets_v2.csv')
    # process_real_and_fake_tweets(df, verbose=True)
    # tweet_dict = extract_tweets_from_dataframe(df, verbose=True)
    # process_real_and_fake_tweets_w_plots(df, verbose=False)
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
    pnmf = ParetoNMF(noise_pct=.20, step=2, pnmf_verbose=True)
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
    tweet_dict = get_important_tweets_and_words_for_barplot(tfidf, H, W,
                                                            tfidf_matrix,
                                                            topic_label,
                                                            word_importance,
                                                            df,
                                                            verbose=True,
                                                            detailed=False)
    if verbose:
        print("fetching took: ", time.time() - start)
    rf_df = compute_real_and_fake_tweets_within_each_topics(topic_label, df)
    make_stacked_barplot(rf_df, tweet_dict)
    make_stacked_barplot_percentage(rf_df, tweet_dict)
