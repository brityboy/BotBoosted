from __future__ import division
import numpy as np
import pandas as pd
import twokenize as tw
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from itertools import combinations
from dit.divergences import jensen_shannon_divergence
import dit
from dill import pickle


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
            pass
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


def fit_LDA(matrix, n_topics):
    '''
    INPUT
         - matrix: takes in a sparse word count matrix
    OUTPUT
         - topics: matrix
    Returns the topic matrix for the specified number of topics requested
    rows pertain to the different words, the columns pertain to the different
    topics
    '''
    lda = LatentDirichletAllocation(n_topics=n_topics, n_jobs=-1)
    topics = lda.fit_transform(matrix)
    return topics


def compute_inter_topic_distance(topics):
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
    pairs = list(combinations(range(n_topics), 2))
    doc_distance_list = []
    for pair in pairs:
        item1, item2 = pair
        vector1 = dit.ScalarDistribution(topics[:, item1])
        vector2 = dit.ScalarDistribution(topics[:, item2])
        doc_distance_list.append(jensen_shannon_divergence([vector1, vector2]))
    return doc_distance_list




if __name__ == "__main__":
    df = pd.read_csv('data/clintontweets.csv')
    documents = [tokenize_tweet(document) for document in df.text.values if type(document) == str]
    # text = df.text.values[49723]
    # text = documents[32]
    # for i, document in enumerate(documents):
    #     print(i, tokenize_tweet(document))
    # print(fix_the_sequence_of_repeated_characters("TTTCCGACTTTTTGACTTACGAAAAAA"))
    # print tw.emoticons.analyze_tweet(':)')
    vectorizer, word_counts_matrix = word_count_vectorizer(documents)
    # topics = fit_LDA(word_counts_matrix, 10)
    topic_label = []
    for document in topics:
        topic_label.append(np.argmax(document))
    # with open('data/lda_sample.pkl', 'w+') as f:
    #     pickle.dump(topics, f)
