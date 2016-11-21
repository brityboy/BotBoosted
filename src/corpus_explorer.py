import pandas as pd
from tweet_text_processor import multiprocess_tokenize_tweet, tfidf_vectorizer
import time
from paretonmf import ParetoNMF
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.cm as cm
from itertools import cycle


def visualize_topics(H):
    '''
    INPUT
         - H matrix of topics
    OUTPUT
         - a scatter plot of the relative location of the different topics
         from each other in a flattened space using PCA
         - color_list - the list of colors to be used in the next
         visualizations of the tweets
    Returns the color list
    '''
    mds = MDS(n_jobs=-1)
    # pca = PCA(n_components=2)
    # hflat = pca.fit_transform(H)
    hflat = mds.fit_transform(H)
    # colors = cm.rainbow(hflat.shape[0]-1)
    colors = cycle(["r", "b", "g", "c", "m", "y", "k", "w"])
    color_list = []
    for i, row in enumerate(hflat):
        color = next(colors)
        plt.scatter(row[0], row[1],
                    label='topic number {}'.format(i+1), color=color)
        color_list.append(color)
    plt.legend(loc='best')
    plt.show()
    return color_list, mds


def visualize_tweets(W, topic_number, color):
    '''
    INPUT
         - W matrix of observations
         - topic_number - this is the number of the topic to be checked
         - color - this is the color to be used in creating the scatterplot
    OUTPUT
         - a scatter plot of the relative location of the different topics
         from each other in a flattened space using multidimensional scaling
    Returns none
    '''
    # mds = MDS(n_jobs=-1)
    topic_list = np.apply_along_axis(np.argmax, 1, W)
    Wsubset = W[topic_list == topic_number]
    pca = PCA(n_components=2)
    pca = PCA(n_components=2)
    hflat = pca.fit_transform(Wsubset)
    plt.scatter(hflat[:, 0], hflat[:, 1], color=color, alpha=.1)
    plt.title('these are the {} tweets in topic # {}'.format(Wsubset.shape[0],
                                                             topic_number+1))
    # plt.show()

if __name__ == "__main__":
    df = pd.read_csv('data/clinton_predicted_tweets_v2.csv')
    print('we are going to process {} tweets'.format(df.shape[0]))
    fakedf = df.query('pred == 1')
    realdf = df.query('pred == 0')
    print('there are {} fake tweets in this query'.format(fakedf.shape[0]))
    print('there are {} real tweets in this query'.format(realdf.shape[0]))

    # print('tokenizing fake tweets...')
    # documents = [document for document in
    #              fakedf.text.values if type(document) == str]
    # start = time.time()
    # tokenized_tweets = multiprocess_tokenize_tweet(documents)
    # print("tokenizing the tweets took: ", time.time() - start)
    # print('creating the tfidf_matrix...')
    # start = time.time()
    # tfidf, tfidf_matrix = tfidf_vectorizer(tokenized_tweets)
    # print("vectorizing took: ", time.time() - start)
    # print('extracting topics...')
    # start = time.time()
    # pnmf = ParetoNMF(noise_pct=.20, step=1, pnmf_verbose=True)
    # pnmf.evaluate(tfidf_matrix)
    # W = pnmf.nmf.transform(tfidf_matrix)
    # H = pnmf.nmf.components_
    # topic_label = np.apply_along_axis(func1d=np.argmax,
    #                                   axis=1, arr=W)
    # print("extracted {} topics: "
    #       .format(pnmf.topic_count), time.time() - start)
    #
    # print('plotting topics regular...')
    # start = time.time()
    # visualize_topics(H, incremental=False)
    # print("plotted topics: ", time.time() - start)
    # plt.show()
    #
    # print('plotting tweets within topics regular...')
    # for topic_number in np.unique(topic_label):
    #     start = time.time()
    #     visualize_tweets(W, topic_number, incremental=False)
    #     print("plotted tweets: ", time.time() - start)

    print('tokenizing real tweets...')
    documents = [document for document in
                 realdf.text.values if type(document) == str]
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

    print('plotting topics regular...')
    start = time.time()
    color_list, mds = visualize_topics(H)
    print("plotted topics: ", time.time() - start)

    print('plotting tweets within topics regular...')
    for topic_number, color in zip(np.unique(topic_label), color_list):
        start = time.time()
        visualize_tweets(W, topic_number, color)
        print("plotted tweets: ", time.time() - start)
        plt.show()
