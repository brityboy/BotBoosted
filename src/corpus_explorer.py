import pandas as pd
from tweet_text_processor import multiprocess_tokenize_tweet, tfidf_vectorizer
from tweet_text_processor import compute_for_word_importance
from tweet_text_processor import get_most_important_tweets_and_words_per_topic
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
    print('determining important words...')
    start = time.time()
    word_importance = compute_for_word_importance(tfidf_matrix, topic_label)
    # word_importance = compute_for_word_importance_lightweight(H)
    print("word importance computations took: ", time.time() - start)
    print('fetching important tweets...')
    start = time.time()
    tweet_dict = get_most_important_tweets_and_words_per_topic(tfidf, H, W,
                                                               tfidf_matrix,
                                                               topic_label,
                                                               word_importance,
                                                               documents,
                                                               verbose=True)
    # print("extracted {} topics: "
    #       .format(pnmf.topic_count), time.time() - start)
    #
    # print('plotting topics regular...')
    # start = time.time()
    # color_list, mds = visualize_topics(H)
    # print("plotted topics: ", time.time() - start)
    #
    # print('plotting tweets within topics regular...')
    # for topic_number, color in zip(np.unique(topic_label), color_list):
    #     start = time.time()
    #     visualize_tweets(W, topic_number, color)
    #     print("plotted tweets: ", time.time() - start)
    #     plt.show()
    mds = MDS(n_jobs=-1)
    hflat = mds.fit_transform(H)
    xs, ys = hflat[:, 0], hflat[:, 1]
    cluster_names = tweet_dict['top_words']
    titles = tweet_dict['exemplary_tweet']
    clusdf = pd.DataFrame(dict(x=xs, y=ys, label=range(hflat.shape[0]), title=titles.values()))
    groups = clusdf.groupby('label')
    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')
    ax.legend(loc='best', numpoints=1)  #show legend with only 1 point

    #add label in x,y position with the label as the film title
    for i in range(len(clusdf)):
        ax.text(clusdf.ix[i]['x'], clusdf.ix[i]['y'], clusdf.ix[i]['title'], size=12)



    plt.show() #show the plot

    #uncomment the below to save the plot if need be
    #plt.savefig('clusters_small_noaxes.png', dpi=200)
