import pandas as pd
# from tweet_text_processor import multiprocess_tokenize_tweet,
# tfidf_vectorizer
# from tweet_text_processor import compute_for_word_importance
# from tweet_text_processor import \
# get_most_important_tweets_and_words_per_topic
import time
from paretonmf import ParetoNMF
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from itertools import cycle
from unidecode import unidecode
import mpld3
from mpld3 import plugins


css = """
table
{
  border-collapse: collapse;
}
th
{
  color: #ffffff;
  background-color: #000000;
}
td
{
  background-color: #cccccc;
}
table, th, td
{
  font-family:Arial, Helvetica, sans-serif;
  border: 1px solid black;
  text-align: right;
}
"""


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


def plot_topics_arrays(H, tweet_dict):
    pca = PCA(n_components=2)
    hflat = pca.fit_transform(H)
    xs, ys = hflat[:, 0], hflat[:, 1]
    topic_size = tweet_dict['topic_size_pct']
    cluster_names = tweet_dict['top_words']
    titles = tweet_dict['exemplary_tweet']
    print('\nthis is the dictionary and the cluster label names')
    print(cluster_names)
    labels = range(hflat.shape[0])
    fig, ax = plt.subplots()  # set size
    for label, x, y in zip(labels, xs, ys):
        # print('\n')
        # print('name: ', cluster_names[label])
        # print('x: ', x)
        # print('y: ', y)
        # print('\n')
        ax.plot(x, y, marker='o', linestyle='', ms=topic_size[label],
                label=unidecode(cluster_names[label]),
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off',
                       top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', bottom='off',
                       top='off', labelbottom='off')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.grid('off')
    for i in labels:
        ax.text(xs[i], ys[i], titles[i], size=12)
        plt.tight_layout()
    ax.legend(loc='best', numpoints=1)


def plot_topics(H, tweet_dict):
    '''
    INPUT
         - H topic matrix from NMF
         - tweet_dict - dictionary of tweet information
            which includes:
                the most important tweet
                the percent of tweets that fall into a certain topic
                the sentence important of each tweet under each topic
                the top words
    OUTPUT
         - plots the relative distance of the tweet topics, as well as
         information about the tweets such as the relative topic size,
         the exemplary tweets, and the top words per topic

    Returns nothing
    '''
    pca = PCA(n_components=2)
    hflat = pca.fit_transform(H)
    xs, ys = hflat[:, 0], hflat[:, 1]
    topic_size = tweet_dict['topic_size_pct']
    cluster_names = tweet_dict['top_words']
    titles = tweet_dict['exemplary_tweet']
    clusdf = pd.DataFrame(dict(x=xs, y=ys, label=range(hflat.shape[0])))
    clusdf['title'] = clusdf['label'].apply(lambda label: titles[label])
    fig, ax = plt.subplots(figsize=(17, 9))
    ax.margins(0.03)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*.8, box.height])
    for name, x, y in zip(clusdf.label.values,
                          clusdf.x.values,
                          clusdf.y.values):
        ax.plot(x, y, marker='o', linestyle='',
                ms=topic_size[name],
                label=cluster_names[name])
        ax.set_aspect('auto')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    for i in range(len(clusdf)):
        ax.text(clusdf.ix[i]['x'], clusdf.ix[i]['y'],
                clusdf.ix[i]['title'], size=12)
    lgnd = ax.legend(loc='center left',
                     bbox_to_anchor=(1, 0.5),
                     numpoints=1,
                     title='Top Words Used in the Topics',
                     frameon=False,
                     markerscale=1)
    # for i in range(H.shape[0]):
    #     lgnd.legendHandles[i]._legmarker.set_markersize(12)
    #     lgnd.legendHandles[i]._legmarker.set_markersize(12)
    plt.show()


def plot_all_tweets(W, topic_label, tweet_dict):
    '''
    INPUT
         - W reduced tweet matrix from NMF
         - topic_list - the list of soft clustered topics in which tweets
         are assigned
         - tweet_dict
    OUTPUT
         - plots the relative distance of the tweet topics, as well as
         information about the tweets such as the relative topic size,
         the exemplary tweets, and the top words per topic

    Returns nothing
    '''
    pca = PCA(n_components=2)
    hflat = pca.fit_transform(W)
    xs, ys = hflat[:, 0], hflat[:, 1]
    cluster_names = tweet_dict['top_words']
    titles = tweet_dict['exemplary_tweet']
    tweet_importance = tweet_dict['tweet_subset_sentimportance']
    clusdf = pd.DataFrame(dict(x=xs, y=ys, label=topic_label))
    groups = clusdf.groupby('label')
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.margins(0.03)
    colors = cycle(["r", "b", "g", "c", "m", "y", "k", "w"])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*.8, box.height])
    for name, group in groups:
        # print(cluster_names[name])
        color = next(colors)
        ax.scatter(group.x, group.y, alpha=1, c=color,
                   label=cluster_names[name], s=4000*tweet_importance[name])
        ax.set_aspect('auto')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
    ax.set_ylabel('PC 2')
    ax.set_xlabel('PC 1')
    groups = clusdf.groupby('label').mean().reset_index()
    groups['title'] = groups.label.apply(lambda label: titles[label])
    for i in range(len(groups)):
        ax.text(groups.ix[i]['x'], groups.ix[i]['y'],
                groups.ix[i]['title'], size=12)
    ax.legend(loc='center left',
              bbox_to_anchor=(1, 0.5),
              numpoints=1,
              title='Top Words Used in the Topics',
              frameon=False,
              markerscale=1)
    plt.show()


def plot_all_tweets_draft1(W, topic_label, tweet_dict):
    '''
    INPUT
         - W reduced tweet matrix from NMF
         - topic_list - the list of soft clustered topics in which tweets
         are assigned
         - tweet_dict
    OUTPUT
         - plots the relative distance of the tweet topics, as well as
         information about the tweets such as the relative topic size,
         the exemplary tweets, and the top words per topic

    Returns nothing
    '''
    pca = PCA(n_components=2)
    hflat = pca.fit_transform(W)
    # retrieve x and y arrays for the reduced dimensions
    xs, ys = hflat[:, 0], hflat[:, 1]
    # refer to the top words as the cluster names
    cluster_names = tweet_dict['top_words']
    # refer to the topic titles as the exemplary tweet
    titles = tweet_dict['exemplary_tweet']
    # refer to the sentence importance in the tweet_dict
    tweet_importance = tweet_dict['tweet_subset_sentimportance']
    # create a clusdf object that has all the information
    clusdf = pd.DataFrame(dict(x=xs, y=ys, label=topic_label))
    groups = clusdf.groupby('label')
    # set up plot
    fig, ax = plt.subplots()  # set size
    # fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05)
    # iterate through groups to layer the plot
    colors = cycle(["r", "b", "g", "c", "m", "y", "k", "w"])
    for name, group in groups:
        # print(cluster_names[name])
        color = next(colors)
        ax.scatter(group.x, group.y, alpha=.1, c=color,
                   label=cluster_names[name], s=4000*tweet_importance[name])
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.set_xlabel('PC 1')
        ax.tick_params(
            axis='y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')
        ax.set_ylabel('PC 2')
        # ax.legend(loc='best', numpoints=1)  # show legend with only 1 point
        ax.legend()  # show legend with only 1 point

    # add label in x,y position with the label as the film title
    groups = clusdf.groupby('label').mean().reset_index()
    groups['title'] = groups.label.apply(lambda label: titles[label])
    for i in range(len(groups)):
        ax.text(groups.ix[i]['x'], groups.ix[i]['y'],
                groups.ix[i]['title'], size=12)
    plt.show()  # show the plot


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


def plot_topics_mpld3(H, tweet_dict):
    '''
    INPUT
         - H topic matrix from NMF
         - tweet_dict - dictionary of tweet information
            which includes:
                the most important tweet
                the percent of tweets that fall into a certain topic
                the sentence important of each tweet under each topic
                the top words
    OUTPUT
         - creates a plot using mpl3d

    Returns nothing
    '''
    pca = PCA(n_components=2)
    hflat = pca.fit_transform(H)
    xs, ys = hflat[:, 0], hflat[:, 1]
    topic_size = tweet_dict['topic_size_pct']
    cluster_names = tweet_dict['top_words']
    titles = tweet_dict['exemplary_tweet']
    clusdf = pd.DataFrame(dict(x=xs, y=ys, label=range(hflat.shape[0])))
    clusdf['title'] = clusdf['label'].apply(lambda label: titles[label])
    fig, ax = plt.subplots(figsize=(17, 9))
    ax.margins(0.03)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*.8, box.height])
    for name, x, y in zip(clusdf.label.values,
                          clusdf.x.values,
                          clusdf.y.values):
        points = ax.plot(x, y, marker='o', linestyle='',
                         ms=topic_size[name],
                         label=cluster_names[name])
        ax.set_aspect('auto')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        tooltip = \
            plugins.PointLabelTooltip(points[0],
                                      labels=[clusdf.title.values[name]])
        plugins.connect(fig, tooltip)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    # for i in range(len(clusdf)):
    #     ax.text(clusdf.ix[i]['x'], clusdf.ix[i]['y'],
    #             clusdf.ix[i]['title'], size=12)
    lgnd = ax.legend(loc='center left',
                     bbox_to_anchor=(1, 0.5),
                     numpoints=1,
                     title='Top Words Used in the Topics',
                     frameon=False,
                     markerscale=1)
    for i in range(H.shape[0]):
        lgnd.legendHandles[i]._legmarker.set_markersize(12)
        lgnd.legendHandles[i]._legmarker.set_markersize(12)
    # plt.show()
    mpld3.show()

if __name__ == "__main__":
    df = pd.read_csv('data/clinton_predicted_tweets_v2.csv')
    print('we are going to process {} tweets'.format(df.shape[0]))
    fakedf = df.query('pred == 1')
    realdf = df.query('pred == 0')
    print('there are {} fake tweets in this query'.format(fakedf.shape[0]))
    print('there are {} real tweets in this query'.format(realdf.shape[0]))

    print('tokenizing tweets tweets...')
    documents = [document for document in
                 fakedf.text.values if type(document) == str]
    start = time.time()
    tokenized_tweets = multiprocess_tokenize_tweet(documents)
    print("tokenizing the tweets took: ", time.time() - start)
    print('creating the tfidf_matrix...')
    start = time.time()
    tfidf, tfidf_matrix = tfidf_vectorizer(tokenized_tweets)
    print("vectorizing took: ", time.time() - start)
    print('extracting topics...')
    start = time.time()
    pnmf = ParetoNMF(noise_pct=.20, start=2, step=1, pnmf_verbose=True)
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
    print("word importance computations took: ", time.time() - start)
    print('fetching important tweets...')
    start = time.time()
    tweet_dict = get_most_important_tweets_and_words_per_topic(tfidf, H, W,
                                                               tfidf_matrix,
                                                               topic_label,
                                                               word_importance,
                                                               documents,
                                                               verbose=True)
