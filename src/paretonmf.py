import numpy as np
from sklearn.decomposition import NMF


class ParetoNMF(object):
    '''
    this is a class that heuristically determines the number of latent topics
    in a corpus of documents iteratively, founded on the following assumptions
    a) n percent of documents within a corpus is noise
    b) more topics extracted from the corpus using NMF if the rich content
    within a corpus, which makes up 1-n percent of the documents within a
    corpus, are the items being decomposed further into topics rather than
    the n percent of documents which is the noise
    c) if by adding another topic to the document, the number of topics under
    which the 1-n percent of documents fall under stay the same, then no more
    additional topics should be added
    d) while there are several topics within a document, a document is assigned
    to fall under a topic based on the topic under which its loading is
    highest along the W matrix

    the following are the parameters to tune the model's performance

    n - float: this is the percent of noise within the corpus. by default, this
    is set to .2 (so as to follow the 80:20 pareto principle)

    step - int: this is the size of the step between how many topics are
    extracted from NMF at a time, by default this is set to 2, but can be
    tuned to a larger number if it is estimated that there are many topics
    in the corpus

    start - int: this is the number of topics to be initially extracted
    from the corpus using NMF, this is initially set to 2 but can be tuned
    if it is estimated that there are many topics in the corpus
    '''


if __name__ == "__main__":
    df = pd.read_csv('data/trumptweets.csv')
