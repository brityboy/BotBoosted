import dill as pickle

if __name__ == "__main__":
    with open('data/test_tweet_scrape.pkl', 'r') as f:
        tweet_list = pickle.load(f)
