# BotBoosted
An App that Measures and Explores the Fabrication of a Queried Twitter Topic

## Abstract

Content is fraudulent or fabricated if it is created by a non-human/non-genuine
account. These accounts include spam-bots, fake-accounts, spammers,
and otherwise --> with the intention of "inflating" or "fabricating" the level
of popularity of a certain topic, so as to mislead people.

Topics whose popularity are inflated can be a Political Candidate, a brand,
a celebrity, or an ideology. For these examples, someone will have something
to gain - at the expense of the truth.

The goal of this project/prototype is to give people a tool to understand
how much of a topic one searches on Twitter is genuine or otherwise, and
what the real and fabricated conversation is about.

This prototype has two main parts:
1. A classifier to determine whether tweet's user account is genuine/not
2. A topic modeller to break down the conversation into understandable
   "chunks" - for the easier interpretation of the user

DISCLAIMER: this app uses Twitter's FREE Rest API in searching for tweets.
The Rest API is able to give users free access to a sample of tweets that are
NOT statistically significant. As a result, no inferences can be made about
the percentage of fraudulence/fabrication of a queried topic through this
prototype. This prototype simply demonstrates how this might work SHOULD
one have access to a statistically significant sample of data.

The backbone of this app is made in python, whose modules are as follows:

1. load_train_data.py
    - These are a series of helper functions that extract the raw information
      from the training data csv files that were put together by
      Stefano Cresci, Roberto Di Pietro, Marinella Petrocchi,
      Angelo Spognardi, and Maurizio Tesconi for their paper
      "Fame for sale: efficient detection of fake Twitter followers."
      http://mib.projects.iit.cnr.it/dataset.html is the link to their data.
      The main orientation of this script is to compile the different csv
      files that this research team put together into one single and properly
      labeled csv.
2. load_test_data.py
    - These helper functions load data from a mongo database that has two
      collections inside it: topictweets and timeline tweets. "topictweets"
      are tweets taken via the tweepy Twitter API wrapper in python. These
      tweets were identified by searching about a topic (i.e. "Hillary
      Clinton Email" or "Donald Trump Sexual Assault" which were hot topics
      prior to the Nov 8 2016 election). "timelinetweets" are the tweets
      taken from the timelines of the users who were identified as users
      who tweeted about the topic (usernames taken from timeline tweets). 200
      tweets were downloaded per user for the data to demonstrate the
      classifier and topic modeler's performance on, but 40 is sufficient.
      The csv file that this script generates already incorporates the
      features needed from the topictweets and timelinetweets collections
3. process_loaded_data.py
    - This script takes functions from load_train_data in order to load
      information, process the different features into the specific items
      needed by the classifier, and then combines the user information
      with the tweet information into a single csv file named "training_df.csv"
4. classification_model.py
    - The classification model runs the training_df.csv information through
      a train test split, random undersampling in order to balance the
      classes for the information that will be used to train and tune a model
      and then several evaluation methods on the model that include 5-fold
      cross validation during gridsearch, and evaluating the performance of
      the classifier on different random samples of different split levels
      taken from the model's unseen test data (the class Eval is in
      evaltestcvbs)
5. evaltestcvbs.py
    - This is a helper class whose purpose is to evaluate a model's performance
      with different random samples generated from unseen test data, with the
      purpose of having a better understanding of the model's average
      performance on out of sample prediction.
6. prediction_model.py
    - The objective of this model is to create a dictionary which has the
      twitter user's user_id and the prediction on whether they are
      a fake account or otherwise
7. load_mongo_tweet_data.py
    - The objective of this model is to get the prediction dictionary from
      the prediction model script, get all the tweets from the mongo db,
      and then create a csv file which labels the tweets as fake or not.
      One critical part of this is that it currently requires two passes
      in terms of downloading tweet data:
      a) get topic related tweets
      b) from the users in step a, get 40 tweets from their timeline
      (future plan)
      c) classify the users
      d) label the tweets

      A strong predictor that is more "lightweight" will be built during the
      course of this project in order to be able to SKIP step b so as to
      make things much faster for users
8. information_gain_ratio.py
    - These are a series of helper functions to compute for the information
      gain ratio of continuous variables using Ross Quinlan's approach as
      appled in the decision tree C4.5, to be used in strengthening a
      classifier's performance by using IGR as feature weights, to be used
      in feature selection by determining the information one can gain from
      different features, and to be used in computing for word importance in
      determine how a word can tell which topic a document falls in
9. tweet_text_processor.py
    - These are a series of functions that do the following things:
      a) Tokenize Tweets in a Twitter specific way (convert links into "url"),
         remove usernames, remove hashtags, correct the spelling of words
         (i.e. "goooooooooood" --> "good") for normalization purposes, convert
         emoticons into words (i.e. :) --> "happy"), remove punctuation,
         remove stopwords
      b) Vectorize the Tweet-Tokenized documents into WordCounts or TFIDF for
         the extraction of topics via Latent Dirichlet Allocation or NMF
         respectively
      c) Label each document with their corresponding topic number and then
         compute for word importance using IGR where the label is the topic
      d) Determine most important words/sentences/tweets by multiplying
         the tf-idf with the IGR weight, as a means of determining the
         "exemplary tweets" that make up that topic
