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
what the real and fabricated conversations are about.

This prototype has three main parts:
1. A classifier to determine whether a tweet's user account is genuine/not
2. A topic modeler to break down the conversation into understandable
   "chunks" - for the easier interpretation of the user, and so that
   the user can see how much of each subtopic within the query is made
   up of real and fake conversations
3. A corpus summarizer that picks out the most important real and fake
   tweets so that the user can easily understand what is going on

DISCLAIMER: this app uses Twitter's FREE Rest API in searching for tweets.
The Rest API is able to give users free access to a sample of tweets that are
NOT statistically significant. As a result, no inferences can be made about
the percentage of fraudulence/fabrication of a queried topic through this
prototype. This prototype simply demonstrates how this might work SHOULD
one have access to a statistically significant sample of data.

### Main Contributions:

Aside from the app prototype, this project makes two major contributions
1. A lightweight classifier that only uses class A features in determining
   whether an account is real or fake, while being as accurate as classification
   models that required class A and class B features
2. A heuristic lightweight model for determining the number of topics in a
  given corpus named Incremental Pareto NMF (IPNMF).

### Major Parts

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

5. lightweight_classifier.py
    - Previous research in the area of classifying twitter users as real or
      fake has done so by using class A (lightweight) and class B (costlier)
      features. Lightweight features include everything that you can get
      from a single tweet (total tweets, follower, likes, account creation date)
      as these are embedded in the json object that one can get when downloading
      a tweet via twitter's API. Costlier features include a user's tweet history,
      meaning the tweets themselves.

      The contribution to the research community of this lightweight classifier
      is a classification method that relies solely on class A features.
      The approach is as follows:
      a) create features from user's account history (total likes, total tweets,
        total followers, total friends, etc)
      b) create features that express relative volume (total likes divided by
        total number of followers, total tweets divided by total number of friends,
        etc) as it was observed that some accounts have hundreds and thousands
        of tweets but very few people in their network
      c) create features that express behavior rate (total likes per day, total
        tweets per day, total likes per friends per day) as the account creation
        date is available in the json object and it was observed that fake
        accounts do "machine gun" tweeting where they tweet very frequently in
        a small period of time. These set of features was added in order
        to also make the model less naive to new users

      No features took the content or the words of the tweet into account
      (i.e. NLP based prediction) as the premise is that a human is always
      behind the message being artificially propagated. The behavior captured
      by the tweet was taken into account by looking at hashtag usage, mentions,
      whether the tweet was favorited by another person, etc.

      The classification model is a random forest ensemble made up of three
      random forest models.
      1. Random Forest 1 (RF1) takes in account history features and relative
      volume features
      2. Random Forest 2 (RF2) takes in behavior rate features that look at
      account history features per day and relative volume features per day
      3. Random Forest 3 (RF3) takes in the predicted probabilities of Random Forest 1
      and Random Forest 2, along with all of these models features, and then
      makes the final prediction.

      The final Random Forest is able to balance out the work of the previous
      ones by understanding the user patterns along the two major facets:
      account history and account behavior rate.

      The ten fold cross validated accuracy of RF1 is 97%, RF2 has 95%,
      and RF3 has 98%. Previous research using this dataset achieved these kinds
      of scores as well. However, they did so with class A and class B features.
      The contribution of this work is that this kind of performance was attained
      using only class A features.

6. evaltestcvbs.py
    - This is a helper class whose purpose is to evaluate a model's performance
      with different random samples generated from unseen test data, with the
      purpose of having a better understanding of the model's average
      performance on out of sample prediction.

7. lightweight_predictor.py
    - The objective of this script is to load the random forest ensemble
      in order to create a dataframe that has the major information about
      the predicted tweets. This information includes the user's screen_name,
      the tweet itself, and whether the tweet is real or fake

8. tweet_text_processor.py
    - These are a series of functions that, at a high level, do the following things:

      a) Tokenize Tweets in a Twitter specific way (convert links into "url"),
         remove usernames, remove hashtags, correct the spelling of words
         (i.e. "goooooooooood" --> "good") for normalization purposes, convert
         emoticons into words (i.e. :) --> "happy"), remove punctuation,
         remove stopwords
      b) Vectorize the Tweet-Tokenized documents into WordCounts or TFIDF for
         the extraction of topics via IPNMF
      c) Soft cluster each document with their corresponding topic number and then
         compute for word importance using a random forest's feature importance
         where the features are the tweet's tfidf values and the labels
         are the soft clustered topic labels for each tweet
      d) Determine most important words/sentences/tweets by multiplying
         the tf-idf with the feature importance, as a means of determining the
         "exemplary tweets" that make up that topic
      e) Create a stacked barplot that shows the distribution of the real
         and fake tweets within the different subtopics of the tweet corpus,
         and a percentage stacked barplot that shows how much of each subtopic
         is real and fake

9. paretonmf.py
    - This is the class that dynamically determines a heuristic count of the
      number of topics inside a corpus using Incremental Pareto NMF.
      The three major parameters to tune for this heuristic include the ff:
      a) noise_pct - this is the percentage of the total documents
         that IPNMF keeps track of in the tail of the topic
         distribution
      b) start - this is the initial number of topics that IPNMF extracts
         from the corpus
      c) step - this is the size of the step, akin to boosting ensemble methods'
         learning rate, that IPNMF takes in incrementally extracting topics
         from the corpus

10. tweet_scraper.py
    - This script is responsible for downloading the tweets into a list of
      json objects that contain the tweet information as well as the basic
      account history for each user

11. tweet_scrape_processor.py
    - This script is responsible for processing each json object into the
      different features necessary for the prediction model script so that
      the random forest ensemble can make predictions on newly downloaded
      tweets, or tweets store in a mongo database, so long as they are
      in the form of json objects in a list

12. main.py
    - This script integrates all of the different modules into two main functions:
      a) botboosted_v3 - this function allows a user to specify a search, and then
         this will download the tweets, classify them, and visualize them
         as different subtopics, and summarize them with exemplary tweets
      b) botboosted_demonstration_v3 - this function replicates the previous
         function but works on tweets inside a local mongodb rather than
         tweets that are downloaded through twitter's api

### Deprecated Modules

These modules were used in the initial version 
