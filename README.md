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
      The main orientation of this module is to compile the different csv
      files that this research team put together into one single and properly
      labeled csv.

3. process_loaded_data.py
    - This module takes functions from load_train_data in order to load
      information, process the different features into the specific items
      needed by the classifier, and then combines the user information
      with the tweet information into a single csv file named "training_df.csv"

4. lightweight_classifier.py
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

5. lightweight_predictor.py
    - The objective of this module is to load the random forest ensemble
      in order to create a dataframe that has the major information about
      the predicted tweets. This information includes the user's screen_name,
      the tweet itself, and whether the tweet is real or fake

6. tweet_text_processor.py
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

7. paretonmf.py
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

8. tweet_scraper.py
    - This module is responsible for downloading the tweets into a list of
      json objects that contain the tweet information as well as the basic
      account history for each user

9. tweet_scrape_processor.py
    - This module is responsible for processing each json object into the
      different features necessary for the prediction model module so that
      the random forest ensemble can make predictions on newly downloaded
      tweets, or tweets store in a mongo database, so long as they are
      in the form of json objects in a list

10. main.py
    - This module integrates all of the different modules into two main functions:
      - botboosted - this function allows a user to specify a search, and then
        this will download the tweets, classify them, and visualize them
        as different subtopics, and summarize them with exemplary tweets
      - botboosted_demonstration - this function replicates the previous
        function but works on tweets inside a local mongodb rather than
        tweets that are downloaded through twitter's api

### Borrowed modules

1. emoticons.py
this module uses regex to identify emoticons in a tweet and convert them
into the equivalent word (i.e. happy face is replaced with the string "happy")
2. twokenize.py
this module uses regex to identify relevant characters inside a tweet such as
the ampersand for mentions or the octothorp for hashtags

These modules were both borrowed from:
Aritter. Twitter NLP. (2016). Github repository https://github.com/aritter/twitter_nlp

### Deprecated Modules

These modules were used in the initial version of this project:

1. classification_model.py
this module has the original feature set used by Azab, A., Idrees, A.,
Mahmoud, M., Hefny, H. in their paper "Fake Account Detection in Twitter Based
on Minimum Weighted Feature set.". This was deprecated because a more efficient
method of classifying tweets was developed in the course of this project

2. corpus_explorer.py
this module has the initial visualizations that included plotting the different
topics on PC1 and PC2, as well as the different tweets on PC1 and PC2, but was
deprecated because a more effective visualization was developed that involved
stacked barplots

3. evaltestcvbs.py
this is a class that analyzes the precision and recall of a classifier
across different split percentages in the test set in order to gain a better
understanding of the classifier's performance

4. information_gain_ratio
this is a set of functions that compute for the information gain raio
from Ross Quinlan's C4.5, and was originally intended to be used to determine
word importance, but was replaced by the feature importance attribute
from sklearn's random forest

5. optimize_model_ensemble.py
this module is used to host a set of functions that would work to get the
best weightings for the predictors created by the deprecated classification_model.py
module, and was removed because of the development of the more lightweight
classifier, the random forest ensemble

6. prediction_model.py
this module is used to make predictions using the models made by the
deprecated classification_model.py module

7. load_test_data.py
 this module was used to load class A and class B feature data from a mongodb
 and process them for the predictions that will be made by the deprecated
 classified model module
