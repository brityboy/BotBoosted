Brian Balagot
Oct 26, 2016

BotBoosted

A Web App that Analyzes the Fabricated Conversation that Bots Have About a Topic on Twitter

Motivation
About a month ago, an article came out in Philippine news talking about how “social media manipulation” is something that is happening – but it is not something that many people are aware of. The thing is, however, as more and more people become connected to the internet and to Online Social Networks (OSN’s), the likelihood that their views and opinions on matters like brands, values, or politics can be swayed. With this, I became very curious about this world of “fabricated content” created by a few people but propagated by bots or fake accounts.

How much of what we see on Social Media is Fabricated? What is the conversation like in the real sphere of social media? In the fabricated sphere? How does the conversation in the fabricated sphere affect the real sphere and vise versa? These then became the questions that I became more interested in trying to understand – most especially the last one: how does the fabricated conversation affect the real conversation.
	Most of the research in this area of OSN bot detection has dealt with the following things:
    a) Classification algorithms using features of OSN users (OSN behavior, content sentiment, etc) to determine whether they are a real user or a fake account (also known as a Sybil – and these Sybils can be spammers, compromised accounts, fake accounts, bots, or paid trolls).
    b) Graph based algorithms that understand the account profile in order to classify them into one of the different user profiles (spammer, compromised account, fake account, bot, etc)

One area of research that stands to gain from the existing classification research in the field is to take a deeper and exploratory look into the behavior of these bots and fake accounts. Bots exist – what are they saying? Of the total posts about a topic in a given time period, how many area real and how many are fabricated? When do they post/tweet about these things? Does their sentiment agree with the majority or the minority of the real users? How do they interact with real users? Have they penetrated communities of real users already – and if so, to what extent?

The final output of this project is a topic visualizer using information on Twitter (because, compared to Facebook, Twitter is more open to sharing its information through its API). Input a search term into a website and then the user can explore the following things about this topic:
  a) For this topic, how many tweets are made by real people? And how many are possibly by bots? (over a time period x)
  b) What are people saying about this topic? With retweets? Net of retweets
  c) What are possible bots saying about this topic?
  d) How similar/different is the tone/sentiment of the conversation?
  e) For this topic: how do the different bots and users interact with each other?
	The goal is to take this kind of procedure and make it available to a lot of people in order to build awareness about the reality and breadth of fake accounts, the fake conversation made, and the effect of this fake conversation on the real conversation.

I was able to get in touch already with two research teams (one from Italy and one from Greece) and both have given me their training data. I have yet to fully explore the data from the Greece research team, but here is a high level description of the data that I was able to get from the Italian research team:

1) 2 groups - spambots and fake twitter followers
2) approximately 17M tweets from spambots
3) another data set of fake followers and real people:
	TFP (the fake project): 100% humans
	E13 (elections 2013): 100% humans
	INT (intertwitter): 100% fake followers
	FSF (fastfollowerz): 100% fake followers
	TWT (twittertechnology): 100% fake followers
taken from http://mib.projects.iit.cnr.it/dataset.html with their permission

Data Project

I would break down the project into the following steps:
1) Get a dataset on which I can train and validate a classifier (either through a researcher who has already done this or through a manually labeled set)
2) Create the different features necessary from the dataset in order to run the algorithms that have been tried and tested from previous research in this domain
3) Build a script that will:
a) scrape twitter for X tweets about a certain topic
b) classify the different users and report the likelihood of their being fake or not
c) take out clusters of sub-topics from the searched topic, summarize the content with a word cloud, and rate the sentiment of the clusters – for the real and the fake conversation
4) Build a script(s) that will create different kinds of visualizations for the gathered information:
a) a social graph for the subset of the OSN users who mentioned the sub-topic, colored by the probability that they are a real user or not (red if very likely not real, green if very real, orange for in between)
b) a line plot that looks at the frequency of this sub-topic being tweeted over time, to show whether the volume of tweets is from a real or fake user
5) Build a web app to host all of the activities above
6) Tools: Python (Tweepy, Graphlab, Numpy, Scikit-Learn, Pandas), MongoDB

References

(n.d.). (2014, Oct 5). #SmartFREEInternet: Anatomy of a black ops campaign on Twitter. Rappler. 	Retrieved from http://www.rappler.com/technology/social-media/71115-anatomy-of-a-twitter-	black-ops-campaign

(n.d.). (2015, Nov 25). Building a Twitter Big Data Pipeline with Python, Cassandra & Redis. Thinking 	Machines Stories. Retrieved from http://stories.thinkingmachin.es/twitter-listener/

(n.d.). (n.d.) How to spot a Twitter bot: 10 easy ways to recognize fake accounts. Twitter Bot 	Detection: News and updates of the best fake twitter followers detection service. Retrieved from 	http://botornot.co/blog/index.php/how-to-spot-a-twitter-bot-10-easy-ways-to-recognize-fake-	accounts/

(n.d.). (2016, Apr 7). ‘Sana ma-rape ka’: Netizens bully anti-Duterte voter. Rappler. Retrieved from 	http://www.rappler.com/move-ph/128602-viral-duterte-supporters-harass-netizen

Andersen, K. Flock. (2016). Github repository https://github.com/kaylaandersen/twitter-graph

Aritter. Twitter NLP. (2016). Github repository https://github.com/aritter/twitter_nlp

Azab, A., Idrees, A., Mahmoud, M., Hefny, H. (Nov 1, 2016). Fake Account Detection in Twitter Based on Minimum Weighted Feature set. World Academy of Science, Engineering and Technology International Journal of Computer, Electrical, Automation, Control and Information Engineering Vol:10, No:1, 2016. Retrieved from http://waset.org/publications/10003176/fake-account-detection-in-twitter-based-on-minimum-weighted-feature-set

Bonzanini, M. (n.d.) Mining Twitter Data with Python (Part 1: Collecting data). Retrieved from 	https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/

Caruncho, E. (2016, Aug 28). Confessions of a troll. The Inquirer. Retrieved from 	http://lifestyle.inquirer.net/236403/confessions-of-a-troll/

Chao, Y., Jialong, Z., Guofei, G. (2014). A Taste of Tweets: Reverse Engineering Twitter Spammers. Retrieved from http://faculty.cse.tamu.edu/guofei/paper/TwitterRE_ACSAC14.pdf

Cocoxu. TwitterParaPhrase. (2015). Github repository. https://github.com/cocoxu/twitterparaphrase

Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., Tesconi, M. (2015). Fame for Sale: efficient 	detection of fake
Twitter followers. Retrieved from Cornell University Library. 	(arXiv:1509.04098)

Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., Tesconi, M. (2014) A Fake Follower Story: improving fake accounts detection on Twitter. Retrieved from http://wafi.iit.cnr.it/fake/fake/app/tr.pdf

Dataquest. (2016, Sep 8). Working with streaming data: Using the Twitter API to capture tweets. 	Retrieved from https://www.dataquest.io/blog/streaming-data-python/

Davis, D., Ferrera, E., Flammini, A., Menczer, F., Varol, O. (2016). BotOrNot: A System to Evaluate 	Social Bots. Retrieved from Cornell University Library. (arXiv:1602.00975)

Davis, J., Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. Retrieved from http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf

Dickerson, J., Kagan, V., Subrahmanian, V. (2014). Using sentiment to detect bots on Twitter: Are 	humans more opinionated than bots? Retrieved from 	http://jpdickerson.com/pubs/dickerson14using.pdf

Dolinar, S. (2015, Jan 29). Collecting Twitter Data: Storing Tweets in Mongodb. Retrieved from 	http://stats.seandolinar.com/collecting-twitter-data-storing-tweets-in-mongodb/

Egele, M., Stringhini, G., Kruegel, C., Vigna, G. (2013). COMPA: Detecting Compromised Accounts 	on Social Networks. Retrieved from 	https://www.cs.ucsb.edu/~vigna/publications/2013_NDSS_compa.pdf

El Berrak, R. Analyzing and Predicting ISIS Tweets. (2016). Github repository 	https://github.com/Redwa/Capstone

Fernandez-Delgado, M., Cernadas, E., Barro, S., Amorim, D. (October 2014). do we Need Hundreds of Classifiers to Solve Real World Classification Problems? Journal of Machine Learning Research 15, 3133-3181 or http://www.jmlr.org/papers/volume15/delgado14a/delgado14a.pdf

Giatsoglou, M., Chatzakou, D., Shah, N., Beutel, A., Faloutsos, C., Vakali, A. ND-SYNC: Detecting Synchronized Fraud Activities. PAKDD (2) 2015: 201-214. http://oswinds.csd.auth.gr/project/ndsync/

Giatsoglou, M., Chatzakou, D., Shah, N., Beutel, A., Faloutsos, C., Vakali, A. (2015). ND-SYNC Training Dataset [Data File]. Retrieved from http://oswinds.csd.auth.gr/project/data/Dataset.zip

Giatsoglou, M., Chatzakou, D., Shah, N., Faloutsos, C., Vakali, A. Retweeting Activity on Twitter: Signs of Deception. PAKDD (1) 2015: 122-134. http://oswinds.csd.auth.gr/project/rtscope/

Giatsoglou, M., Chatzakou, D., Shah, N., Faloutsos, C., Vakali, A. (2015). RTSCOPE Training Dataset and Code [Data File and Code]. Retrieved from http://oswinds.csd.auth.gr/project/data/RTSCOPE.zip

Gonzales, G. (2016, Sep 13). To curb hoaxes, Facebook must accept it’s a media company. Rappler. 	Retrieved from http://www.rappler.com/technology/features/145940-hoaxes-facebook-trending-	topics

Henrique, J. GetOldTweets-python. (2016). Github repository https://github.com/Jefferson-	Henrique/GetOldTweets-python

Herrman, J. (2016, Aug 24). Inside Facebook’s (Totally Insane, Unintentionally Gigantic, 	Hyperpartisan) Political-Media Machine. The New York Times Magazine. Retrieved from 	http://www.nytimes.com/2016/08/28/magazine/inside-facebooks-totally-insane-unintentionally-gigantic-hyperpartisan-political-media-machine.html?_r=0

Hofilena, C. (2016, Oct 9). Fake accounts, manufactured reality on social media. Rappler. Retrieved 	from http://www.rappler.com/newsbreak/investigative/148347-fake-accounts-manufactured-	reality-	social-media

Jeske, E. Gravitty. (2014). GitHub repository https://github.com/ericjeske/gravitty

Joseph, K., Landwehr, P., Carley, K. (2014). Two 1%s don't make a whole: Comparing simultaneous samples from Twitter's Streaming API. Retrieved from https://www.cs.cmu.edu/~kjoseph/papers/sbp_14.pdf

Karambelkar, B. (2015, Jan 5). How to use Twitter’s Search REST API most effectively. Retrieved 	from http://www.karambelkar.info/2015/01/how-to-use-twitters-search-rest-api-most-	effectively./

Karegowda, A., Manjunath, A., Jarayam, M. (July-December 2010) Comparative Study of Attribute Selection Using Gain Ratio and Correlation Based Feature Selection. International Journal of Information Technology and Knowledge Management, Volume 2 No 2, pp 271-277 or https://www.researchgate.net/profile/Jayaram_Ma/publication/228919572_Comparative_study_of_attribute_selection_using_gain_ratio_and_correlation_based_feature_selection/links/53fd54de0cf2364ccc08aa41.pdf?origin=publication_detail

Kemp, S. Digital in 2016 [PowerPoint Slides]. Retrieved from http://wearesocial.com/sg/special-	reports/digital-2016

King, G., Pan, J., Roberts, M. (2016). How the Chinese Government Fabricates Social Media Posts for 	Strategic Distraction, not Engaged Argument. Retrieved from 	http://gking.harvard.edu/files/gking/files/50c.pdf

Kontaxis, G., Polakis, I., Ioannidis, S., Markatos, E. (2011). Detecting Social Network Profile Cloning. 	Retrieved from SysSec Consortium. (n.d.).

Krombozol, K., Merkl, D., Weippl, E. (2012 Jul 26). Fake identities in Social Media: A Case Study on 	the Sustainability of the Facebook Business Model. The Society of Service Science, Volume 4, 	175-212. doi 10.1007/s12927-012-008-z

Medina, A. (2016, Oct 27). Mocha Uson Hits Rappler For Spreading ‘Fake News’. FHM Philippines. 	Retrieved from http://www.fhm.com.ph/news/mocha-uson-hits-rappler-for-spreading-fake-	news-a00169-20161027

Mori, T., Kikuchi, M., Yoshia K. (2001). Term Weighting Method based on Information Gain Ratio for Summarizing Documents retrieved by IR systems. Retrieved from http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings2/morisumm.pdf

Morstatter, F., Pfeffer, J., Liu, H., Carley, K. (2013). Is the Sample Good Enough? Comparing Data from Twitter's Streaming API with Twitter's Firehose. Retrieved from Cornell University Library (arXiv:1306:5204)

O'Connor, B, Krieger, M., Ahn, D. (2010. TweetMotif: Exploratory Search and Topic Summarization for Twitter. ICWSM-2010. Retrieved from http://www.aaai.org/ocs/index.php/ICWSM/ICWSM10/paper/viewFile/1540/1907

Qiang C., Michael S., Xiaowei Y., and Pregueiro, T. (2012). Aiding the Detection of Fake Accounts in 	Large Scale Social Online Services. Retrieved from  	https://users.cs.duke.edu/~qiangcao/publications/sybilrank.pdf

Quinlan, R. (1993). C4.5 Programs for Machine Learning. San Mateo, California: Morgan Kaufmann Publishers.

Reddy, R., Kumar, N. (2012). Automatic Detection of Fake Profiles in Online Social Networks. 	Retrieved from Ethesis @ NIT Rourkela. (3578).

Ressa, M. (2016, Oct 8). How Facebook algorithms impact democracy. Rappler. Retrieved from 	http://www.rappler.com/newsbreak/148536-facebook-algorithms-impact-democracy

Ressa, M. (2016, Oct 3). Propaganda war: Weaponizing the internet. Rappler. Retrieved from 	http://www.rappler.com/nation/148007-propaganda-war-weaponizing-internet

Rose, B. (N.A.) Document Clustering with Python. Retrieved from http://brandonrose.org/clustering

Thomas, K., Grier, C., Paxson, V., Song, D. (2011). Suspended Accounts in Retrospect: An Analysis of Twitter Spam. Retrieved from http://www.icir.org/vern/papers/twitter-susp-accounts.imc2011.pdf

Thinking Machines Data Science. (2016, Apr 1). KathNiel, Twitter bots, polls: Quality, not just buzz. 	Rappler. Retrieved from http://www.rappler.com/technology/social-media/127920-kathniel-	twitter-bots-elections-quality-buzz

Truthy. Bot or Not. (2016). Github repository https://github.com/truthy/botornot-python

Spiral of Silence. (n.d.). Retrieved Oct 27, 2016 from Mass Communication Theory 	https://masscommtheory.com/theory-overviews/spiral-of-silence/

Vladislav, K., Luja, N., Orozco, A. (2014). Detecting Subversion on Twitter. Retrieved from https://courses.csail.mit.edu/6.857/2014/files/11-kontsevoi-lujan-orozco-subversion-twitter.pdf

Willoughby, C. Chattermapper.info. (2015) Github repository 	https://github.com/seawilloughby/media_mapper

Yanofsky. Tweet Dumper. (2013). Github repository https://gist.github.com/yanofsky/5436496
