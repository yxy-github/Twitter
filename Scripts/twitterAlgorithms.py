import csv
from datetime import datetime
import numpy as np
import pandas as pd
import re
import sqlite3
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import ward, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import *
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, normalize

class TwitterAlgorithms:
    ''' Class for algorithms to preprocess tweets, extract features, perform clustering, 
            and detect trends'''
    def __init__(self, tweets_filename = "Tweets.db", results_filename = "Results.csv",
            notification_filename = "Notification.txt", alerts_filename = "Alerts.db"):
        self.nPush = 0
        
        # 1. Connect to the database.
        self.conn = sqlite3.connect(tweets_filename)
        self.conn.text_factory = str
        self.c = self.conn.cursor()
        self.connAlert = sqlite3.connect(alerts_filename)
        self.connAlert.text_factory = str
        self.cAlert = self.connAlert.cursor()
        self.cAlert.execute('DROP TABLE IF EXISTS Alert')
        self.cAlert.execute('CREATE TABLE Alert (Created_At timestamp, ID integer, \
                Tweet text, Source text, URL text)')

        # 2. Load the stop words.
        # The lookup is O(1) for each lookup. If we use a list, the run time is
        # O(n) where n is the number of words.
        self.stops = set(stopwords.words("english"))

        # 3. Open the file for recording the alerts and notification.
        self.fNotification = open(notification_filename, 'w')
        fAlert = open(results_filename, 'wb')
        self.writer = csv.writer(fAlert)        
        self.writer.writerow(["Flag", "Time", "Tweet", "Source", "Link", "nTweets_in_Cluster",
                "nSource", "BestSim", "AveSim", "nFeatures", "nClusters"])               
        
    # This private function cleans the raw text and converts it to a string of words.
    def __clean_tokenize(self, raw_text):
        ''' Returns the tweets in the form of words.'''
        # 1. Remove RT, @USER, :, lead/trailing space.
        raw_text =  re.sub(r"RT @\S+", "", raw_text)
        raw_text =  re.sub(r"@\S+\s*", "", raw_text)
        # Some tweets ends the sentence with ':'. Replace these ':' with '.'.
        raw_text =  re.sub(r":\s+$", ".", raw_text)
        # Some tweets has ':' in the middle sentence followed by "".
        # Replace these ':' with '.'
        raw_text =  re.sub(r":\s+(\"|\')", ". ", raw_text)
        # Some tweets start with e.g. "INDEPENDENT FRONT PAGE:" etc.
        # Replace the first ":" in the sentence with "*".
        raw_text =  re.sub(":", "*", raw_text, 1)
        raw_text =  re.sub(r"^.+\*", "", raw_text)
        raw_text =  raw_text.strip()
        
        # 2. Remove non-letters characters
        raw_letters = re.sub("[^a-zA-Z]", " ", raw_text)
        
        # 3. Convert the text to lower case and then split it into words.
        words = raw_letters.lower().split()            
        
        # 4. Remove stop words.
        clean_words = [w for w in words if not w in self.stops]
            
        # 5. Stem the words.
        #porter_stemmer = nltk.stem.PorterStemmer()
        #for j, word in enumerate(clean_words):
        #    clean_words[j] = porter_stemmer.stem(word)
            
        # 6. Join the words back into one string separated by space
        return( " ".join(clean_words))

    # This private function builds a DataFrame for the tweets retrieved from the database
    # and removes duplicates based on the tweets' content.
    def __get_dataFrame(self, results, dropDuplicates = True):
        ''' Returns the DataFrame of the tweets.'''
        # 1. Create a DataFrame.
        tweets = pd.DataFrame()
        tweets['date'] = map(lambda tweet: tweet[0], results)    
        tweets['id'] = map(lambda tweet: tweet[1], results)
        tweets['text'] = map(lambda tweet: tweet[2], results)
        tweets['source'] = map(lambda tweet: tweet[3], results)
        tweets['link'] = map(lambda tweet: tweet[4], results)
        tweets['time'] = map(lambda tweet: float(tweet[5]), results)
        
        # 2. Remove duplicates.
        if dropDuplicates is True:
            tweets = tweets.drop_duplicates('text')
            # Reset the indices of the DataFrame.
            tweets = tweets.reset_index()
            del tweets['index']
        return tweets
            
    # This private function generates the feature matrix from the cleaned and tokenized tweets.
    def __generate_features(self, clean_data, tweets, lsa = True, addTime = True):
        ''' Returns the feature matrix.'''
        # 1. Convert a collection of text documents to a matrix of token counts.
        vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,
                preprocessor = None, stop_words = None, max_features = 5000)

        # 2. Learn the vocabulary dictionary and return term-document matrix.
        features = vectorizer.fit_transform(clean_data)
        features = features.toarray()     # Convert to a Numpy array
                
        # 3. Transform a count matrix to a normalized tf or tf-idf representation.
        tfidf_transformer = TfidfTransformer()
        features_tfidf = tfidf_transformer.fit_transform(features)
        
        if lsa is True:
            # 4. Estimate the number of SVD components to keep.
            vocab = vectorizer.get_feature_names()
            nComponent = int(round(len(vocab) / 10))
            if nComponent < features.shape[1]:
                nComponent = features.shape[1] - 1
            
            # 5. Apply SVD for dimensionality reduction (LSA).
            svd = TruncatedSVD(n_components = nComponent, random_state = 42)
            lsa = make_pipeline(svd, Normalizer(copy = False))
            features_all = lsa.fit_transform(features_tfidf)
        else:
            features_all = features_tfidf
        
        # 6. Add the normalize time difference feature to the feature matrix.
        if addTime is True:
            normTime = normalize(tweets['time'])
            features_all = np.concatenate((features_all, np.transpose(normTime)), axis = 1)  
        return features_all
    
    # This private function prints the notification message on console.
    def __print_notification(self, results):
        print "******************************************"
        print "Notification ", self.nPush
        print "******************************************"
        print "Time (UTC): ", str(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        print "Push notification to the user!!!"
        print "Message sent: \'", results[-1][3], "\': ", results[-1][2], \
                ", link: ", results[-1][4], " created at: ", results[-1][0]    
        print "******************************************"
        
    # This private function prints the results on the console.
    def __print_results(self, tweets, nCluster):
        print "The size of the best cluster is ", self.nTweets
        print "All the tweets in the cluster:"
        print tweets['text'][nCluster[0]]
        print "******************************************"
        print "The top tweet is:"
        print self.bestTweet
        print "The max cosine similarity value: ", self.maxSimilarity
        print "The average cosine similarity value: ", self.aveSimilarity
        print "The number of news agencies in the cluster: ", self.nAgencies
        
    # This private function stores the important tweet to the database
    def __store_alert(self, tweets):
        self.cAlert.execute('INSERT INTO Alert(Created_At, ID, Tweet, Source, URL) VALUES (?,?,?,?,?)',
                ([tweets['date'].tolist()[self.iBestTweet], tweets['id'].tolist()[self.iBestTweet].item(),
                tweets['text'].tolist()[self.iBestTweet], tweets['source'].tolist()[self.iBestTweet], 
                tweets['link'].tolist()[self.iBestTweet]]))
        self.connAlert.commit()
        
    # This private function cleans all tweets.
    def __tokenize_tweets(self, tweets):
        ''' Returns the cleaned + tokenized words.'''
        clean_data = []
        for t in tweets['text']:
            clean_data.append(self.__clean_tokenize(t))
        return clean_data

    # This function builds a feature matrix for the raw tweets retrieved from the database.
    def build_features(self, results, dropDuplicates, lsa, addTime):
        ''' Returns the DataFrame of the tweets and the feature matrix.'''
        tweets = self.__get_dataFrame(results, dropDuplicates)
        clean_data = self.__tokenize_tweets(tweets)
        features = self.__generate_features(clean_data, tweets, lsa, addTime)
        return (tweets, features)
    
    # This function compares the similarity between all the important alerts recorded
    def compare_alerts(self, results):
        ''' Returns flagPush: True if pull notification is triggered and False otherwise.'''
        flagPush = False
        if len(results) > 1:
            tweets, features = self.build_features(results, dropDuplicates = False, lsa = True, 
                    addTime = False)
            distAlerts = cosine_similarity(features[-1,:], features[0:features.shape[0]-1,:])
            vote = sum([1 if x >= 0.9 else 0 for x in distAlerts[0]])
            if vote == 0:
                flagPush = True
        elif len(results) == 1:
            flagPush = True
        return flagPush
        
    # This function clusters the features using Hierarchical Clustering.
    def find_clusters(self, features):
        ''' Returns the clusters and their centroids.'''
        # 1. Cluster the data.
        totalClusters = int(round(features.shape[0] / 2))
        distance = 1 - pairwise_distances(features, metric = "cosine")
        # Ward minimizes the sum of squared differences within all clusters.
        # It is a variance-minimizing approach, which is similar to the k-means objective function.
        linkage_matrix = ward(distance)
        clusters = fcluster(linkage_matrix, totalClusters, criterion = 'maxclust')
        print "Number of clusters:", totalClusters
        
        # 2. Find the centroid for each cluster.
        centroid = np.empty([totalClusters, features.shape[1]])
        for i in range(1, totalClusters + 1):
            nCluster = np.where(clusters == i)
            centroid[i-1,:] = np.mean(features[nCluster], axis = 0)
        return (clusters, centroid)
        
    # This function generates an alert for trending tweets.
    def generate_alert(self, tweets):
        ''' Member:
                self.flag: 1 if a tweet is deemed important and 0 otherwise.'''
        if self.maxSimilarity > 0.9 and self.aveSimilarity > 0.86 and \
                self.nAgencies > 2 and self.nTweets > 4 and (nTweets / (nAgencies * 1.0)) < 1.5:
            self.flag = 1
            print "This is an important tweet!!!"
            self.__store_alert(tweets)
            # Decide if a pull notification should be triggered.
            alertDateTime = tweets['date'].tolist()[self.iBestTweet]
            self.generate_notification(alertDateTime)
        else:
            print "This is not an important tweet."
            self.flag = 0
    
    # This function decides whether or not to trigger the pull notification module.
    # If triggered, a notification message will be generated.
    def generate_notification(self, alertDateTime):
        alertDate = alertDateTime[0:alertDateTime.index(' ')]
        self.cAlert.execute('SELECT *,((strftime(\'%s\',?) - strftime(\'%s\',Created_At)) / 60) FROM Alert \
                WHERE Created_at >= date(?) AND Created_at < date(?,\'+1 day\')',
                [alertDateTime, alertDate, alertDate])
        results = self.cAlert.fetchall()
        flagPush = self.compare_alerts(results)    
        if flagPush is True:
            self.nPush += 1
            self.fNotification.write("Notification %s\n" % self.nPush)
            self.fNotification.write("Message:\n")
            self.fNotification.write("\'%s\': %s, link: %s, created at: %s\n\n" \
                    % (results[-1][3], results[-1][2], results[-1][4], results[-1][0]))           
            self.__print_notification(results)            
        
    # This function selects the 20-minute worth of news tweets.
    def select_tweets(self, newest):
        ''' Returns the tweets retrieved.'''
        self.newest = newest
        self.c.execute('SELECT *,((strftime(\'%s\',?) - strftime(\'%s\',Created_At)) / 60) AS Difference \
                FROM Tweet WHERE Difference BETWEEN 0 AND 121 \
                ORDER BY Created_At DESC', self.newest[0])
        results = self.c.fetchall()
        return results
    
    # This function selects the best tweets and prints the results on the console.
    def select_bestTweets(self, clusters, centroid, features, tweets):
        ''' Members:
                self.bestTweet: the best tweet
                self.nTweets: number of tweets in the best cluster
                self.nAgencies: number of news agencies contributing to the best cluster
                self.maxSimilarity: maximum cosine similarity
                self.aveSimilarity: average cosine similarity
                self.iBestTweet: index of the best tweet'''
        # 1. Select the best cluster (the cluster with the most tweets).
        uniqueCluster, counts = np.unique(clusters, return_counts = True)
        (mx, imx) = max((v, i) for i, v in enumerate(counts))
        nCluster = np.where(clusters == (imx + 1))
        self.nTweets = nCluster[0].size  # The number of tweets in the cluster.
            
        # 2. Compute cosine similarity of each member.
        distBestCluster = cosine_similarity(centroid[imx,:], features[nCluster])
        (self.maxSimilarity, iTweet) = max((v, i) for i,v in enumerate(distBestCluster[0,:]))
        self.aveSimilarity = np.mean(distBestCluster)
        
        # 3. Select the best tweet (the one with the max cosine similarity).
        self.bestTweet = tweets['text'][nCluster[0][iTweet]]
        self.iBestTweet = nCluster[0][iTweet]
        agencies, counts = np.unique(tweets['source'][nCluster[0]], return_counts = True)
        self.nAgencies = len(agencies)
        
        # 4. Print the results.
        self.__print_results(tweets, nCluster)
    
    # This function write the results related to the alert to the csv file.
    def write_results(self, tweets, totalFeatures, totalClusters):
        output = [self.flag, self.newest[0], tweets['text'][self.iBestTweet],
                tweets['source'][self.iBestTweet], tweets['link'][self.iBestTweet],
                self.nTweets, self.nAgencies, self.maxSimilarity, self.aveSimilarity,
                totalFeatures, totalClusters]
        self.writer.writerows([output])
