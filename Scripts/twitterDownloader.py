import csv
import pandas as pd
import pytz
import re
import sqlite3
import time
import tweepy
            
class TwitterDownloader():
    ''' Class to download and store streets in a database'''
    def __init__(self, tweets_filename = "Tweets.db"):
        # 1. Create database and credentials.        
        self.__create_database(tweets_filename)
        self.nCredentials = 6
        self.credentials = retrieve_credentials(self.nCredentials)
        
        # 2. Generate the list of the news agencies.
        # The Sun is excluded since it is not active.
        # Blomberg is excluded since it does not exist.
        self.users = ["nytimes", "thetimes", "ap", "cnn", "bbcnews", "cnet", "msnuk",
                "telegraph", "usatoday", "wsj", "washingtonpost", "bostonglobe",
                "newscomauhq", "skynews", "sfgate", "ajenglish", "independent",
                "guardian", "latimes", "reutersagency", "abc", "bw", "time"]
        self.nUsers = len(self.users)
        
        # 3. Initialize the IDs of the tweets as 0.
        self.newest_ids = [0] * self.nUsers
    
    # This private function creates a database to store the tweets.
    def __create_database(self, db_filename):
        ''' The name of the table is Tweet.'''
        self.conn = sqlite3.connect(db_filename)
        self.conn.text_factory = str
        self.c = self.conn.cursor()
        self.c.execute('DROP TABLE IF EXISTS Tweet')
        self.c.execute('CREATE TABLE Tweet (Created_At timestamp, ID integer, Tweet text, \
                Source text, URL text)')
                
    # This private function download tweets from a specific user.
    def __download_tweets(self, screen_name, since_id):
        ''' Returns the ID of the latest tweet.'''
        # 1. Authorize twitter and initialize tweepy.
        auth = tweepy.OAuthHandler(self.credential[0], self.credential[1])
        auth.set_access_token(self.credential[2], self.credential[3])
        self.api = tweepy.API(auth)
        
        # 2. Make a request for the most recent tweets.
        new_tweets = self.__request_tweets(screen_name, since_id)
        nTweet = len(new_tweets)
        print screen_name, ": Number of tweets downloaded: %s" % nTweet
        
        # 3. Get the id of the latest tweet.
        newest_id = since_id
        if nTweet > 0:
            newest_id = new_tweets[0].id_str
        
        # 4. Insert the tweets into the database
        self.__insert_records(new_tweets, screen_name)
        
        return newest_id
    
    # This private function insert records into the database
    def __insert_records(self, new_tweets, screen_name):
        for tweet in new_tweets:
            tweet = self.__split_tweets(tweet)
        self.c.executemany('INSERT INTO Tweet(Created_At, ID, Tweet, Source, URL) VALUES (?,?,?,?,?)',
        ([tweet.created_at, tweet.id_str, tweet.text.encode("utf-8"), screen_name, tweet.link] \
        for tweet in new_tweets))
        self.conn.commit()
        
    # This private function separate the link from a tweet's content
    def __split_tweets(self, tweet):
        ''' Returns tweet with text and link separated.'''
        http = re.search("(?P<url>https?://[^\s]+)", tweet.text)
        if http is not None:
            tweet.link = http.group("url")
        else:
            tweet.link = ""
        tweet.text = re.sub(r"http\S+", "", tweet.text)
        return tweet
    
    # This private function requests tweets from Twitter API
    def __request_tweets(self, screen_name, id):
        ''' Returns the tweets obtained from Twitter.'''
        ''' The rate limit exception can be handled by detecting the error code 88, 
            followed by a pause of 15 minutes. But if we only request for tweets
            every 10 minutes, this rate limit will not be exceeded, so this exception
            is not handled here. An exception that is of concern here is the connection
            problem as the internet connection might be disconnected temporary.'''
        new_tweets = []
        while True:
            try:
                if (id == 0):
                    new_tweets = self.api.user_timeline(screen_name = screen_name, count = 25)
                else:
                    new_tweets = self.api.user_timeline(screen_name = screen_name, count = 200,
                            since_id = id)
                break
            except tweepy.TweepError, ex:
                print "Error: %s" % ex
        return new_tweets
        
    # This functions pick one of the available sets of credentials.
    # (To prevent the twitter rate limit problem).
    def select_credentials(self, index = 0):
        n = index % self.nCredentials
        self.credential = [self.credentials['ckey'][n], self.credentials['csecret'][n],
                self.credentials['akey'][n], self.credentials['asecret'][n]]

    # This function download tweets from all news agencies that are of interest.
    def download_tweets_all(self, id):
        '''Members:
            self.newest_ids: list of the newest IDs for the newest tweet''' 
        newest_ids = []
        i = 0
        for user in self.users:
            id[i] = self.__download_tweets(user, since_id = id[i])
            newest_ids.append(id[i])
            i += 1
        self.newest_ids = newest_ids


# This private function retrieves the information to have access to Twitter API.
def retrieve_credentials(n):
    ''' Returns the information needed for Twitter API authorization'''
    # 1. Read the information from token.txt.
    with open("../Input/token.txt") as f:
        reader = csv.reader(f)
        for row in reader:
            content =  row
     
    # 2. Assign Twitter API credentials to a DataFrame.
    credentials = pd.DataFrame()
    credentials['ckey'] = content[0:n]
    credentials['csecret'] = content[n:2*n]
    credentials['akey'] = content[2*n:3*n]
    credentials['asecret'] = content[3*n:4*n]
    return credentials
