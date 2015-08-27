import os
import sched
import time
from twitterAlgorithms import TwitterAlgorithms
from twitterDownloader import TwitterDownloader

schedule = sched.scheduler(time.time, time.sleep)

# This functions will be executed every 10 minutes.
# The scheduled tasks are downloading data and processing the data.
def schedule_task(downloader, algorithm):
    global counter
    print "\nIteration: ", counter
    schedule.enter(600, 1, schedule_task, (downloader, algorithm, ))
    (newest, downloader) = download_data(counter, downloader)
    process_data(newest, algorithm)
    counter += 1

# This functions executes all the methods required to download and store tweets.
def download_data(counter, downloader):
    '''Returns the newest datetime from the database.'''
    downloader.select_credentials(counter)
    newest_ids = downloader.newest_ids
    downloader.download_tweets_all(newest_ids)
    downloader.c.execute('SELECT Created_At FROM Tweet ORDER BY Created_At DESC LIMIT 1')
    newest = downloader.c.fetchall()
    return newest, downloader

# This functions executes all the methods required to process the tweets and generates an output.   
def process_data(newest, algorithm):
    results = algorithm.select_tweets(newest)
    tweets, features = algorithm.build_features(results, dropDuplicates = True, lsa = True, addTime = True)
    clusters, centroid = algorithm.find_clusters(features)
    algorithm.select_bestTweets(clusters, centroid, features, tweets)
    algorithm.generate_alert(tweets)
    algorithm.write_results(tweets, features.shape[1], centroid.shape[0])
    

if __name__ == '__main__':
    counter = 0
    if not os.path.isdir('../Output'):
        os.mkdir('../Output')
    downloader = TwitterDownloader("../Output/Tweets.db")
    algorithm = TwitterAlgorithms("../Output/Tweets.db", "../Output/Results.csv", 
            "../Output/Notification.txt", "../Output/Alerts.db")
    schedule.enter(1, 1, schedule_task, (downloader, algorithm, ))
    schedule.run()
    algo.conn.close()
    conn.close()
 