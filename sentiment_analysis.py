import geohash 
import pandas as pd
from textblob import TextBlob
from ttp import ttp
import numpy as np
from nltk.corpus import stopwords 
from bs4 import BeautifulSoup
import re
import warnings
warnings.filterwarnings('ignore')

tt_p = ttp.Parser()
happy_log_probs = {}
sad_log_probs = {}

def readSentimentList(file_name):
    ifile = open(file_name, 'r')
    ifile.readline() 
    
    for line in ifile:
        try:
            tokens = line[:-1].split(',')
            happy_log_probs[tokens[0]] = float(tokens[1])
            sad_log_probs[tokens[0]] = float(tokens[2])
        except IndexError:
            print line
            

    return happy_log_probs, sad_log_probs

def classifySentiment(words):
    words=words
    if len(words) <= 4:
        return 0
    happy_probs = [happy_log_probs[word] for word in words if word in happy_log_probs]
    sad_probs = [sad_log_probs[word] for word in words if word in sad_log_probs]

    tweet_happy_log_prob = np.sum(happy_probs)
    tweet_sad_log_prob = np.sum(sad_probs)

    prob_happy = np.reciprocal(np.exp(tweet_sad_log_prob - tweet_happy_log_prob) + 1)
    prob_sad = 1 - prob_happy
    return prob_happy

def str_to_words( tweet ):
    #removing html
    review_text = BeautifulSoup(tweet).get_text() 
    #removing urls
    no_urls = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', BeautifulSoup(review_text).get_text())
    letters_only = re.sub("[^a-zA-Z]", " ", no_urls) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words )) 

def main():
    boulder_twitter = pd.read_csv('boulder_twitter.csv', names=['uname','lat','lon','tweet'] )
    boulder_twitter.dropna(inplace=True)
    #geohash encode with precision 6 (approx 0.6km * 0.6km) square
    boulder_twitter['geohash'] = boulder_twitter.apply(lambda x:geohash.encode(x.lat,x.lon,6), axis=1)
    # sentiment dataset from https://github.com/jmorales4/W205-RJCL-Public/blob/master/twitter_sentiment_list.csv
    happy_log_probs, sad_log_probs = readSentimentList('twitter_sentiment_list')
    boulder_twitter['parsed_tweet'] = boulder_twitter.tweet.apply(lambda x:str_to_words(x))
    boulder_twitter['sentiment'] = boulder_twitter.parsed_tweet.apply(lambda x:classifySentiment(x.split(' ')))
    out = boulder_twitter.groupby('geohash')['sentiment'].mean().reset_index()
    out['sentiment'] = out.sentiment.apply(lambda x:int(x*100))
    out.to_csv('/home/ranjitha/myapp/boulder_sentiment.tsv', index=False, sep='\t')
    

if __name__=='__main__':
    main()

