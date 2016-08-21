# Python
Python sentiment analysis 

python script that I wrote for the sentiment analysis. This script takes the boulder_twitter.csv and twitter_sentiment_list.csv (sentiment dataset from https://github.com/jmorales4/W205-RJCL-Public/blob/master/twitter_sentiment_list.csv) as inputs and gives out boulder_sentiment.tsv which has geohash and sentiment score.

In order to visualize the geohashes on a map, I ran another python script

python geohash2kml.py boulder_sentiment.tsv boulder_sentiment.kml

(taken from https://github.com/abeusher/geohash2kml), which gives out boulder_sentiment.kml.

