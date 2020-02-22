import sys
import keys.twiKey as twiKey
import tweepy
import json

import time
import random

class StreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status)
    def on_error(self,status_code):
        print(status_code)

auth = tweepy.OAuthHandler(twiKey.apiKey, twiKey.apiKeySec)
auth.set_access_token(twiKey.accTok, twiKey.accTokSec)
api = tweepy.API(auth)

stream_listener = StreamListener()
stream = tweepy.Stream(auth = api.auth, listener = stream_listener)
stream.filter(track = ['Nathan is cool'])
'''
def get_tweet(api, username):
    page = random.randint(1,5)
    tweet = api.user_timeline(username, page = page)
    for i in tweet:
        print(i.text.encode("utf-8"))
    return

def get_user(api,ID):
    tweets = api.GetStreamSample()
    count = 0
    for tweet in tweets:
        if count == 100:
            break
        count += 1
        ID.append(tweet['user']['id'])
    print(ID)

users = []
get_user(api,users)
'''