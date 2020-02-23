import sys
import keys.twiKey as twiKey
import tweepy
import json
import pandas

class StreamListener(tweepy.StreamListener):
    def __init__(self,api=None):
        super(StreamListener, self).__init__()
        self.count = 0
    def on_status(self, status):
        end = 5
        try:
            if self.count == end:
                return False
            if (not status._json['text'].startswith('RT')) and (status._json['lang'] == 'en') and (not status._json['text'].startswith('@')):
                x = status._json['id_str']
                y = status._json['text']
                newDict['id'].append(x)
                newDict['text'].append(y)
                self.count += 1
                return True
        except BaseException as e:
            print("Error on_status %s" % str(e))

    def on_error(self,status_code):
        print(status_code)

auth = tweepy.OAuthHandler(twiKey.apiKey, twiKey.apiKeySec)
auth.set_access_token(twiKey.accTok, twiKey.accTokSec)
api = tweepy.API(auth)

newDict = {"id": [], 'text':[]}
stream_listener = StreamListener()
stream = tweepy.Stream(auth = api.auth, listener = stream_listener)
open("tweets.json", 'w')
stream.sample()
data = pandas.DataFrame(newDict)
print(data)