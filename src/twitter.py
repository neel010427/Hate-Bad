import sys
import keys.twiKey as twiKey
import tweepy
import json
import requests
import time 
    

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
                x = {'id': status._json['id_str'], 'text': status._json['text']}
                with open("tweets.json", 'a') as tf:
                    if self.count == 0:
                        tf.write('[')
                    json.dump(x,tf)
                    if self.count != end-1:
                        tf.write(',\n')
                    else:
                        tf.write(']')
                    self.count += 1
                return True
        except BaseException as e:
            print("Error on_status %s" % str(e))

    def on_error(self,status_code):
        print(status_code)
'''
    @property
    def add_tweet_to_collection(self):
        """ :reference: https://developer.twitter.com/en/docs/tweets/curate-a-collection/api-reference/post-collections-entries-add
        """
        return bind_api(
            api=self,
            path='/collections/entries/add.json',
            method='POST',
            payload_type='json',
            allowed_param=['id', 'tweet_id'],
            require_auth=True
    )
'''
auth = tweepy.OAuthHandler(twiKey.apiKey, twiKey.apiKeySec)
auth.set_access_token(twiKey.accTok, twiKey.accTokSec)
api = tweepy.API(auth)



stream_listener = StreamListener()
stream = tweepy.Stream(auth = api.auth, listener = stream_listener)
#api.add_tweet_to_collection(1231343829782941697,1231369195754930177)
# POST https://api.twitter.com/1.1/collections/entries/add.json?tweet_id=1231369195754930177&id=custom-1231343829782941697
'''
API_ENDPOINT = "https://api.twitter.com/1.1/collections/entries/add.json"

data = {'auth': twiKey.apiKey,
        'Name': "Example", 
        'id': 1231343829782941697, 
        'tweet_id': 1231369195754930177, 
        } 

r = requests.post(url = API_ENDPOINT, data = data) 

pastebin_url = r.text 
print("The pastebin URL is:%s"%pastebin_url) 
'''
open("tweets.json", 'w')
stream.sample()


'''
class streamer():
    def stream_tweets(self, out, hash_tag):
        stream_listener = StreamListener(out)
        auth = tweepy.OAuthHandler(twiKey.apiKey, twiKey.apiKeySec)
        auth.set_access_token(twiKey.accTok, twiKey.accTokSec)
        stream = tweepy.Stream(auth, stream_listener)
        stream.filter(track = hash_tag)

class StreamListener(tweepy.StreamListener): 
    def _init_(self, out):
        self.out = out
    def on_status(self, status):
        try:
            with open(self.out, 'a') as tf:
                tf.write(status)
                print(status)
            return true
        except BaseException as e:
            print("Error: %s" % str(e))
        return true
    def on_error(self,status_code):
        print(status_code)

tag = ["neel katur"]
output = "tweets.json"
twStream = streamer()
twStream.stream_tweets(output,tag)



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