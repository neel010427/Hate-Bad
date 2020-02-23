import keys.twiKey as twiKey
import tweepy
import json



auth = tweepy.OAuthHandler(twiKey.apiKey, twiKey.apiKeySec)
auth.set_access_token(twiKey.accTok, twiKey.accTokSec)
api = tweepy.API(auth)

def trend(api):
    class pos():
        def __init__(self):
            self.text = ['']
            self.pos = ['']
    #---------------Check positivity of trends-----------------------
    trends = api.trends_place(23424977)
    trendPos = pos()
    for i in trends:
        for t in i['trends']:
            trendPos.text.append(t['name'])

    pos = json.load(open("tweets.json"))
    neg = json.load(open("tweets.json"))
    for j in trendPos.text:
        posCount = 0
        negCount = 0
        for i in pos:
            if i['text'].find(j) != -1:
                posCount += 1
        for k in neg:
            if k['text'].find(j) != -1:
                negCount += 1
        if negCount > posCount:
            trendPos.pos.append('neg')
        else:
            trendPos.pos.append('pos')
    print(trendPos.text)
    print(trendPos.pos)

#---------------------print out puppies---------------
def puppies(api):
    pup = api.user_timeline('CuteEmergency', count = 3, page = 1)
    for i in pup:
        print(i._json['id'])

#-----------------flex-----------------------
def flex(api):
    api.update_with_media(filename = 'woof.gif')
flex(api)