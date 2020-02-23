#import base64
import keys.twiKey as twiKey
#import requests
import tweepy

auth = tweepy.OAuthHandler(twiKey.apiKey, twiKey.apiKeySec)
auth.set_access_token(twiKey.accTok, twiKey.accTokSec)
api = tweepy.API(auth)
api.retweet(1231410006337171458)

# #Define your keys from the developer portal
# client_key = twiKey.apiKey
# client_secret = twiKey.apiKeySec
# #Reformat the keys and encode them
# key_secret = '{}:{}'.format(client_key, client_secret).encode('ascii')

# # Transform from bytes to bytes that can be printed
# b64_encoded_key = base64.b64encode(key_secret)
# #Transform from bytes back into Unicode
# b64_encoded_key = b64_encoded_key.decode('ascii')

# base_url = 'https://api.twitter.com/'
# auth_url = '{}oauth2/token'.format(base_url)
# auth_headers = {
#     'Authorization': 'Basic {}'.format(b64_encoded_key),
#     'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
# }
# auth_data = {
#     'grant_type': 'client_credentials'
# }
# auth_resp = requests.post(auth_url, headers=auth_headers, data=auth_data)

# access_token = auth_resp.json()['access_token']

# search_url = "https://api.twitter.com/1.1/collections/entries/add.json"

# search_headers = {
#     'Authorization': 'Bearer {}'.format(access_token)
# }

# search_params = {'Name': "Example", 
#         'id': 1231343829782941697, 
#         'tweet_id': 1231369195754930177, 
#         } 

# r = requests.post(url = search_url, headers = search_headers, data = search_params) 
# # print(r.status_code)
# # print(r.json().keys())
# pastebin_url = r.text 
# print("The pastebin URL is:%s"%pastebin_url) 