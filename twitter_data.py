import time
from secret_keys import consumer_key, consumer_secret, access_token, access_token_secret
from tweepy import API, OAuthHandler, Cursor, RateLimitError
import mongo

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except RateLimitError:
            time.sleep(15 * 60)

# Creating the API
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth)

# Getting the data
for tweet in limit_handled(Cursor(api.search, q="#ClimateChange", lang="en", tweet_mode="extended").items()):
    tweet._json["_id"] = tweet._json.pop("id")
    print(tweet._json)
    mongo.create_mongo_collection("twitter", tweet._json)
