import time, datetime, json
from preprocessing_functions import preprocess_text
from secret_keys import consumer_key, consumer_secret, access_token, access_token_secret
from tweepy import API, OAuthHandler, Cursor, RateLimitError, TweepError
from pymongo import MongoClient

class TweetMiner(object):

    api = None
    # data = []

    def __init__(self):
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def get_tweets_with_id(self):

        client = MongoClient("mongodb://localhost:27017/")
        db = client["web_mining"]
        collection = db["twitter"]
        x = collection.find({"_id": 1236135735201329153})
        ids_list = [
            row["_id"] for row in x
            if not row["full_text"].startswith("RT @") and ("promo" or "giveaway") not in row["full_text"] and len(row["full_text"].split()) >= 5
        ]
        # print(ids_list)

        count = 0

        for tweet_id in ids_list:
            try:
                tweet = self.api.get_status(tweet_id, tweet_mode="extended")
                pre_tweet = self.preprocess_tweet(tweet)
                # self.data.append(tweet)
                # json_str = json.dumps(pre_tweet._json)
                # parsed = json.loads(json_str)
                print(json.dumps(pre_tweet, indent=4, sort_keys=True))
            except TweepError:
                count += 1

    def preprocess_tweet(self, tweet):
        tweet_dict = dict()
        tweet_dict["_id"] = tweet._json["id"]
        tweet_dict["text"] = preprocess_text(tweet._json["full_text"])
        tweet_dict["hashtags"] = [hashtag["text"] for hashtag in tweet._json["entities"]["hashtags"]]
        tweet_dict["mentions"] = [hashtag["name"] for hashtag in tweet._json["entities"]["user_mentions"]]
        tweet_dict["hashtags"] = [hashtag["text"] for hashtag in tweet._json["entities"]["hashtags"]]
        tweet_dict["urls"] = [hashtag["url"] for hashtag in tweet._json["entities"]["urls"]]
        tweet_dict["user_id"] = tweet._json["user"]["id"]
        tweet_dict["user_name"] = tweet._json["user"]["name"]
        tweet_dict["user_screen_name"] = tweet._json["user"]["screen_name"]
        tweet_dict["user_location"] = tweet._json["user"]["location"]
        tweet_dict["user_description"] = tweet._json["user"]["description"]
        tweet_dict["user_followers"] = tweet._json["user"]["followers_count"]
        tweet_dict["user_friends"] = tweet._json["user"]["friends_count"]
        tweet_dict["user_listed"] = tweet._json["user"]["listed_count"]
        tweet_dict["user_favourites"] = tweet._json["user"]["favourites_count"]
        ts = time.strftime('%Y-%m', time.strptime(tweet._json["user"]["created_at"], '%a %b %d %H:%M:%S +0000 %Y'))
        date_time_obj = datetime.datetime.strptime(ts, '%Y-%m')
        end_date = datetime.datetime.now()
        num_months = (end_date.year - date_time_obj.year) * 12 + (end_date.month - date_time_obj.month)
        tweet_dict["user_months"] = num_months
        tweet_dict["user_statuses"] = tweet._json["user"]["statuses_count"]
        tweet_dict["user_verified"] = int(tweet._json["user"]["verified"])
        tweet_dict["geo"] = tweet._json["geo"]
        tweet_dict["place"] = tweet._json["place"]
        tweet_dict["coordinates"] = tweet._json["coordinates"]
        tweet_dict["retweets"] = tweet._json["retweet_count"]
        tweet_dict["favorites"] = tweet._json["favorite_count"]
        tweet_dict["is_quoted"] = tweet._json["is_quote_status"]
        return tweet_dict

    # Getting the data
# i = 0
# for tweet in Cursor(api.search, q="@mattduss", lang="en", tweet_mode="extended").items(): #q="#ClimateChange"
#     # tweet._json["_id"] = tweet._json.pop("id")
#     #
#     # tweet_dict = dict()
#     #
#     # print(tweet._json)
#     # mongo.create_mongo_collection("twitter", tweet._json)
#     json_str = json.dumps(tweet._json)
#     parsed = json.loads(json_str)
#     print(json.dumps(parsed, indent=4, sort_keys=True))
#     i += 1
#     if i > 50:
#         break
