import datetime
import time
from tweepy import API, OAuthHandler, Cursor, TweepError
from mongo import MongoHandler
from preprocessing_functions import preprocess_text
from secret_keys import consumer_key, consumer_secret, access_token, access_token_secret


class TweetMiner(object):

    api = None
    connection = None

    def __init__(self):
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.connection = MongoHandler()

    def get_tweets_with_id(self):

        old_posts = self.connection.retrieve_from_collection("twitter")
        new_posts = self.connection.retrieve_from_collection("twitter_new")
        new_ids_list = [row["_id"] for row in new_posts]
        ids_list = [
            row["_id"] for row in old_posts
            if not row["_id"] in new_ids_list and not row["full_text"].startswith("RT @") and ("promo" or "giveaway") not in row["full_text"] and len(row["full_text"].split()) >= 5
        ]

        print("Starting...")
        count0 = 0
        count1 = 0
        for tweet_id in ids_list:
            try:
                # tweet = self.api.get_status(tweet_id, tweet_mode="extended")._json
                tweets = self.connection.get_with_id("twitter", {"_id": tweet_id})
                for tweet in tweets:
                    _pre_tweet = self.preprocess_tweet(tweet)
                    # print(json.dumps(pre_tweet, indent=4, sort_keys=True))
                    count1 += 1
            except TweepError:
                count0 += 1

        print("--------------------------------")
        print(f"Number of found: {count1}")
        print("--------------------------------")
        print(f"Number of not found: {count0}")

    def preprocess_tweet(self, tweet):
        tweet_dict = dict()
        tweet_dict["_id"] = tweet["_id"]
        created_at = time.strftime('%Y-%m-%d', time.strptime(tweet["created_at"], '%a %b %d %H:%M:%S +0000 %Y'))
        tweet_dict["created_at"] = created_at
        tweet_dict["text"] = preprocess_text(tweet["full_text"])
        tweet_dict["hashtags"] = [hashtag["text"] for hashtag in tweet["entities"]["hashtags"]]
        tweet_dict["mentions"] = [hashtag["name"] for hashtag in tweet["entities"]["user_mentions"]]
        tweet_dict["hashtags"] = [hashtag["text"] for hashtag in tweet["entities"]["hashtags"]]
        tweet_dict["urls"] = [hashtag["url"] for hashtag in tweet["entities"]["urls"]]
        tweet_dict["user_id"] = tweet["user"]["id"]
        tweet_dict["user_name"] = tweet["user"]["name"]
        tweet_dict["user_screen_name"] = tweet["user"]["screen_name"]
        tweet_dict["user_location"] = tweet["user"]["location"]
        tweet_dict["user_followers"] = tweet["user"]["followers_count"]
        tweet_dict["user_friends"] = tweet["user"]["friends_count"]
        tweet_dict["user_listed"] = tweet["user"]["listed_count"]
        tweet_dict["user_favourites"] = tweet["user"]["favourites_count"]
        ts = time.strftime('%Y-%m', time.strptime(tweet["user"]["created_at"], '%a %b %d %H:%M:%S +0000 %Y'))
        date_time_obj = datetime.datetime.strptime(ts, '%Y-%m')
        end_date = datetime.datetime.now()
        num_months = (end_date.year - date_time_obj.year) * 12 + (end_date.month - date_time_obj.month)
        tweet_dict["user_months"] = num_months
        tweet_dict["user_statuses"] = tweet["user"]["statuses_count"]
        tweet_dict["user_verified"] = int(tweet["user"]["verified"])
        tweet_dict["retweets"] = tweet["retweet_count"]
        tweet_dict["favorites"] = tweet["favorite_count"]
        tweet_dict["is_quoted"] = tweet["is_quote_status"]
        self.connection.store_to_collection(tweet_dict, "twitter_new")
        return tweet_dict

    def get_new_tweets(self):
        count = 0
        for tweet in Cursor(self.api.search, q="@#ClimateChange", lang="en", tweet_mode="extended").items():
            if not tweet._json["full_text"].startswith("RT @") and ("promo" or "giveaway") not in tweet._json["full_text"] and len(tweet._json["full_text"].split()) >= 5:
                count += 1
                self.preprocess_tweet(tweet._json)
        print("--------------------------------")
        print(f"Number of found: {count}")

    # Getting the data
# i = 0
# for tweet in Cursor(api.search, q="@", lang="en", tweet_mode="extended").items(): #q="#ClimateChange"
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
