import datetime
import time
import re
import pymongo
import lexicons
from tweepy import API, OAuthHandler, Cursor, TweepError
from mongo import MongoHandler
from preprocessing_functions import preprocess_text
from secret_keys import consumer_key, consumer_secret, access_token, access_token_secret
import textblob
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect


def sentiment_analyzer_scores(sentence):
    score = SentimentIntensityAnalyzer().polarity_scores(sentence)
    return score

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

    def get_user_tweets(self):
        re_list = []
        # for user in lexicons.deniers:
        # for user in lexicons.non_deniers:
        for user in lexicons.new_profiles2:
            count_tweets = 0
            for i in range(1, 20):  # starting with 1-10 # 1-50 for new profiles
                statuses = self.api.user_timeline(screen_name=user,
                                                  count=50, page=i, lang="en",
                                                  tweet_mode="extended") # = user !!!
                for status in statuses:
                    if any(keyword in status.full_text for keyword in lexicons.keywords) \
                            and len(status.full_text.split()) >= 5 \
                            and detect(status.full_text) == 'en':
                            # and not status.full_text.startswith("RT @"):
                        status_dict = dict()
                        status_dict["_id"] = status.id
                        status_dict["user_name"] = status.author.screen_name
                        status_dict["location"] = status.author.location
                        status_dict["description"] = preprocess_text(status.author.description)
                        status_dict['date'] = f"{status.created_at.year}-{status.created_at.month}-{status.created_at.day}"
                        clean_text = preprocess_text(re.sub(r'^RT\s@\w+:', r'', status.full_text))
                        status_dict["text"] = clean_text

                        status_dict["sentiment"] = round(sentiment_analyzer_scores(status.full_text)['compound'], 3)

                        subj = TextBlob(''.join(status.full_text)).sentiment
                        status_dict["subjectivity"] = round(subj[1],3)

                        # status_dict["label"] = 0 # non - denier
                        # status_dict["label"] = 1 # denier
                        try:
                            self.connection.store_to_collection(status_dict, "new_twitter_profiles")  # new_twitter_profiles for training data
                            count_tweets += 1
                        except pymongo.errors.DuplicateKeyError:
                            print(status.id)
                            print("\n")
                            continue
                re_list.append(statuses)
            print("Found ",count_tweets," relevant tweets by the user: ",status.author.screen_name)
        return re_list

