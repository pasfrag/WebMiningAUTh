# from twitter_data import TweetMiner
# from insta_data import InstaMiner
import json

from mongo import MongoHandler

# t_miner = TweetMiner()
# print("---------------------Old tweets---------------------")
# t_miner.get_tweets_with_id()
# print("---------------------New tweets---------------------")
# t_miner.get_new_tweets()
# print("---------------------Instagram----------------------")
# i_miner = InstaMiner()
# i_miner.preprocess_posts()

handler = MongoHandler()

with open('tweets_nrc.json', 'r', encoding="utf8") as json_file:
    data = json.load(json_file)

for tweet in data:
    handler.store_to_collection(tweet, "tweets_nrc_dump")


# nrc_tweets = handler.retrieve_from_collection("tweets_nrc_dump")
# for tweet in nrc_tweets:
#     sup_tweet1 = handler.get_with_id("twitter_tati", {"_id": tweet["_id"]})
#     sup_tweet = sup_tweet1[0]
#     tweet["s_anger"] = sup_tweet["anger"]
#     tweet["s_disgust"] = sup_tweet["disgust"]
#     tweet["s_fear"] = sup_tweet["fear"]
#     tweet["s_joy"] = sup_tweet["joy"]
#     tweet["s_sadness"] = sup_tweet["sadness"]
#     tweet["s_surprise"] = sup_tweet["surprise"]
#     tweet["JoyEmojis"] = sup_tweet["JoyEmojis"]
#     tweet["SadEmojis"] = sup_tweet["surprise"]
#     tweet["DisgustEmojis"] = sup_tweet["DisgustEmojis"]
#     tweet["SurpriseEmojis"] = sup_tweet["SurpriseEmojis"]
#     tweet["FearEmojis"] = sup_tweet["FearEmojis"]
#     tweet["AngerEmojis"] = sup_tweet["AngerEmojis"]
#     handler.store_to_collection(tweet, "twitter_final")
