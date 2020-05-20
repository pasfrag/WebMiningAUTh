from twitter_data import TweetMiner
# from insta_data import InstaMiner
from mongo import MongoHandler

t_miner = TweetMiner()
# print("---------------------Old tweets---------------------")
# t_miner.get_tweets_with_id()
# print("---------------------New tweets---------------------")
# t_miner.get_new_tweets()
# print("---------------------Instagram----------------------")
# i_miner = InstaMiner()
# i_miner.preprocess_posts()

user = t_miner.get_user_tweets()
