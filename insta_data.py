import json
import datetime
from secret_keys import insta_username, insta_password
from instaloader import Instaloader
from mongo import MongoHandler
from langdetect import detect
from preprocessing_functions import preprocess_text, get_hashtags, get_mentions


class InstaMiner(object):

    loader = None
    connection = None

    def __init__(self):
        loader = Instaloader(
            download_pictures=False, download_video_thumbnails=False, download_videos=False, compress_json=False,
            sleep=True
        )
        # loader.login(insta_username, insta_password)
        self.connection = MongoHandler()

    # Retrieve new posts from Instagram
    def get_new_posts(self):

        for post in self.loader.get_hashtag_posts('climatechange'):
            # Keeping only necessary k-v
            # print(post._node)
            new_post = dict()
            new_post["_id"] = post._node.pop("id")
            print(json.dumps(post._node, indent=4, sort_keys=True))
            try:
                new_post["caption"] = post._node["edge_media_to_caption"]["edges"][0]["node"]["text"]
            except:
                new_post["caption"] = None
            try:
                new_post["location"] = post._node["location"]
            except:
                new_post["location"] = None
            try:
                new_post["shortcode"] = post._node["shortcode"]
            except:
                new_post["shortcode"] = None
            try:
                new_post["timestamp"] = post._node["taken_at_timestamp"]
            except:
                new_post["timestamp"] = None
            try:
                new_post["liked_by"] = post._node["edge_liked_by"]["count"]
            except:
                new_post["liked_by"] = None
            try:
                new_post["user_id"] = post._node["owner"]["id"]
            except:
                new_post["user_id"] = None
            try:
                new_post["username"] = post._node["owner"]["username"]
            except:
                new_post["username"] = None
            try:
                new_post["is_verified"] = post._node["owner"]["is_verified"]
            except:
                new_post["is_verified"] = None
            try:
                new_post["is_private"] = post._node["owner"]["is_private"]
            except:
                new_post["is_private"] = None

            self.connection.store_to_collection(new_post, "instagram")

    # Preprocesses instagram posts
    def preprocess_posts(self):
        posts = self.connection.retrieve_from_collection("instagram")

        count = 0
        for post in posts:
            if post["caption"]:
                try:
                    if not len(post["caption"].split()) < 5 and detect(post["caption"]) == 'en':
                        new_post = dict()
                        new_post["_id"] = int(post["_id"])
                        new_post["hashtags"] = get_hashtags(post["caption"])
                        new_post["mentions"] = get_mentions(post["caption"])
                        new_post["caption"] = preprocess_text(post["caption"])
                        new_post["shortcode"] = post["shortcode"]
                        new_post["user_id"] = post["user_id"]
                        new_post["likes"] = post["liked_by"]
                        new_post["created_at"] = datetime.datetime.fromtimestamp(post["timestamp"]).strftime("%Y-%m-%d")
                        self.connection.store_to_collection(new_post, "instagram_new")
                        count += 1
                except:
                    print(1)

        print("--------------------------------")
        print(f"Number of found: {count}")
