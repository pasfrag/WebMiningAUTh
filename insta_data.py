import json
from secret_keys import insta_string
from instaloader import Instaloader
import mongo

loader = Instaloader(
    download_pictures=False, download_video_thumbnails=False, download_videos=False, compress_json=False, sleep=True
)
loader.login(insta_string)

for post in loader.get_hashtag_posts('climatechange'):

    # Keeping only necessary k-v
    # print(post._node)
    # new_post = dict()
    # new_post["_id"] = post._node.pop("id")
    print(json.dumps(post._node, indent=4, sort_keys=True))
    # try:
    #     new_post["caption"] = post._node["edge_media_to_caption"]["edges"][0]["node"]["text"]
    # except:
    #     new_post["caption"] = None
    # try:
    #     new_post["location"] = post._node["location"]
    # except:
    #     new_post["location"] = None
    # try:
    #     new_post["shortcode"] = post._node["shortcode"]
    # except:
    #     new_post["shortcode"] = None
    # try:
    #     new_post["timestamp"] = post._node["taken_at_timestamp"]
    # except:
    #     new_post["timestamp"] = None
    # try:
    #     new_post["liked_by"] = post._node["edge_liked_by"]["count"]
    # except:
    #     new_post["liked_by"] = None
    # try:
    #     new_post["user_id"] = post._node["owner"]["id"]
    # except:
    #     new_post["user_id"] = None
    # try:
    #     new_post["username"] = post._node["owner"]["username"]
    # except:
    #     new_post["username"] = None
    # try:
    #     new_post["is_verified"] = post._node["owner"]["is_verified"]
    # except:
    #     new_post["is_verified"] = None
    # try:
    #     new_post["is_private"] = post._node["owner"]["is_private"]
    # except:
    #     new_post["is_private"] = None

    # mongo.create_mongo_collection("instagram", new_post)
    # print("Hi")
