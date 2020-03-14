from instaloader import Instaloader
import mongo

loader = Instaloader(download_pictures=False, download_video_thumbnails=False, download_videos=False, compress_json=False)
loader.login("username", "password")

for post in loader.get_hashtag_posts('climatechange'):

    # Keeping only necessary k-v
    print(post._node)
    new_post = dict()
    new_post["_id"] = post._node.pop("id")
    new_post["caption"] = post._node["edge_media_to_caption"]["edges"][0]["node"]["text"]
    try:
        new_post["location"] = post._node["location"]
    except:
        new_post["location"] = None
    new_post["shortcode"] = post._node["shortcode"]
    new_post["timestamp"] = post._node["taken_at_timestamp"]
    # new_post["tracking_token"] = post._node["tracking_token"]
    new_post["liked_by"] = post._node["edge_liked_by"]["count"]
    # new_post["edge_media_preview_like"] = post._node["edge_media_preview_like"]["count"]
    new_post["user_id"] = post._node["owner"]["id"]
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

    mongo.create_mongo_collection("instagram", new_post)
    print("Hi")
