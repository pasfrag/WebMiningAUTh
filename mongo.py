from pymongo import MongoClient

def create_mongo_collection(col_name, dict):
    client = MongoClient("mongodb://localhost:27017/")

    db = client["web_mining"]

    collection = db[col_name]

    collection.insert_one(dict)
