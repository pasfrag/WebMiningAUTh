from pymongo import MongoClient

def create_mongo_collection(col_name, dict):
    client = MongoClient("mongodb://localhost:27017/")

    db = client["web_mining"]

    colection = db[col_name]

    colection.insert_one(dict)
