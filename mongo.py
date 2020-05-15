from pymongo import MongoClient

class MongoHandler(object):
    client = None
    db = None

    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["web_mining"]

    def store_to_collection(self, dictionary, col_name):
        collection = self.db[col_name]
        collection.insert_one(dictionary)

    def retrieve_from_collection(self, col_name):
        collection = self.db[col_name]
        return collection.find()

    def get_with_id(self, col_name, dict_wit_id):
        collection = self.db[col_name]
        return collection.find(dict_wit_id)
