from sklearn.model_selection import train_test_split
from LikePrediction.NLPbased.like_prediction_vectorizer import vectorize_train_dataframe, vectorize_test_dataframe
from mongo import MongoHandler
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding, Flatten, Softmax

mongo_connect = MongoHandler()
tweets = mongo_connect.retrieve_from_collection("twitter_final")  # Retrieve tweets from collection

mongo_tweets = pd.DataFrame(list(tweets))
mongo_tweets = mongo_tweets.sample(frac=1, random_state=1)
mongo_tweets = mongo_tweets[
    ['_id', 'text', 'negative', 'positive:', 's_anger', 's_disgust', 's_fear', 's_joy', 's_sadness', 's_surprise',
     'favorites']]  # Keep emotion analysis features and labels
mongo_tweets.rename(columns={"positive:": "positive"}, inplace=True)
# print(mongo_tweets.columns)
# print(mongo_tweets)
# Handle like prediction as a classification problem by creating 4 different bins for like prediction (0-1, 2-5, 6-10, 11+)
labels = mongo_tweets[['favorites']]
labels['favorites'] = np.where(labels['favorites'].between(0, 1), 0, labels['favorites'])
labels['favorites'] = np.where(labels['favorites'].between(2, 5), 1, labels['favorites'])
labels['favorites'] = np.where(labels['favorites'].between(6, 10), 2, labels['favorites'])
labels['favorites'] = np.where(labels['favorites'] > 10, 3, labels['favorites'])

# print(labels.favorites.value_counts())

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(mongo_tweets, labels, test_size=0.3, train_size=0.7)

# print(X_train, X_train.shape)
# print(X_test, X_test.shape)
# print(y_train, y_train.shape)
# print(y_test, y_test.shape)

# Vectorize the dataset (tfidf or count vectorizer)
X_train = vectorize_train_dataframe(X_train)
X_test = vectorize_test_dataframe(X_test)

# Keep only the features
X_train = X_train.iloc[:, 2:]
X_test = X_test.iloc[:, 2:]
# print(X_train.shape[1])

# Create classification machine learning models
nb = MultinomialNB(alpha=1)
# nb = GaussianNB()
# dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
# rf = RandomForestClassifier(max_depth=10, random_state=0)

# Train the model and make predictions
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Evaluate the model with different metrics
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))
