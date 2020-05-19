import pandas as pd
from networkx.drawing.tests.test_pylab import plt
from sklearn.ensemble import RandomForestClassifier
from SentimentAnalysis.Supervised.feature_extraction import vectorize_test_dataframe, vectorize_train_dataframe, \
    preprocces_dataframe
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from mongo import MongoHandler
import numpy as np
mongo_connect = MongoHandler()
from SentimentAnalysis.Supervised.feature_extraction import preprocces_mongo_tweets

tweets = mongo_connect.retrieve_from_collection("twitter")

# Read tweets from mongo
mongo_tweets = pd.DataFrame(list(tweets))
mongo_tweets = mongo_tweets.sample(frac=1, random_state=1)
mongo_tweets = mongo_tweets[['text']]


# Read the dataset
data = pd.read_csv("2018-11 emotions-classification-train.txt", sep="\t")

# Preprocces the dataset
emotions = preprocces_dataframe(data)
emotions_categories = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

# Split the dataset to train and validation set (for test set we will use the loaded tweets)
train_dataset, val_dataset = train_test_split(emotions, test_size=0.3, train_size=0.7)

# Vectorize tweet text
train_emotions = vectorize_train_dataframe(train_dataset)
val_emotions = vectorize_test_dataframe(val_dataset)
test_emotions = preprocces_mongo_tweets(mongo_tweets)
X_test = test_emotions.iloc[:, 1:]
print(X_test)
# Define feature columns
X_train = train_emotions.iloc[:, 6:]
X_val = val_emotions.iloc[:, 6:]
# Define target labels
y_train = train_emotions.iloc[:, :6]
y_val = val_emotions.iloc[:, :6]

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
print(y_train, y_val)

# Create models
nb = OneVsRestClassifier(MultinomialNB(), n_jobs=1)
dt = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=1)
lr = OneVsRestClassifier((LogisticRegression(solver='lbfgs')))
rf = RandomForestClassifier(max_depth=4, random_state=0)

# Save predictions to dataframe
predictions = pd.DataFrame(columns=emotions_categories)

# for model in [nb, lr, dt, rf]:
for emotion in emotions_categories:
    nb.fit(X_train, y_train[emotion])
    y_pred = nb.predict(X_test)
    predictions[emotion] = y_pred
    # print('For emotion {}'.format(emotion))
    # print('Accuracy is {}'.format(accuracy_score(y_val[emotion], y_pred)))
    # print('F1 is {}'.format(f1_score(y_val[emotion], y_pred, average='micro')))
    # print('Hamming loss is {}'.format(f1_score(y_val[emotion], y_pred)))


print(predictions)