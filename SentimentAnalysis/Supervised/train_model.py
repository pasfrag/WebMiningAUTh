from SentimentAnalysis.Supervised.feature_extraction import vectorize_test_dataframe, vectorize_train_dataframe, \
    preprocces_dataframe, preprocces_mongo_tweets_and_posts
import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from mongo import MongoHandler

# Connect to mongo to retrieve tweets and insta post
mongo_connect = MongoHandler()
tweets = mongo_connect.retrieve_from_collection("twitter")  # Retrieve tweets from collection
insta = mongo_connect.retrieve_from_collection("instagram")  # Retrieve insta posts from collection

# Read tweets from mongo
mongo_tweets = pd.DataFrame(list(insta))
mongo_tweets = mongo_tweets.sample(frac=1, random_state=1)
mongo_tweets = mongo_tweets[['test']]

# Read instagram posts from mongo
# mongo_insta = pd.DataFrame(list(insta))
# mongo_insta = mongo_tweets.sample(frac=1, random_state=1)
# mongo_insta = mongo_tweets[['caption']]

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
test_emotions = preprocces_mongo_tweets_and_posts(mongo_tweets)

# Define feature columns
X_train = train_emotions.iloc[:, 6:]
X_val = val_emotions.iloc[:, 6:]
X_test = test_emotions.iloc[:, 1:]
# X_test = test_emotions.iloc[:, 2:] For instagram posts

# Define target labels
y_train = train_emotions.iloc[:, :6]
y_val = val_emotions.iloc[:, :6]

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
print(y_train, y_val)

# Define strategy and classifier
nb = OneVsRestClassifier(MultinomialNB(), n_jobs=1)
# dt = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=1)  # Decision Trees
# lr = OneVsRestClassifier((LogisticRegression(solver='lbfgs'))) # Logistic Regression
# rf = OneVsRestClassifier(RandomForestClassifier(max_depth=4, random_state=0)) #Random Forest

# Other strategies used for Multilabel Data
# nb = BinaryRelevance(MultinomialNB())
# nb = LabelPowerset(MultinomialNB())
# nb = ClassifierChain(MultinomialNB())

# Save predictions to dataframe
predictions = pd.DataFrame(columns=emotions_categories)

for emotion in emotions_categories:
    nb.fit(X_train, y_train[emotion])
    y_pred = nb.predict(X_test)
    predictions[emotion] = y_pred
    # Print metrics to evaluate the model
    # print('For emotion {}'.format(emotion))
    # print('Accuracy is {}'.format(accuracy_score(y_val[emotion], y_pred)))
    # print('F1 is {}'.format(f1_score(y_val[emotion], y_pred, average='micro')))
    # print('Hamming loss is {}'.format(f1_score(y_val[emotion], y_pred)))

# Concatenate text and predictions
mongo_tweets.reset_index(drop=True, inplace=True)
predictions.reset_index(drop=True, inplace=True)
predictions = pd.concat([predictions, mongo_tweets], axis=1)

# Connect to database and save the predictions for each tweet/insta post
client = mongo_connect.client
database = client['supervised_sentiment_analysis']
collection = database['twitter_emotions']
predictions.reset_index(inplace=True)
predictions_dict = predictions.to_dict("records")

# Insert predictions to collection
collection.insert_many(predictions_dict)