import pickle
from sklearn.model_selection import train_test_split
from LikePrediction.NLPbased.like_prediction_vectorizer import vectorize_train_dataframe, vectorize_test_dataframe,read_tweets_and_instaposts
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

coll_name = 'twitter_final'
mongo_tweets, labels = read_tweets_and_instaposts(coll_name)

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(mongo_tweets, labels, test_size=0.3, train_size=0.7)

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
# print(precision_score(y_test, y_pred, average="macro"))
# print(recall_score(y_test, y_pred, average="macro"))

# # Save model to pickle
# filename = 'models/naive_bayes.pkl'
# with open(filename, 'wb') as file:
#     pickle.dump(nb, file)