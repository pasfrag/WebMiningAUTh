import pandas as pd
from SentimentAnalysis.Supervised.feature_extraction import vectorize_test_dataframe, vectorize_train_dataframe, preprocces_dataframe
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

# Read the dataset
data = pd.read_csv("2018-11 emotions-classification-train.txt", sep="\t")

# Preprocces the dataset
emotions = preprocces_dataframe(data)

# Split the dataset to train and validation set (for test set we will use the loaded tweets)
train_dataset, val_dataset = train_test_split(emotions, test_size=0.4, train_size=0.6)

# Vectorize tweet text
train_emotions = vectorize_train_dataframe(train_dataset)
val_emotions = vectorize_test_dataframe(val_dataset)

# Define feature columns
X_train = train_emotions.iloc[:,6:]
X_val = val_emotions.iloc[:, 6:]
# Define target labels
y_train = train_emotions.iloc[:, :6]
y_val = val_emotions.iloc[:, :6]

# Create and train multinomiaL naive bayes classifier
nb = OneVsRestClassifier(GaussianNB())
# Create and train logistic regression
# lrg = OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
# Create and train multinomiaL naive bayes classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_val)

# Calculate accuracy and f1
print("Accuracy is", accuracy_score(y_val, y_pred))
print("F1 score for each clas: ", f1_score(y_val, y_pred, average=None))
print("Hamming loss is: ", hamming_loss(y_val, y_pred))