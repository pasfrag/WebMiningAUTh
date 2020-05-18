import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, model_selection, svm
from sklearn.linear_model import LogisticRegression
from mongo import MongoHandler
import pickle

mongo_connect = MongoHandler()
profiles = mongo_connect.retrieve_from_collection("twitter_profiles")

df = pd.DataFrame(list(profiles))
df = df.sample(frac=1, random_state=1)
text = df['text']
Y = df['label']

def dummy(doc):
    return doc


tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None)
x_train, x_test, y_train, y_test = model_selection.train_test_split(text, Y, random_state=1, test_size=0.1)
tfidf.fit(x_train)
text_train = tfidf.transform(x_train)
pickle.dump(tfidf, open("tfidf.pickle", "wb"))
x_train = pd.DataFrame(text_train.todense())

text_test = tfidf.transform(x_test)
x_test = pd.DataFrame(text_test.todense())

# model = LogisticRegression(random_state=0, solver='lbfgs')
model = svm.SVC(max_iter=1000)
model.fit(x_train, y_train)
pickle.dump(model, open('svm.pickle', 'wb'))

# svm_load = pickle.load(open('svm.pickle', 'rb'))
# tfidf_load = pickle.load(open('tfidf.pickle', 'rb'))

text_test = tfidf.transform(x_test)
x_test = pd.DataFrame(text_test.todense())

y_predict = model.predict(x_test)
print("Accuracy Score: %.4f" % metrics.accuracy_score(y_test, y_predict))
print("Precision: %.4f" % metrics.precision_score(y_test, y_predict, average="macro"))
print("Recall: %.4f" % metrics.recall_score(y_test, y_predict, average="macro"))
print(" F1: %.4f" % metrics.f1_score(y_test, y_predict, average="macro"))
print(metrics.confusion_matrix(y_test, y_predict))


