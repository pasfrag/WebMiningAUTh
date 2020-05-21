import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, model_selection, svm
from sklearn.linear_model import LogisticRegression
from mongo import MongoHandler
import pickle
import lexicons
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from keras.optimizers import Adam
# from keras.models import Sequential
# from keras.layers import Dense

mongo_connect = MongoHandler()
# mongo_connect.delete_with_name("new_twitter_profiles","Electroversenet")

def get_user_names():
    df = pd.read_csv('user_names.csv')
    # print (pd.DataFrame(users.values.tolist()).stack().value_counts())
    users = df.groupby('user_screen_name').agg(['nunique']).reset_index(drop=False)
    users = users.sample(frac=1, random_state=1)
    profiles = []
    for user in users['user_screen_name']:
        if user not in lexicons.already_loaded:
            profiles.append(user)
    return profiles


def dummy(doc):
    return doc

def test_new_profiles():

    # svm_load = pickle.load(open('svm.pickle', 'rb'))
    LR = pickle.load(open('LR2.pickle', 'rb'))
    tfidf_load = pickle.load(open('tfidf.pickle', 'rb'))
    new_profiles = mongo_connect.retrieve_from_collection("twitter_profiles_1K") # new_twitter_profiles
    df = pd.DataFrame(list(new_profiles))
    df.groupby('user_name')

    count_den = count_non = count_unc = 0
    for user in df.groupby('user_name'):
        # name = user[1].iloc[1,1]
        text = user[1]['text']
        sent = user[1]['sentiment']
        subj = user[1]['subjectivity']
        subj = round(subj.sum() / len(subj), 2)
        sent = round(sent.sum() / len(sent), 2)

        x = tfidf_load.transform(text)
        x_test = pd.DataFrame(x.todense())
        y_predict = LR.predict(x_test)

        count0 = sum(y_predict == 0)
        count1 = sum(y_predict == 1)

        # print("\nPrediction for user: ", name)
        div1 = (count1 + 0.001) / (count0 + 0.001)
        div0 = (count0 + 0.001) / (count1 + 0.001)

        # print(round(div1,2), round(div0, 2))
        if div1 > 2:
            # print("Class: DENIER")
            count_den += 1
        elif div0 > 2:
            # print("Class: NON-DENIER")
            count_non += 1
        else:
            # print("Class: UNCERTAIN")
            count_unc += 1

        # print("User's sentiment score: ", sent, " and subjectivity score:", subj)

        # print("Score:  %.2f" % div1)
        #
        # if div > 0.7:
        #     print("Class: Denier")
        # elif div < 0.3:
        #     print("Class: Non-denier")
        # else:
        #     print("Uncertain prediction")

        # if name == 'realDonaldTrump':
        #     for i in range(len(y_predict)):
        #         print(text.iloc[i], y_predict[i])
    print("Found: ",count_den,"deniers, ",count_non,"non-deniers and",count_unc,"uncertain cases" )

def train_profiling_model():
    profiles = mongo_connect.retrieve_from_collection("twitter_profiles")

    df = pd.DataFrame(list(profiles))
    df = df.sample(frac=1, random_state=1)
    text = df['text']
    y = df['label']

    tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None)

    def test_model():
        x_train, x_test, y_train, y_test = model_selection.train_test_split(text, y, random_state=1, test_size=0.1)
        tfidf.fit(x_train)
        text_train = tfidf.transform(x_train)
        x_train = pd.DataFrame(text_train.todense())
        model = LogisticRegression(random_state=0, solver='lbfgs')
        # model = svm.SVC(gamma=0.2, max_iter=1000)
        model.fit(x_train, y_train)

        text_test = tfidf.transform(x_test)
        x_test = pd.DataFrame(text_test.todense())

        y_predict = model.predict(x_test)
        print("Accuracy Score: %.4f" % metrics.accuracy_score(y_test, y_predict))
        print("Precision: %.4f" % metrics.precision_score(y_test, y_predict, average="macro"))
        print("Recall: %.4f" % metrics.recall_score(y_test, y_predict, average="macro"))
        print(" F1: %.4f" % metrics.f1_score(y_test, y_predict, average="macro"))
        print(metrics.confusion_matrix(y_test, y_predict))

    def export_final_model():
        tfidf.fit(text)
        pickle.dump(tfidf, open("tfidf.pickle", "wb"))
        text_tf = tfidf.transform(text)
        x = pd.DataFrame(text_tf.todense())
        model = LogisticRegression(random_state=0, solver='lbfgs')
        # model = svm.SVC(gamma=0.2, max_iter=1000)

        model.fit(x, y)
        pickle.dump(model, open('LR2.pickle', 'wb'))
        # pickle.dump(model2, open('svm2.pickle', 'wb'))

    # test_model()
    # export_final_model()

# train_profiling_model()
test_new_profiles()

print("---------------------------------------------------------------------------------")

def test_doc2vec():
    profiles = mongo_connect.retrieve_from_collection("twitter_profiles")

    df = pd.DataFrame(list(profiles))
    df = df.sample(frac=1, random_state=1)

    train, test = model_selection.train_test_split(df, test_size=0.3, random_state=42)

    cores = multiprocessing.cpu_count() / 2
    train_documents = [TaggedDocument(words=[word for word in tweet["text"]], tags=[tweet["label"]]) for _index, tweet in
                       train.iterrows()]
    test_documents = [TaggedDocument(words=[word for word in tweet["text"]], tags=[tweet["label"]]) for _index1, tweet in
                      test.iterrows()]

    model_docs = Doc2Vec(train_documents, vector_size=200, window=20, min_count=5, workers=cores, epochs=30)
    model_docs.train(train_documents, total_examples=model_docs.corpus_count, epochs=model_docs.epochs)

    x_train1 = [model_docs.infer_vector(doc.words) for doc in train_documents]
    x_test1 = [model_docs.infer_vector(doc.words) for doc in test_documents]

    x_train = pd.DataFrame(x_train1)
    x_test = pd.DataFrame(x_test1)

    model_nn = Sequential()
    model_nn.add(Dense(512, input_dim=200, activation="sigmoid"))
    model_nn.add(Dense(256, activation="sigmoid"))
    model_nn.add(Dense(128, activation="sigmoid"))
    model_nn.add(Dense(64, activation="sigmoid"))
    model_nn.add(Dense(2, activation="softmax"))

    optimizer = Adam(lr=0.01)
    model_nn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model_nn.fit(x_train, train["label"], epochs=30, batch_size=32)

    y_predict = model_nn.predict(x_test)
    # y_predict = (y_predict > 0.5)

    print("Accuracy Score: %.4f" % metrics.accuracy_score(test["label"], y_predict))
    print("Precision: %.4f" % metrics.precision_score(test["label"], y_predict, average="macro"))
    print("Recall: %.4f" % metrics.recall_score(test["label"], y_predict, average="macro"))
    print(" F1: %.4f" % metrics.f1_score(test["label"], y_predict, average="macro"))
    print(metrics.confusion_matrix(test["label"], y_predict))

# test_doc2vec()
