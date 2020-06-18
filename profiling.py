import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, model_selection, svm
from sklearn.linear_model import LogisticRegression
from mongo import MongoHandler
from secret_keys import insta_username, insta_password
import pickle
import lexicons
from nltk.corpus import words
from collections import Counter
from geotext import GeoText
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
# from visualization import word_cloud


mongo_connect = MongoHandler()


def dummy(doc):
    return doc


# Opinion mining function. Classifies Tweeter users into "Deniers, Non-Deniers or Uncertain"
# regarding the issue of Climate Change denialism
def predict_denier_profiles():
    # svm_load = pickle.load(open('final_models/final_svm.pickle', 'rb'))  # unused SVM model

    # loading the trained Logistic Regression model
    LR = pickle.load(open('final_models/final_LR.pickle', 'rb'))

    # loading the pre-trained Tf-Idf Lexicon
    tfidf_load = pickle.load(open('final_models/final_tfidf.pickle', 'rb'))

    # Loading new, previously unseen twitter profiles
    new_profiles = mongo_connect.retrieve_from_collection("twitter_profiles_1K")
    df = pd.DataFrame(list(new_profiles))
    df.groupby('user_name')
    collect_profiles = dict()  # collection for each user

    # Groups all the tweets by each user
    count_total = count_den = count_non = count_unc = 0
    for user in df.groupby('user_name'):
        name = user[0]
        text = user[1]['text']
        sent = user[1]['sentiment']
        subj = user[1]['subjectivity']

        # average scores of sentiment and subjectivity
        subj = round(subj.sum() / len(subj), 2)
        sent = round(sent.sum() / len(sent), 2)

        collect_profiles["_id"] = count_total
        count_total += 1
        collect_profiles["name"] = name

        # adding all separate tweets into one long list
        full_text = []
        for t in text:
            full_text.append(t)

        # transform textual data with the use of the pre-trained lexicon
        x = tfidf_load.transform(text)
        x_test = pd.DataFrame(x.todense())

        # make prediction based on the pre-trained Logistic Regression model
        y_predict = LR.predict(x_test)

        count0 = sum(y_predict == 0)
        count1 = sum(y_predict == 1)

        # calculating a threshold for each class
        div1 = (count1 + 0.001) / (count0 + 0.001)
        div0 = (count0 + 0.001) / (count1 + 0.001)

        if div1 >= 2:
            collect_profiles["prediction"] = "DENIER"
            count_den += 1
        elif div0 > 1.5:
            collect_profiles["prediction"] = "NON-DENIER"
            count_non += 1
        else:
            collect_profiles["prediction"] = "UNCERTAIN"
            count_unc += 1

        # storing a dictionary collection for each user back to the database
        collect_profiles["div1"] = round(div1, 2)
        collect_profiles["div0"] = round(div0, 2)
        collect_profiles["tweet count"] = user[1].__len__()
        collect_profiles["subjectivity"] = subj
        collect_profiles["sentiment"] = sent
        collect_profiles["text"] = full_text

        collect_profiles["anger"] = user[1]['anger'].sum() / len(user[1]['anger'])
        collect_profiles["anticipation"] = user[1]['anticipation'].sum() / len(user[1]['anticipation'])
        collect_profiles["disgust"] = user[1]['disgust'].sum() / len(user[1]['disgust'])
        collect_profiles["joy"] = user[1]['joy'].sum() / len(user[1]['joy'])
        collect_profiles["sadness"] = user[1]['sadness'].sum() / len(user[1]['sadness'])
        collect_profiles["surprise"] = user[1]['surprise'].sum() / len(user[1]['surprise'])
        collect_profiles["trust"] = user[1]['trust'].sum() / len(user[1]['trust'])

        mongo_connect.store_to_collection(collect_profiles, "twitter_profiles_final")

    print("Found: ", count_den, "deniers, ", count_non, "non-deniers and", count_unc, "uncertain cases")


# gets the unique users from our initial twitter dataset and filters out the users used for the ML training
def get_user_names():
    df = pd.read_csv('data/user_names.csv')
    # print (pd.DataFrame(users.values.tolist()).stack().value_counts())
    users = df.groupby('user_screen_name').agg(['nunique']).reset_index(drop=False)
    users = users.sample(frac=1, random_state=1)
    profiles = []
    for user in users['user_screen_name']:
        if user not in lexicons.already_loaded:
            profiles.append(user)
    return profiles


# Stratified 10 fold cross validation
def k_fold_cv(model, x_train, y_train, version):
    accuracy_model = []
    precision_model = []
    recall_model = []
    f1_model = []
    total_cost = []
    kf = model_selection.StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(x_train, y_train):
        if version:
            x_tr, x_ts = x_train.iloc[train_index], x_train.iloc[test_index]
            y_tr, y_ts = y_train.iloc[train_index], y_train.iloc[test_index]
        else:
            x_tr, x_ts = x_train[train_index], x_train[test_index]
            y_tr, y_ts = y_train[train_index], y_train[test_index]

        model = model.fit(x_tr, y_tr)
        y_predict = model.predict(x_ts)
        accuracy_model.append(metrics.accuracy_score(y_ts, y_predict, normalize=True) * 100)
        precision_model.append(metrics.precision_score(y_ts, y_predict, average="macro") * 100)
        recall_model.append(metrics.recall_score(y_ts, y_predict, average="macro") * 100)
        f1_model.append(metrics.f1_score(y_ts, y_predict, average="macro") * 100)
    print(round(np.mean(accuracy_model), 4))
    print(round(np.mean(precision_model), 4))
    print(round(np.mean(recall_model), 4))
    print(round(np.mean(f1_model), 4))
    print(np.mean(total_cost))

    print("K Fold Accuracy scores:", accuracy_model)
    print("Mean KFold Accuracy: ", round(np.mean(accuracy_model), 4))
    print("K Fold Precision scores:", precision_model)
    print("Mean K Fold Precision scores:", round(np.mean(precision_model), 4))
    print("K Fold Recall scores:", recall_model)
    print("Mean K Fold Recall scores:", round(np.mean(recall_model), 4))
    print("K Fold F1 scores:", f1_model)
    print("Mean KFold F1: ", round(np.mean(f1_model), 4))


# function that trains a Machine Learning algorithm to identify Climate Change Deniers and Non-Deniers
# based on textual information in the form of tf-idf
def train_profiling_model():
    profiles = mongo_connect.retrieve_from_collection("twitter_profiles")

    df = pd.DataFrame(list(profiles))
    df = df.sample(frac=1, random_state=1)
    text = df['text']
    y = df['label']

    tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None)

    # training and testing the data on Logistic Regression and SVM
    def test_model():
        x_train, x_test, y_train, y_test = model_selection.train_test_split(text, y, random_state=1, test_size=0.1)
        tfidf.fit(x_train)
        text_train = tfidf.transform(x_train)
        x_train = pd.DataFrame(text_train.todense())

        # model = MultinomialNB()
        # model = svm.SVC(gamma=0.2, max_iter=1000)
        # model = RandomForestClassifier(random_state=1)
        # model = DecisionTreeClassifier(random_state=1)
        model = LogisticRegression(random_state=1, solver='lbfgs')
        k_fold_cv(model, x_train, y_train, True)

        model.fit(x_train, y_train)
        text_test = tfidf.transform(x_test)
        x_test = pd.DataFrame(text_test.todense())

        y_predict = model.predict(x_test)
        print("Accuracy Score: %.4f" % metrics.accuracy_score(y_test, y_predict))
        print("Precision: %.4f" % metrics.precision_score(y_test, y_predict, average="macro"))
        print("Recall: %.4f" % metrics.recall_score(y_test, y_predict, average="macro"))
        print(" F1: %.4f" % metrics.f1_score(y_test, y_predict, average="macro"))
        print(metrics.confusion_matrix(y_test, y_predict))

    # retrains the ML model on the whole dataset and finalises it
    def export_final_model():
        tfidf.fit(text)
        pickle.dump(tfidf, open("final_models/final_tfidf.pickle", "wb"))
        text_tf = tfidf.transform(text)
        x = pd.DataFrame(text_tf.todense())
        model = LogisticRegression(random_state=0, solver='lbfgs')
        # model = svm.SVC(gamma=0.2, max_iter=1000)

        model.fit(x, y)
        pickle.dump(model, open('final_models/final_LR.pickle', 'wb'))
        # pickle.dump(model2, open('final_models/final_svm.pickle', 'wb'))

    test_model()
    # export_final_model()


# Unused Neural Network model due to low performance

# def test_doc2vec():
#     from keras.optimizers import Adam
#     from keras.models import Sequential
#     from keras.layers import Dense
#     import multiprocessing
#     from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#     profiles = mongo_connect.retrieve_from_collection("twitter_profiles")
#
#     df = pd.DataFrame(list(profiles))
#     df = df.sample(frac=1, random_state=1)
#
#     train, test = model_selection.train_test_split(df, test_size=0.3, random_state=42)
#
#     cores = multiprocessing.cpu_count() / 2
#     train_documents = [TaggedDocument(words=[word for word in tweet["text"]], tags=[tweet["label"]]) for _index, tweet in
#                        train.iterrows()]
#     test_documents = [TaggedDocument(words=[word for word in tweet["text"]], tags=[tweet["label"]]) for _index1, tweet in
#                       test.iterrows()]
#
#     model_docs = Doc2Vec(train_documents, vector_size=200, window=20, min_count=5, workers=cores, epochs=30)
#     model_docs.train(train_documents, total_examples=model_docs.corpus_count, epochs=model_docs.epochs)
#
#     x_train1 = [model_docs.infer_vector(doc.words) for doc in train_documents]
#     x_test1 = [model_docs.infer_vector(doc.words) for doc in test_documents]
#
#     x_train = pd.DataFrame(x_train1)
#     x_test = pd.DataFrame(x_test1)
#
#     model_nn = Sequential()
#     model_nn.add(Dense(512, input_dim=200, activation="sigmoid"))
#     model_nn.add(Dense(256, activation="sigmoid"))
#     model_nn.add(Dense(128, activation="sigmoid"))
#     model_nn.add(Dense(64, activation="sigmoid"))
#     model_nn.add(Dense(2, activation="softmax"))
#
#     optimizer = Adam(lr=0.01)
#     model_nn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#
#     model_nn.fit(x_train, train["label"], epochs=30, batch_size=32)
#
#     y_predict = model_nn.predict(x_test)
#     # y_predict = (y_predict > 0.5)
#
#     print("Accuracy Score: %.4f" % metrics.accuracy_score(test["label"], y_predict))
#     print("Precision: %.4f" % metrics.precision_score(test["label"], y_predict, average="macro"))
#     print("Recall: %.4f" % metrics.recall_score(test["label"], y_predict, average="macro"))
#     print(" F1: %.4f" % metrics.f1_score(test["label"], y_predict, average="macro"))
#     print(metrics.confusion_matrix(test["label"], y_predict))


# function for downloading new instagram profiles (not individual posts like in insta_data.py)
def insta_profiles():
    import instaloader
    L = instaloader.Instaloader(
        download_pictures=False, download_video_thumbnails=False, download_videos=False, compress_json=False,
        sleep=True)
    L.login(insta_username, insta_password)

    df = pd.read_csv('data/insta_users.csv')
    user_id = df.groupby('user_id').agg(['nunique']).reset_index(drop=False)
    user_id = user_id['user_id']

    for i in range(0, 501):
        user_dict = dict()
        print(user_id[i])
        profile = instaloader.Profile.from_id(L.context, user_id[i])

        limit_posts = 0
        user_dict['_id'] = int(user_id[i])
        hashtags = []
        for post in profile.get_posts():
            for tag in post.caption_hashtags:
                hashtags.append(tag)
            limit_posts += 1
            if limit_posts >= 50:
                break
        user_dict['hashtags'] = hashtags
        mongo_connect.store_to_collection(user_dict, "ig_users")


# detecting a user's interests based on their most used hashtags
def user_interests():
    ig_users = mongo_connect.retrieve_from_collection("ig_users")
    df = pd.DataFrame(list(ig_users))
    hashtags = df['hashtags']
    english_dict = words.words()
    remove = ['covid','virus','pandemic','corona','quarantine','isolation','repost','giveaway','week','weekend']

    all_tags = [] # for all the users
    for tag in hashtags: # [0:100]:
        en_tag = [] # for each user
        for word in tag:
            if word in english_dict and word not in remove and not GeoText(word).country_mentions: #  keep only words that exist in the english dictionary and remove irrelevant words from list: remove and country names
                en_tag.append(word)
                all_tags.append(word)
        counts = Counter(en_tag)
        if len(counts) > 0:
            interests = counts.most_common(5) # keep the five most common hashtags for each user
            print(interests)

    total_count = Counter(all_tags)
    # word_cloud(all_tags)
    print(total_count.most_common(20))
    return all_tags, total_count
