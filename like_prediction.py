import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, tree, linear_model, naive_bayes, linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.svm import LinearSVC
from profiling import k_fold_cv

from mongo import MongoHandler
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# function for black box model interpretability with the method of feature importance
def tree_feature_importance(tree_model,columns, x):
    weights = tree_model.feature_importances_
    model_weights = pd.DataFrame({'features': list(columns), 'weights': list(weights)})
    model_weights = model_weights.sort_values(by='weights', ascending=False)
    plt.figure(num=None, figsize=(12, 12), dpi=150, facecolor='w', edgecolor='k')
    sns.barplot(x="weights", y="features", data=model_weights)
    plt.xticks(rotation=90)
    plt.show()

def evaluation(y_test, y_predict):
    print("Accuracy Score: %.4f" % metrics.accuracy_score(y_test, y_predict))
    print("Balanced accuracy Score: %.4f" % metrics.balanced_accuracy_score(y_test, y_predict))
    print("Precision: %.4f" % metrics.precision_score(y_test, y_predict, average="macro"))
    print("Recall: %.4f" % metrics.recall_score(y_test, y_predict, average="macro"))
    print(" F1: %.4f" % metrics.f1_score(y_test, y_predict, average="macro"))
    print(metrics.confusion_matrix(y_test, y_predict))

# Base model for Like prediction. Uses the amount of followers, friends, statuses and others as features
def base_model():
    mongo_connect = MongoHandler()
    like_tweets = mongo_connect.retrieve_from_collection("twitter_new")
    df = pd.DataFrame(list(like_tweets))

    # text = df['text']
    # df = df.drop(['user_name','user_location','hashtags','mentions','created_at'],axis=1)

    # Column / Feature selection
    base = df[['user_followers', 'user_friends', 'user_favourites',
               'user_months', 'user_statuses', 'user_verified',
               'retweets']]
    per_month = round((base['user_statuses'] + 1) / (base['user_months'] + 1), 2)
    per_month = pd.DataFrame(per_month)
    per_month.columns = ['tweet_per_month']
    base = pd.concat([base, per_month], axis=1)
    target = df['favorites']
    # base = base[['user_followers', 'retweets', 'user_favourites', 'user_statuses']]
    columns = base.columns.values.tolist()

    # Tranform the problem of regression into a multi-class classification. Classes: zero, low, medium, high
    for i in range(len(target)):
        if 0 < target[i] < 6:
            target[i] = 1
        elif 5 < target[i] < 11:
            target[i] = 2
        elif target[i] >= 11:
            target[i] = 3

    # target.hist()
    # plt.show()
    nm1 = NearMiss(version=1) # Under-sampling technique
    base, target = nm1.fit_resample(base, target)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(base, target, random_state=1, test_size=0.3)

    # testing different ML algorithms
    # model = tree.DecisionTreeClassifier(criterion="entropy", random_state=5)  # class_weight="balanced")
    # model = linear_model.LogisticRegression(solver="lbfgs", random_state=5)
    # model = naive_bayes.MultinomialNB()
    # model = tree.ExtraTreeClassifier(random_state=5)
    # model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=10), n_estimators=1000, random_state=5)
    model = RandomForestClassifier(n_estimators=500, random_state=5)

    k_fold_cv(model, x_train, y_train, False)

    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    evaluation(y_test, y_predict)

    # feature importance / interpretability
    new_y_train = model.predict(x_train)
    tree_model = tree.DecisionTreeClassifier(criterion="entropy", random_state=5)  # class_weight="balanced")
    tree_model.fit(x_train, new_y_train)
    tree_feature_importance(tree_model, columns, x_train)  # calls function for interpretable ML

