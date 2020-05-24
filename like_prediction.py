import pandas as pd
from sklearn import model_selection, metrics, tree,linear_model, naive_bayes
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from mongo import MongoHandler
import seaborn as sns

def tree_feature_importance(tree_model, x):
    weights = tree_model.feature_importances_
    model_weights = pd.DataFrame({'features': list(x.columns), 'weights': list(weights)})
    model_weights = model_weights.sort_values(by='weights', ascending=False)
    plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    sns.barplot(x="weights", y="features", data=model_weights)
    plt.xticks(rotation=90)
    plt.show()

# conda install -c conda-forge imbalanced-learn

mongo_connect = MongoHandler()
like_tweets = mongo_connect.retrieve_from_collection("twitter_new")
df = pd.DataFrame(list(like_tweets))

target = df['favorites']
text = df['text']
df = df.drop(['user_name','user_location','hashtags','mentions','created_at'],axis=1)
base = df[['user_followers','user_friends','user_favourites',
           'user_months','user_statuses','user_verified',
           'retweets']]
per_month = round((base['user_statuses']+1)/(base['user_months']+1),2)
per_month = pd.DataFrame(per_month)
per_month.columns = ['tweet_per_month']
base = pd.concat([base, per_month], axis=1)
base = base[['user_followers', 'retweets', 'user_favourites', 'user_statuses']]

for i in range(len(target)):
    if 0 < target[i] < 6:
        target[i] = 1
    elif 5 < target[i] < 11:
        target[i] = 2
    elif target[i] >= 11:
        target[i] = 3

target.hist()
plt.show()

# x_train, x_test, y_train, y_test = model_selection.train_test_split(base, target, random_state=1, test_size=0.3)

# sm = SMOTE(random_state=2)
# x_train, y_train = sm.fit_sample(x_train, y_train)

nm1 = NearMiss(version=1)
# x_train, y_train = nm1.fit_resample(x_train, y_train)
base, target = nm1.fit_resample(base, target)

x_train, x_test, y_train, y_test = model_selection.train_test_split(base, target, random_state=1, test_size=0.3)


# model = tree.DecisionTreeClassifier(criterion="entropy", random_state=5)  # class_weight="balanced")
# model = linear_model.LogisticRegression(multi_class="multinomial", random_state=5)
# model = naive_bayes.MultinomialNB(random_state=5)
# model = LinearSVC(random_state=5)
# model = tree.ExtraTreeClassifier(random_state=5)
model = RandomForestClassifier(n_estimators=500, random_state=5, n_jobs=4)
# model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=10), n_estimators=1000, random_state=5)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print("Accuracy Score: %.4f" % metrics.accuracy_score(y_test, y_predict))
print("Balanced accuracy Score: %.4f" % metrics.balanced_accuracy_score(y_test, y_predict))
print("Precision: %.4f" % metrics.precision_score(y_test, y_predict, average="macro"))
print("Recall: %.4f" % metrics.recall_score(y_test, y_predict, average="macro"))
print(" F1: %.4f" % metrics.f1_score(y_test, y_predict, average="macro"))
print(metrics.confusion_matrix(y_test, y_predict))

new_y_train = model.predict(x_train)

tree_model = tree.DecisionTreeClassifier(criterion="entropy", random_state=5)  # class_weight="balanced")
tree_model.fit(x_train, new_y_train)
tree_feature_importance(tree_model, x_train)
