from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from SentimentAnalysis.Supervised.feature_extraction_helpers import cv, tfidf
import pandas as pd
from mongo import MongoHandler
import numpy as np

# Define vocabulary size
NUM_WORDS = 20000
# Define maximum length of input vetor size
MAX_LEN = 50
# Define dummy tokenizer
tokenizer = Tokenizer(num_words=NUM_WORDS, split=",", lower=True)


def read_tweets_and_instaposts(collection):
    mongo_connect = MongoHandler()
    tweets = mongo_connect.retrieve_from_collection(collection)  # Retrieve tweets from collection
    tweets = pd.DataFrame(list(tweets))
    tweets = tweets.sample(frac=1, random_state=1)
    tweets = tweets[
        ['_id', 'text', 'negative', 'positive:', 's_anger', 's_disgust', 's_fear', 's_joy', 's_sadness', 's_surprise',
         'favorites']]  # Keep emotion analysis features and y
    tweets.rename(columns={"positive:": "positive"}, inplace=True)
    # print(tweets.columns)
    # print(tweets)
    # Handle like prediction as a classification problem by creating 4 different bins for like prediction (0-1, 2-5, 6-10, 11+)
    y = tweets[['favorites']]
    y['favorites'] = np.where(y['favorites'].between(0, 1), 0, y['favorites'])
    y['favorites'] = np.where(y['favorites'].between(2, 5), 1, y['favorites'])
    y['favorites'] = np.where(y['favorites'].between(6, 10), 2, y['favorites'])
    y['favorites'] = np.where(y['favorites'] > 10, 3, y['favorites'])
    return tweets, y


def split_and_preprocces(tweets, labels):
    tokenizer.fit_on_texts(tweets['text'].values)
    word_index = tokenizer.word_index
    print('Dataset contains %s unique tokens.' % len(word_index))
    X = tokenizer.texts_to_sequences(tweets['text'].values)
    X = pad_sequences(X, maxlen=MAX_LEN)
    Y = pd.get_dummies(labels['favorites']).values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    return X_train, X_test, Y_train, Y_test


def vectorize_train_dataframe(df):
    # Vectorize the preprocced tweet text
    # text_cv = cv.fit_transform(df['text']).toarray()
    # df2 = pd.DataFrame(text_cv)
    text_tfidf = tfidf.fit_transform(df['text']).toarray()
    df2 = pd.DataFrame(text_tfidf)
    # Reset index to avoid troubles in concatenation
    df.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df_concat = pd.concat([df, df2], axis=1)
    # Select only the columns we need
    return df_concat


def vectorize_test_dataframe(df):
    # Vectorize the preprocced tweet text
    # text_cv = cv.transform(df['text']).toarray()
    # df2 = pd.DataFrame(text_cv)
    text_tfidf = tfidf.transform(df['text']).toarray()
    df2 = pd.DataFrame(text_tfidf)
    df.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df_concat = pd.concat([df, df2], axis=1)
    return df_concat
