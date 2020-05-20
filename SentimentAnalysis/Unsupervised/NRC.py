import os

import pandas as pd
from mongo import MongoHandler

# connection = MongoHandler()

def get_emotions(words_list):
    filepath = "C:\\Users\\pasca\\Desktop\\Projects\\Python_Projects\\WebMiningAUTh\\SentimentAnalysis\\Unsupervised\\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    emolex_df = pd.read_csv(filepath, names=["word", "emotion", "association"], sep='\t')
    emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association')
    word_emotions = emolex_df.iloc[:, 0].tolist()

    anger = anticipation = disgust = fear = joy = negative = positive = sadness = surprise = trust = 0
    for word in words_list:
        if word in word_emotions:
            emotion = emolex_words.loc[word]
            anger += emotion.anger
            anticipation += emotion.anticipation
            disgust += emotion.disgust
            fear += emotion.fear
            joy += emotion.joy
            negative += emotion.negative
            positive += emotion.positive
            sadness += emotion.sadness
            surprise += emotion.surprise
            trust += emotion.trust
    num_of_words = len(words_list)
    anger /= num_of_words
    anticipation /= num_of_words
    disgust /= num_of_words
    fear /= num_of_words
    joy /= num_of_words
    negative /= num_of_words
    positive /= num_of_words
    sadness /= num_of_words
    surprise /= num_of_words
    trust /= num_of_words
    return anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust

# # Variables
# filepath = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
# emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], sep='\t')
# emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association')
#
# word_emotions = emolex_df.iloc[:, 0].tolist()

# tweets = connection.retrieve_from_collection("twitter_new")
#
# for tweet in tweets:
#     # anger = anticipation = disgust = fear = joy = negative = positive = sadness = surprise = trust = 0
#     # for word in tweet['text']:
#     #     if word in word_emotions:
#     #         emotion = emolex_words.loc[word]
#     #         anger += emotion.anger
#     #         anticipation += emotion.anticipation
#     #         disgust += emotion.disgust
#     #         fear += emotion.fear
#     #         joy += emotion.joy
#     #         negative += emotion.negative
#     #         positive += emotion.positive
#     #         sadness += emotion.sadness
#     #         surprise += emotion.surprise
#     #         trust += emotion.trust
#     # num_of_words = len(tweet)
#     # anger /= num_of_words
#     # anticipation /= num_of_words
#     # disgust /= num_of_words
#     # fear /= num_of_words
#     # joy /= num_of_words
#     # negative /= num_of_words
#     # positive /= num_of_words
#     # sadness /= num_of_words
#     # surprise /= num_of_words
#     # trust /= num_of_words
#
#     anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust = get_emotions(tweet['text'])
#
#     tweet["anger"] = anger
#     tweet["anticipation"] = anticipation
#     tweet["disgust"] = disgust
#     tweet["fear"] = fear
#     tweet["joy"] = joy
#     tweet["negative"] = negative
#     tweet["positive:"] = positive
#     tweet["sadness"] = sadness
#     tweet["surprise"] = surprise
#     tweet["trust"] = trust
#
#     connection.store_to_collection(tweet, "tweets_nrc")
