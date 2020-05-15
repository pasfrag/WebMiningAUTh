import pandas as pd
from lexicons import joy_emojis, anger_emojis, sad_emojis, surprise_emojis, fear_emojis, disgust_emojis
from preprocessing_functions import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer

def dummy_tokenizer(doc):
    return doc

text = ["today", "is", "a", "great", "😀", "😀"]

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_tokenizer,
    preprocessor=dummy_tokenizer,
    token_pattern=None)

emotions = pd.read_csv("2018-11 emotions-classification-train.txt", sep="\t")
selected_emotions = emotions.drop(['anticipation', 'love', 'optimism', 'pessimism', 'trust'], axis=1)

selected_emotions['Tweet'] = selected_emotions['Tweet'].apply(lambda x: preprocess_text(x))
print(selected_emotions['Tweet'])
selected_emotions['PreprocessedTweet'] = selected_emotions['Tweet'].apply(lambda x: extract_emojis_semantic(x))
selected_emotions["JoyEmojis"] = selected_emotions['Tweet'].apply(lambda x: find_number_of_emojis(x, joy_emojis))
selected_emotions["SadEmojis"] = selected_emotions['Tweet'].apply(lambda x: find_number_of_emojis(x, sad_emojis))
selected_emotions["DisgustEmojis"] = selected_emotions['Tweet'].apply(
    lambda x: find_number_of_emojis(x, disgust_emojis))
selected_emotions["SurpriseEmojis"] = selected_emotions['Tweet'].apply(
    lambda x: find_number_of_emojis(x, surprise_emojis))
selected_emotions["FearEmojis"] = selected_emotions['Tweet'].apply(lambda x: find_number_of_emojis(x, fear_emojis))
selected_emotions["AngerEmojis"] = selected_emotions['Tweet'].apply(lambda x: find_number_of_emojis(x, anger_emojis))

selected_emotions["tfidf"] = tfidf.fit_transform(selected_emotions['PreprocessedTweet'])


