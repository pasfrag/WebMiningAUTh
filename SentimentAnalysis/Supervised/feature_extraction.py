from SentimentAnalysis.Supervised.feature_extraction_helpers import find_number_of_emojis, extract_emojis_semantic, dummy_tokenizer, tfidf
import pandas as pd
import numpy as np

from lexicons import joy_emojis, anger_emojis, sad_emojis, surprise_emojis, fear_emojis, disgust_emojis
from preprocessing_functions import preprocess_text



def preprocces_dataframe(df):
    # Preprocces tweets to match with our global preproccesing
    df['Tweet'] = df['Tweet'].apply(lambda x: preprocess_text(x))
    # Replace emojis with their meanings
    df['PreprocessedTweet'] = df['Tweet'].apply(lambda x: extract_emojis_semantic(x))
    # Create columns to store number of emojis occurence based on their emotion (already existing labeled emojis)
    df["JoyEmojis"] = df['Tweet'].apply(lambda x: find_number_of_emojis(x, joy_emojis))
    df["SadEmojis"] = df['Tweet'].apply(lambda x: find_number_of_emojis(x, sad_emojis))
    df["DisgustEmojis"] = df['Tweet'].apply(lambda x: find_number_of_emojis(x, disgust_emojis))
    df["SurpriseEmojis"] = df['Tweet'].apply(lambda x: find_number_of_emojis(x, surprise_emojis))
    df["FearEmojis"] = df['Tweet'].apply(lambda x: find_number_of_emojis(x, fear_emojis))
    df["AngerEmojis"] = df['Tweet'].apply(lambda x: find_number_of_emojis(x, anger_emojis))
    # df["target"] = df[['joy','anger','disgust','fear','sadness','surprise']].apply(np.array, axis=1)
    return df


def vectorize_train_dataframe(df):
    # Vectorize the preprocced tweet text
    text_tfidf = tfidf.fit_transform(df['PreprocessedTweet']).toarray()
    df2 = pd.DataFrame(text_tfidf)
    # Reset index to avoid troubles in concatenation
    df.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df_concat = pd.concat([df, df2], axis=1)
    # Select only the columns we need
    df = df_concat.drop(['ID','Tweet', 'PreprocessedTweet','anticipation','love','optimism','pessimism','trust'], axis=1)
    return df


def vectorize_test_dataframe(df):
    # Vectorize the preprocced tweet text
    text_tfidf = tfidf.transform(df['PreprocessedTweet']).toarray()
    df2 = pd.DataFrame(text_tfidf)
    df.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df_concat = pd.concat([df, df2], axis=1)
    # Select only the columns we need
    df = df_concat.drop(['ID', 'Tweet', 'PreprocessedTweet', 'anticipation', 'love', 'optimism', 'pessimism', 'trust'],
                        axis=1)
    return df
