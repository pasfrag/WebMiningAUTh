from SentimentAnalysis.Supervised.feature_extraction_helpers import cv,tfidf
import pandas as pd


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