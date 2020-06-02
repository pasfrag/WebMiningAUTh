from emot.emo_unicode import UNICODE_EMO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# Helper functions to extract features and preprocces the dataset
# Extract semantic of the emoji
def extract_emojis_semantic(tokens):
    for emot in UNICODE_EMO:
        for token in tokens:
            if token == emot:
                tokens.append(token.replace(emot, "_".join(
                    UNICODE_EMO[emot].replace(",", "").replace(":", "").split())))  # Replace emoji with their meaning
                tokens.remove(emot)
    return tokens


# Find number of appearences of an emoji inside a tweet
def find_number_of_emojis(tokens, emojis):
    count = 0
    for token in tokens:
        if token in emojis:
            count += 1
    return count


# Create dummy tokenizer to pass it as parameter to tfidf vectorizer later
def dummy_tokenizer(doc):
    return doc


# Create tfidf vectorizer to represent tweet text as a frequency calibrated vector
tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_tokenizer,
    preprocessor=dummy_tokenizer,
    token_pattern=None,
    min_df=10,
    max_df=100
)


# Create count vectorizer
cv = CountVectorizer(
    analyzer='word',
    tokenizer=dummy_tokenizer,
    preprocessor=dummy_tokenizer,
    ngram_range=(1, 1),
    min_df=10,
    max_df=100)
