import re
from nltk import TweetTokenizer, WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet
from lexicons import contractions, punctuations, en_stopwords

# Preprocesses text from tweeter
def preprocess_text(tweet_text):
    tweet_tokenizer = TweetTokenizer()

    tokens = [token.lower().lstrip("@").lstrip("#") for token in tweet_tokenizer.tokenize(tweet_text)]
    tokens_no_contra = [contractions[token].split() if token in contractions else [token] for token in tokens]
    flat_list = [item for sublist in tokens_no_contra for item in sublist]
    tokens_semi_final = [token for token in flat_list if token not in punctuations and token not in en_stopwords]
    final_t = [token.replace("'s", "") for token in tokens_semi_final if not re.match('((www\.[^\s]+)|(https?://[^\s]+))', token)]

    text = []
    wnl = WordNetLemmatizer()
    tagged = pos_tag(final_t)
    for word, tag_prior in tagged:
        tag = nltk_tag_to_wordnet_tag(tag_prior)
        word = "not" if word == "n't" else word
        if tag:
            text.append(wnl.lemmatize(word.lower(), tag))
        else:
            text.append(wnl.lemmatize(word.lower()))

    return text

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Get hashtags from tweets
def get_hashtags(text):
    tweet_tokenizer = TweetTokenizer()
    return [token for token in tweet_tokenizer.tokenize(text) if re.match("^#(\w+)", token)]

# Get mentions from tweets
def get_mentions(text):
    tweet_tokenizer = TweetTokenizer()
    return [token for token in tweet_tokenizer.tokenize(text) if re.match("^@(?!.*\.\.)(?!.*\.$)[^\W][\w.]{0,29}$", token)]