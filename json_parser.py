import pandas as pd
from collections import Counter

class VisualizationParsing(object):
    data = None
    twitter_profiles = None
    instagram_profiles = None

    def __init__(self):
        self.data = pd.read_json("twitter.json")
        self.twitter_profiles = pd.read_json("twitter_profiles_final.json")
        self.instagram_profiles = pd.read_json("ig_interests.json")
        self.stops = ["+", '`', '”', '...', '…', '“', ']', '[', '=', '?', '!', '’', ',', '‘', '1', '2', '3', '4', '5', '9', '10']

    # Returns the final data to be visualized
    def finalize_data(self):
        data = dict()
        data['average_emotions'] = self.average_emotions()
        data['tweets_per_day'] = self.tweets_per_day()

        believers, uncertains, deniers, believers_scores, uncertains_scores, deniers_scores = self.user_profiles_emotions()
        data['believers'] = believers
        data['uncertains'] = uncertains
        data['deniers'] = deniers
        data['believers_scores'] = believers_scores
        data['uncertains_scores'] = uncertains_scores
        data['deniers_scores'] = deniers_scores

        anger_s, anticipation_s, disgust_s, fear_s, joy_s, negative_s, positive_s, sadness_s, surprise_s, trust_s = self.emotion_per_day()
        data['anger_s'] = anger_s
        data['anticipation_s'] = anticipation_s
        data['disgust_s'] = disgust_s
        data['fear_s'] = fear_s
        data['joy_s'] = joy_s
        data['negative_s'] = negative_s
        data['positive_s'] = positive_s
        data['sadness_s'] = sadness_s
        data['surprise_s'] = surprise_s
        data['trust_s'] = trust_s

        data['hashtags'] = self.hashtags_counts()
        data['mentions'] = self.mentions_counts()
        data['instagram'] = self.instagram_interests_counts()
        data['words'] = self.word_counts()

        data['believers_words'] = self.believers_counts()
        data['deniers_words'] = self.deniers_counts()

        return data

    # Returns the total average emotion score
    def average_emotions(self):
        emotions = list()
        emotions.append(self.data.anger.mean())
        emotions.append(self.data.anticipation.mean())
        emotions.append(self.data.disgust.mean())
        emotions.append(self.data.fear.mean())
        emotions.append(self.data.joy.mean())
        emotions.append(self.data.negative.mean())
        emotions.append(self.data["positive:"].mean())
        emotions.append(self.data.sadness.mean())
        emotions.append(self.data.surprise.mean())
        emotions.append(self.data.trust.mean())
        return emotions

    # Returns the number of tweets per days
    def tweets_per_day(self):
        number_of_tweets = list()
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-03-07"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-03-08"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-03-09"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-03-10"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-03-11"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-03-12"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-03-13"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-03-14"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-03-15"]))
        number_of_tweets.append(None)
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-05-05"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-05-06"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-05-07"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-05-08"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-05-09"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-05-10"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-05-11"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-05-12"]))
        number_of_tweets.append(len(self.data[self.data.created_at == "2020-05-13"]))
        return number_of_tweets

    # Returns emotions, sentiment and subjectivity for the three types of user (non-deniers, uncertain, deniers)
    def user_profiles_emotions(self):
        believers = []
        uncertains = []
        deniers = []
        believers_scores = []
        uncertains_scores = []
        deniers_scores = []

        # denier_users = self.twitter_profiles[(self.twitter_profiles.anger != 0) and (self.twitter_profiles.anticipation != 0) and (self.twitter_profiles.disgust != 0) and (self.twitter_profiles.joy != 0) and (self.twitter_profiles.sadness != 0) and (self.twitter_profiles.surprise != 0) and (self.twitter_profiles.trust != 0)]

        believers.append(self.twitter_profiles[self.twitter_profiles.prediction == "NON-DENIER"].anger.mean())
        believers.append(self.twitter_profiles[self.twitter_profiles.prediction == "NON-DENIER"].anticipation.mean())
        believers.append(self.twitter_profiles[self.twitter_profiles.prediction == "NON-DENIER"].disgust.mean())
        believers.append(self.twitter_profiles[self.twitter_profiles.prediction == "NON-DENIER"].joy.mean())
        believers_scores.append(self.twitter_profiles[self.twitter_profiles.prediction == "NON-DENIER"].sentiment.mean())
        believers_scores.append(self.twitter_profiles[self.twitter_profiles.prediction == "NON-DENIER"].subjectivity.mean())
        believers.append(self.twitter_profiles[self.twitter_profiles.prediction == "NON-DENIER"].sadness.mean())
        believers.append(self.twitter_profiles[self.twitter_profiles.prediction == "NON-DENIER"].surprise.mean())
        believers.append(self.twitter_profiles[self.twitter_profiles.prediction == "NON-DENIER"].trust.mean())

        uncertains.append(self.twitter_profiles[self.twitter_profiles.prediction == "UNCERTAIN"].anger.mean())
        uncertains.append(self.twitter_profiles[self.twitter_profiles.prediction == "UNCERTAIN"].anticipation.mean())
        uncertains.append(self.twitter_profiles[self.twitter_profiles.prediction == "UNCERTAIN"].disgust.mean())
        uncertains.append(self.twitter_profiles[self.twitter_profiles.prediction == "UNCERTAIN"].joy.mean())
        uncertains_scores.append(self.twitter_profiles[self.twitter_profiles.prediction == "UNCERTAIN"].sentiment.mean())
        uncertains_scores.append(self.twitter_profiles[self.twitter_profiles.prediction == "UNCERTAIN"].subjectivity.mean())
        uncertains.append(self.twitter_profiles[self.twitter_profiles.prediction == "UNCERTAIN"].sadness.mean())
        uncertains.append(self.twitter_profiles[self.twitter_profiles.prediction == "UNCERTAIN"].surprise.mean())
        uncertains.append(self.twitter_profiles[self.twitter_profiles.prediction == "UNCERTAIN"].trust.mean())

        deniers.append(self.twitter_profiles[self.twitter_profiles.prediction == "DENIER"].anger.mean())
        deniers.append(self.twitter_profiles[self.twitter_profiles.prediction == "DENIER"].anticipation.mean())
        deniers.append(self.twitter_profiles[self.twitter_profiles.prediction == "DENIER"].disgust.mean())
        deniers.append(self.twitter_profiles[self.twitter_profiles.prediction == "DENIER"].joy.mean())
        deniers_scores.append(self.twitter_profiles[self.twitter_profiles.prediction == "DENIER"].sentiment.mean())
        deniers_scores.append(self.twitter_profiles[self.twitter_profiles.prediction == "DENIER"].subjectivity.mean())
        deniers.append(self.twitter_profiles[self.twitter_profiles.prediction == "DENIER"].sadness.mean())
        deniers.append(self.twitter_profiles[self.twitter_profiles.prediction == "DENIER"].surprise.mean())
        deniers.append(self.twitter_profiles[self.twitter_profiles.prediction == "DENIER"].trust.mean())

        return believers, uncertains, deniers, believers_scores, uncertains_scores, deniers_scores

    # Returns emotion scores per day
    def emotion_per_day(self):
        anger_s = []
        anticipation_s = []
        disgust_s = []
        fear_s = []
        joy_s = []
        negative_s = []
        positive_s = []
        sadness_s = []
        surprise_s = []
        trust_s = []
        anger_s.append(self.data[self.data.created_at == "2020-03-07"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-03-08"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-03-09"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-03-10"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-03-11"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-03-12"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-03-13"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-03-14"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-03-15"].anger.mean())
        anger_s.append(None)
        anger_s.append(self.data[self.data.created_at == "2020-05-05"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-05-06"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-05-07"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-05-08"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-05-09"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-05-10"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-05-11"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-05-12"].anger.mean())
        anger_s.append(self.data[self.data.created_at == "2020-05-13"].anger.mean())

        anticipation_s.append(self.data[self.data.created_at == "2020-03-07"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-03-08"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-03-09"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-03-10"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-03-11"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-03-12"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-03-13"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-03-14"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-03-15"].anticipation.mean())
        anticipation_s.append(None)
        anticipation_s.append(self.data[self.data.created_at == "2020-05-05"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-05-06"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-05-07"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-05-08"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-05-09"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-05-10"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-05-11"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-05-12"].anticipation.mean())
        anticipation_s.append(self.data[self.data.created_at == "2020-05-13"].anticipation.mean())

        disgust_s.append(self.data[self.data.created_at == "2020-03-07"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-03-08"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-03-09"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-03-10"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-03-11"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-03-12"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-03-13"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-03-14"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-03-15"].disgust.mean())
        disgust_s.append(None)
        disgust_s.append(self.data[self.data.created_at == "2020-05-05"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-05-06"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-05-07"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-05-08"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-05-09"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-05-10"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-05-11"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-05-12"].disgust.mean())
        disgust_s.append(self.data[self.data.created_at == "2020-05-13"].disgust.mean())

        fear_s.append(self.data[self.data.created_at == "2020-03-07"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-03-08"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-03-09"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-03-10"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-03-11"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-03-12"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-03-13"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-03-14"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-03-15"].fear.mean())
        fear_s.append(None)
        fear_s.append(self.data[self.data.created_at == "2020-05-05"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-05-06"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-05-07"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-05-08"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-05-09"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-05-10"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-05-11"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-05-12"].fear.mean())
        fear_s.append(self.data[self.data.created_at == "2020-05-13"].fear.mean())

        joy_s.append(self.data[self.data.created_at == "2020-03-07"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-03-08"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-03-09"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-03-10"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-03-11"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-03-12"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-03-13"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-03-14"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-03-15"].joy.mean())
        joy_s.append(None)
        joy_s.append(self.data[self.data.created_at == "2020-05-05"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-05-06"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-05-07"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-05-08"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-05-09"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-05-10"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-05-11"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-05-12"].joy.mean())
        joy_s.append(self.data[self.data.created_at == "2020-05-13"].joy.mean())

        negative_s.append(self.data[self.data.created_at == "2020-03-07"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-03-08"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-03-09"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-03-10"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-03-11"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-03-12"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-03-13"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-03-14"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-03-15"].negative.mean())
        negative_s.append(None)
        negative_s.append(self.data[self.data.created_at == "2020-05-05"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-05-06"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-05-07"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-05-08"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-05-09"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-05-10"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-05-11"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-05-12"].negative.mean())
        negative_s.append(self.data[self.data.created_at == "2020-05-13"].negative.mean())

        positive_s.append(self.data[self.data.created_at == "2020-03-07"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-03-08"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-03-09"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-03-10"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-03-11"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-03-12"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-03-13"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-03-14"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-03-15"]['positive:'].mean())
        positive_s.append(None)
        positive_s.append(self.data[self.data.created_at == "2020-05-05"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-05-06"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-05-07"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-05-08"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-05-09"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-05-10"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-05-11"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-05-12"]['positive:'].mean())
        positive_s.append(self.data[self.data.created_at == "2020-05-13"]['positive:'].mean())

        sadness_s.append(self.data[self.data.created_at == "2020-03-07"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-03-08"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-03-09"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-03-10"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-03-11"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-03-12"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-03-13"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-03-14"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-03-15"].sadness.mean())
        sadness_s.append(None)
        sadness_s.append(self.data[self.data.created_at == "2020-05-05"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-05-06"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-05-07"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-05-08"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-05-09"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-05-10"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-05-11"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-05-12"].sadness.mean())
        sadness_s.append(self.data[self.data.created_at == "2020-05-13"].sadness.mean())

        surprise_s.append(self.data[self.data.created_at == "2020-03-07"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-03-08"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-03-09"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-03-10"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-03-11"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-03-12"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-03-13"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-03-14"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-03-15"].surprise.mean())
        surprise_s.append(None)
        surprise_s.append(self.data[self.data.created_at == "2020-05-05"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-05-06"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-05-07"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-05-08"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-05-09"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-05-10"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-05-11"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-05-12"].surprise.mean())
        surprise_s.append(self.data[self.data.created_at == "2020-05-13"].surprise.mean())

        trust_s.append(self.data[self.data.created_at == "2020-03-07"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-03-08"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-03-09"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-03-10"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-03-11"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-03-12"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-03-13"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-03-14"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-03-15"].trust.mean())
        trust_s.append(None)
        trust_s.append(self.data[self.data.created_at == "2020-05-05"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-05-06"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-05-07"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-05-08"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-05-09"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-05-10"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-05-11"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-05-12"].trust.mean())
        trust_s.append(self.data[self.data.created_at == "2020-05-13"].trust.mean())
        return anger_s, anticipation_s, disgust_s, fear_s, joy_s, negative_s, positive_s, sadness_s, surprise_s, trust_s

    # Returns twitter hashtags
    def hashtags_counts(self):
        cnt = Counter()

        for row in self.data.hashtags:
            for word in row:
                cnt[word.lower()] += 1
        keys = list(cnt.keys())
        values = list(cnt.values())
        ret_lst = []
        for k, v in zip(keys, values):
            if v > 99:
                ins_dct = dict()
                ins_dct['x'] = k
                ins_dct['value'] = v
                ret_lst.append(ins_dct)
        return ret_lst

    # Returns instagram interests
    def mentions_counts(self):
        cnt = Counter()

        for row in self.data.mentions:
            for word in row:
                cnt[word.lower()] += 1
        keys = list(cnt.keys())
        values = list(cnt.values())
        ret_lst = []
        for k, v in zip(keys, values):
            if v > 50:
                ins_dct = dict()
                ins_dct['x'] = k
                ins_dct['value'] = v
                ret_lst.append(ins_dct)
        return ret_lst

    # Returns instagram interests
    def instagram_interests_counts(self):
        cnt = Counter()

        for word in self.instagram_profiles.hashtags:
            # for word in row:
            cnt[word.lower()] += 1
        keys = list(cnt.keys())
        values = list(cnt.values())
        ret_lst = []
        for k, v in zip(keys, values):
            if v > 100:
                ins_dct = dict()
                ins_dct['x'] = k
                ins_dct['value'] = v
                ret_lst.append(ins_dct)
        return ret_lst

    # Returns word counts
    def word_counts(self):
        cnt = Counter()

        for row in self.data.text:
            for word in row:
                cnt[word.lower()] += 1
        keys = list(cnt.keys())
        values = list(cnt.values())
        ret_lst = []
        for k, v in zip(keys, values):
            if v > 700 and k not in self.stops:
                ins_dct = dict()
                ins_dct['x'] = k
                ins_dct['value'] = v
                ret_lst.append(ins_dct)
        return ret_lst

    def believers_counts(self):
        cnt = Counter()

        for row in self.twitter_profiles[self.twitter_profiles.prediction == "NON-DENIER"].text:
            for lst in row:
                for word in lst:
                    cnt[word.lower()] += 1
        keys = list(cnt.keys())
        values = list(cnt.values())
        ret_lst = []
        for k, v in zip(keys, values):
            if v > 400 and k not in self.stops:
                ins_dct = dict()
                ins_dct['x'] = k
                ins_dct['value'] = v
                ret_lst.append(ins_dct)
        return ret_lst

    def deniers_counts(self):
        cnt = Counter()

        for row in self.twitter_profiles[self.twitter_profiles.prediction == "DENIER"].text:
            for lst in row:
                for word in lst:
                    cnt[word.lower()] += 1
        keys = list(cnt.keys())
        values = list(cnt.values())
        ret_lst = []
        for k, v in zip(keys, values):
            if v > 100 and k not in self.stops:
                ins_dct = dict()
                ins_dct['x'] = k
                ins_dct['value'] = v
                ret_lst.append(ins_dct)
        return ret_lst
