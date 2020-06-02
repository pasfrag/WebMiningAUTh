import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from mongo import MongoHandler
import matplotlib.pyplot as plt
import seaborn as sns

mongo_connect = MongoHandler()
profiles = mongo_connect.retrieve_from_collection("twitter_profiles")

df = pd.DataFrame(list(profiles))
df = df.sample(frac=1, random_state=1)
df = df.drop('_id', axis=1)

text = df['text']
Y = df['label']

# data = df[['sentiment','subjectivity', 'label']]
# corr = data.corr()
# cor_plot = sns.heatmap(corr, annot=True)
# # plt.show()

# df[df['label'] == 0].hist()
# df[df['label'] == 1].hist()
# print(df[df['label'] == 0].describe())
# print(df[df['label'] == 1].describe())

def word_cloud(df):
    import wordcloud
    from wordcloud import WordCloud
    from PIL import Image

    char_mask = np.array(Image.open("data/instagram.png"))
    image_colors = wordcloud.ImageColorGenerator(char_mask)

    wc0 = WordCloud(background_color="white", max_words=200, width=400, height=400, mask=char_mask, random_state=1)\
       .generate(' '.join(df))

    # wc0 = WordCloud(background_color="white", max_words=200, width=400, height=400, mask=char_mask, random_state=1)\
    #    .generate(df['text'][df['label'] == 0].to_string())

    # wc1 = WordCloud(background_color="white", max_words=200, width=400, height=400, mask=char_mask, random_state=1)\
    #    .generate(df['text'][df['label'] == 1].to_string())

    plt.axis("off")
    plt.imshow(wc0.recolor(color_func=image_colors))
    # plt.imshow(wc1.recolor(color_func=image_colors))

# word_cloud(df)

