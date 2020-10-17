import pandas as pd
import numpy as np

df = pd.read_json("ig_users500.json")

vocabulary = set()
words = [x for hashtag in df["hashtags"] for x in hashtag]
counts = dict()

for word in words:
    if word in counts:
        counts[word] += 1
    else:
        counts[word] = 1

d = [k for k, v in counts.items() if v >= 20]

for x in d:
    vocabulary.add(x)

vocabulary = list(vocabulary)
coocurrence_matrix = np.zeros((len(vocabulary), len(vocabulary)))
edges = []

for user_hashtags in df["hashtags"]:
    for ind, hashtag1 in enumerate(list(set(user_hashtags))[:-1]):
        for hashtag2 in list(set(user_hashtags))[ind + 1:]:
            if hashtag1 in vocabulary and hashtag2 in vocabulary:
                ind1 = vocabulary.index(hashtag1)
                ind2 = vocabulary.index(hashtag2)
                coocurrence_matrix[ind1, ind2] += 1
                coocurrence_matrix[ind2, ind1] += 1
                # edges.append(f"{hashtag1},{hashtag2},Undirected")

for i in range(coocurrence_matrix.shape[0] -1):
    for j in range(i+1, coocurrence_matrix.shape[0]):
        if coocurrence_matrix[i, j] > 5:
            edges.append(f"{vocabulary[i]},{vocabulary[j]},Undirected")

with open('listfile.csv', 'w', encoding="utf-8") as filehandle:
    for edge in list(set(edges)):
        filehandle.write(edge)
        filehandle.write('\n')


