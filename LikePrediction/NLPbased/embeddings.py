import numpy as np
from keras import Sequential
from keras.layers import Dropout, Conv1D, Embedding, MaxPooling1D, LSTM, Dense
from LikePrediction.NLPbased.like_prediction_vectorizer import read_tweets_and_instaposts, split_and_preprocces, NUM_WORDS, tokenizer
from LikePrediction.NLPbased.metrics import recall,f1,precision

# Read tweets from connection and preprocess them to have the right input for our nn
coll_name = 'twitter_final'
tweets, labels = read_tweets_and_instaposts(coll_name)
X_train, X_test, y_train, y_test = split_and_preprocces(tweets, labels)

# Read the pretrained embeddings from glove
embeddings_idx = dict()
file = open(r'C:\Users\passt\PycharmProjects\WebMiningAUTh\LikePrediction\NLPbased\glove6b300dtxt\glove.6B.300d.txt', encoding='utf-8')
for line in file:
    vals = line.split()
    word = vals[0]
    numbers = np.asarray(vals[1:], dtype='float32')
    embeddings_idx[word] = numbers
file.close()

# Create matrix for the pretrained embeddings
embeddings = np.zeros((NUM_WORDS, 300))
for token, idx in tokenizer.word_index.items():
    if idx > NUM_WORDS - 1:
        break
    else:
        vector = embeddings_idx.get(token)
        if vector is not None:
            embeddings[idx] = vector

# Create model
glove = Sequential()
glove.add(Embedding(NUM_WORDS, 300, input_length=50, weights=[embeddings], trainable=False))
glove.add(Dropout(0.2))
glove.add(Conv1D(64, 5, activation='relu'))
glove.add(MaxPooling1D(pool_size=4))
glove.add(LSTM(100, return_sequences=True))
glove.add(Dropout(0.2))
glove.add(LSTM(100))
glove.add(Dense(4, activation='softmax'))
glove.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',f1,precision, recall])

# Train the model and make predictions
glove.fit(X_train, y_train, epochs=5)
y_pred = glove.predict(X_test)

print('Test set')

loss, accuracy, f1_score, precision, recall = glove.evaluate(X_test, y_test, verbose=0)
print(loss, accuracy, f1_score, precision, recall)


# Save model weights
# glove.save('models/pretrained.h5')


