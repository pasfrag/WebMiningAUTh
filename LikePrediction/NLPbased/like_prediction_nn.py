from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from LikePrediction.NLPbased.like_prediction_vectorizer import read_tweets_and_instaposts, split_and_preprocces
from LikePrediction.NLPbased.metrics import recall,precision,f1
# Read tweets
coll_name = 'twitter_final'
tweets, labels = read_tweets_and_instaposts(coll_name)

# Read tweets from connection and preprocess them to have the right input for our nn
X_train, X_test, y_train, y_test = split_and_preprocces(tweets, labels)

model = Sequential()
model.add(Embedding(20000, 150, input_length=X_train.shape[1]))
model.add(LSTM(100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',f1,precision, recall])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=128)

# Evaluate model
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print(loss, accuracy, f1_score, precision, recall)

# Save model weights
# model.save('models/2LSTM100.h5')