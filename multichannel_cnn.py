# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:14:10 2019

@author: jaydeep thik
"""

import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers

max_features = 10000
max_len = 500

(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=max_features)

X_train_pad = pad_sequences(X_train, max_len, padding='post')
X_test_pad = pad_sequences(X_test, max_len, padding='post') 

inputs  = models.Input(shape=(500,))

#channel #1
embedding1 = layers.Embedding(input_dim=max_features, output_dim=100)(inputs)
conv1 = layers.Conv1D(32, 4 ,activation='relu')(embedding1)
drop1 = layers.Dropout(0.5)(conv1)
mp1 = layers.MaxPool1D()(drop1)
flat1 = layers.Flatten()(mp1)

#channel #2
embedding2 = layers.Embedding(input_dim=max_features, output_dim=100)(inputs)
conv2 = layers.Conv1D(32, 6 ,activation='relu')(embedding2)
drop2 = layers.Dropout(0.5)(conv2)
mp2 = layers.MaxPool1D()(drop2)
flat2 = layers.Flatten()(mp2)

#channel #3
embedding3 = layers.Embedding(input_dim=max_features, output_dim=100)(inputs)
conv3 = layers.Conv1D(32, 8 ,activation='relu')(embedding3)
drop3 = layers.Dropout(0.5)(conv3)
mp3 = layers.MaxPool1D()(drop3)
flat3 = layers.Flatten()(mp3)

merged = layers.concatenate([flat1, flat2, flat3])

fc1 = layers.Dense(10, activation='relu')(merged)

output = layers.Dense(1, activation='sigmoid')(fc1)

model = models.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['acc'])
history = model.fit(X_train_pad, y_train, epochs=10, shuffle=True, validation_split=0.2)
