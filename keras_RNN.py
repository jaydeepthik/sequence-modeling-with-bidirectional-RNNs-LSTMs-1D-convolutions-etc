# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 15:27:35 2018

@author: jaydeep thik
"""

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers, models

max_features = 10000
max_len = 500
batch_size = 32

(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

model = models.Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)