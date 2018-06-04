# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:05:58 2018

@author: jaydeep thik
"""

from keras import layers, models
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import optimizers

max_features = 10000
max_len = 500

(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = sequence.pad_sequences(X_train, max_len)
X_test = sequence.pad_sequences(X_test, max_len)

model = models.Sequential()
model.add(layers.Embedding(max_features, 30, input_length=max_len))
model.add(layers.Conv1D(16, 5, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
model.add(layers.Bidirectional(layers.LSTM(32, dropout=0.50, recurrent_dropout=0.5)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['acc'])
history  = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
