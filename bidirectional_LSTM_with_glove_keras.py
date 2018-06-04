# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 13:24:01 2018

@author: jaydeep thik

"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers, models, optimizers
import numpy as np
import os

def load_data():
    main_dir = "F:/machine learning/code/NLP_word_embedding/imdb_SA_embeddings/aclImdb"
    train_dir = os.path.join(main_dir, 'train')
    
    labels=[]
    text = []
    
    for label in ['pos', 'neg']:
        working_dir = os.path.join(train_dir, label)
        for text_file in os.listdir(working_dir):
            f= open(os.path.join(working_dir, text_file), encoding='utf8')
            text.append(f.read())
            f.close()
            
            if label=='pos':
                labels.append(1)
            else:
                labels.append(0)
                
    return text, labels            

## loading GloVe
def load_glove(dim):
    glove_dir = "F:/machine learning/code/NLP_word_embedding/data"
    
    embedding = {}
    f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'), encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coeff = np.asarray(values[1:], dtype = 'float32')
        embedding[word]=coeff
    f.close()
    
    #creating embedding matrix
    embedding_dim = dim
    embedding_matrix = np.zeros((max_words, embedding_dim))
    
    for word, idx in word_index.items():
        if idx < max_words:
            embedding_vector = embedding.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx]= embedding_vector
    return embedding_matrix     

maxlen = 500
training_samples = 10000
validation_samples = 5000
max_words = 10000

#tokenizing
tokenizer = Tokenizer(num_words=max_words)
text, labels = load_data()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

#get word indices of all the unique words in the text
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print(data.shape)
print(labels.shape)

indices = np.array(range(labels.shape[0]))
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

X_train = data[:training_samples]
y_train = labels[:training_samples]

X_val = data[training_samples:training_samples+validation_samples]
y_val = labels[training_samples:training_samples+validation_samples]


embedding_dim = 50
#model
model = models.Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
#model.add(layers.Flatten())
#model.add(layers.Dense(32, activation='relu'))
model.add(layers.Bidirectional(layers.LSTM(16, return_sequences=True)))
model.add(layers.LSTM(32, return_sequences=False))
model.add(layers.Dense(1, activation='sigmoid'))

#prevent overriding of the embedding matrix
embedding_matrix = load_glove(embedding_dim)
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=128, epochs=30, validation_data=(X_val, y_val))
