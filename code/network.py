import os
import pickle

import numpy as np
# from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Merge

data_dim = 16
timesteps = 16
nb_classes = 7

encoder_a = Sequential()
encoder_a.add(Embedding(output_dim=200, input_dim=data_dim, input_length=timesteps, mask_zero=True))
encoder_a.add(LSTM(32))

encoder_b = Sequential()
encoder_b.add(Embedding(output_dim=200, input_dim=data_dim, input_length=timesteps, mask_zero=True))
encoder_b.add(LSTM(32))

decoder = Sequential()
decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
decoder.add(Dense(32, activation='sigmoid'))
decoder.add(Dense(nb_classes, activation='softmax'))

decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')
