import os
import pickle

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Merge, MaxPooling1D, TimeDistributedDense, Flatten

class NNModel:

    def __init__(self, data_dim=16, timesteps=16, nb_classes=7):

        # encode the first entity
        encoder_a = Sequential()
        encoder_a.add(Embedding(output_dim=200, input_dim=data_dim, input_length=timesteps, mask_zero=False))
        encoder_a.add(LSTM(32, return_sequences=True))
        encoder_a.add(MaxPooling1D(pool_length=2))
        encoder_a.add(Flatten())

        # encode the second entity
        encoder_b = Sequential()
        encoder_b.add(Embedding(output_dim=200, input_dim=data_dim, input_length=timesteps, mask_zero=False))
        encoder_b.add(LSTM(32, return_sequences=True))
        encoder_b.add(MaxPooling1D(pool_length=2))
        encoder_b.add(Flatten())

        # combine and classify entities as a single relation
        decoder = Sequential()
        decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
        decoder.add(Dense(32, activation='sigmoid'))
        decoder.add(Dense(nb_classes, activation='softmax'))

        decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.classifier = decoder

    def train(self, notes, epochs=100):

        tlinklabels = []
        X1 = []
        X2 = []

        for i, note in enumerate(notes):
            tlinklabels += note.get_tlink_labels()
            entity_pairs = note.get_tlinked_entities()

            # TODO: get the sdp for each pair. Need to figure out how to get token IDs for each entity
            # TODO: format the data appropriately. Minimally must split dependency path
            # X1 += sdp_left
            # X2 += sdp_right

        # may need to reformat labels using to_categorical
        self.classifier.fit([X1, X2], tlinklabels, nb_epoch=epochs)

if __name__ == "__main__":
    test = NNModel()

    input1 = np.random.random((10000,16))
    input2 = np.random.random((10000,16))
    labels = np.random.randint(7, size=(10000,1))
    print labels
    labels = to_categorical(labels,7)
    test.classifier.fit([input1,input2], labels, nb_epoch=100)
    print test.classifier.predict_classes([input1,input2])
    pass
