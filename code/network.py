import os
import pickle

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Merge, MaxPooling1D, TimeDistributedDense, Flatten
from notes.TimeNote import TimeNote

class NNModel:

    def __init__(self, data_dim=200, timesteps=16, nb_classes=7):
        '''
        Creates a neural network with the specified conditions.
        '''
        # TODO: remove embedding layer once word2vec embeddings are added

        # encode the first entity
        encoder_a = Sequential()
        encoder_a.add(LSTM(300, input_dim=data_dim, input_length=timesteps, return_sequences=True))
        encoder_a.add(MaxPooling1D(pool_length=2))
        encoder_a.add(Flatten())

        # encode the second entity
        encoder_b = Sequential()
        encoder_b.add(LSTM(300, input_dim=data_dim, input_length=timesteps, return_sequences=True))
        encoder_b.add(MaxPooling1D(pool_length=2))
        encoder_b.add(Flatten())

        # combine and classify entities as a single relation
        decoder = Sequential()
        decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
        decoder.add(Dense(100, activation='sigmoid'))
        decoder.add(Dense(nb_classes, activation='softmax'))

        decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.classifier = decoder

    def train(self, notes, epochs=100):
        '''
        obtains entity pairs and tlink labels from every note passed, and uses them to train the network.
        '''

        tlinklabels = []
        X1 = []
        X2 = []

        for i, note in enumerate(notes):
            # get tlink lables
            tlinklabels += note.get_tlink_labels()

            # retrieve tlinks from the note and properly format them
            left_ids, right_ids = _get_token_id_subpaths(note)
            left_path = note.get_tokens_from_ids(left_ids)
            right_path = note.get_tokens_from_ids(right_ids)
            print right_path
            # X1 += left_path
            # X2 += right_path

        # print tlinklabels

        # may need to reformat labels using to_categorical
        # self.classifier.fit([X1, X2], tlinklabels, nb_epoch=epochs)

    def predict(self, notes):
        pass

def _get_token_id_subpaths(note):
    pairs = note.get_tlinked_entities()

    left_paths = []
    right_paths = []

    for pair in pairs:
        target_id = ''
        source_id = ''

        # extract and reformat the ids in the pair to be of form t# instead of w#
        # events may be linked to document creation time, which will not have an id
        if 'id' in pair["target_entity"][0]:
            target_id = pair["target_entity"][0]["id"]
            target_id = 't' + target_id[1:]
        if 'id' in pair["src_entity"][0]:
            source_id = pair["src_entity"][0]["id"]
            source_id = 't' + source_id[1:]

        left_paths, right_paths = note.dependency_paths.get_left_right_subpaths(target_id, source_id)

    return left_paths, right_paths

def _pre_process_labels(labels):
    '''convert tlink labels to integers so they can be processed accordingly'''

    processed_labels = []
    for label in labels:
        if label == "SIMULTANEOUS":
            processed_labels.append(1)
        elif label == "BEFORE":
            processed_labels.append(2)
        elif label == "AFTER":
            processed_labels.append(3)
        elif label == "IS_INCLUDED":
            processed_labels.append(4)
        elif label == "BEGUN_BY":
            processed_labels.append(5)
        elif label == "ENDED_BY":
            processed_labels.append(6)
        else:  # label for pairs which are not linked
            processed_labels.append(0)

    return processed_labels

if __name__ == "__main__":
    test = NNModel()
    tmp_note = TimeNote("APW19980418.0210.tml.TE3input", "APW19980418.0210.tml")
    print tmp_note.pre_processed_text[2][16]

    test.train([tmp_note])

    labels = tmp_note.get_tlink_labels()
    labels = _pre_process_labels(labels)
    labels = to_categorical(labels,7)
    print len(labels)
    print labels
    input1 = np.random.random((len(labels),300))
    input2 = np.random.random((len(labels),300))
    # labels = np.random.randint(7, size=(10000,1))
    test.classifier.fit([input1,input2], labels, nb_epoch=100)
    print test.classifier.predict_classes([input1,input2])
    pass
