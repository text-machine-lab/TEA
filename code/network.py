import os
import pickle

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Graph
from keras.layers import Embedding, LSTM, Dense, Merge, MaxPooling1D, TimeDistributedDense, Flatten, Masking, Input, Permute
from notes.TimeNote import TimeNote
from gensim.models import word2vec

class NNModel:

    def __init__(self, data_dim=300, max_len=16, nb_classes=7):
        '''
        Creates a neural network with the specified conditions.
        '''
        # encode the first entity
        encoder_L = Sequential()
        # encoder_L.add(Masking(mask_value=0., input_shape=(data_dim, max_len)))
        encoder_L.add(LSTM(300, input_shape=(data_dim, max_len), return_sequences=True))
        encoder_L.add(MaxPooling1D(pool_length=300))
        encoder_L.add(Flatten())

        # encode the second entity
        encoder_R = Sequential()
        # encoder_R.add(Masking(mask_value=0., input_shape=(data_dim, max_len)))
        encoder_R.add(LSTM(300, input_shape=(data_dim, max_len), return_sequences=True))
        encoder_R.add(MaxPooling1D(pool_length=300))
        encoder_R.add(Flatten())

        # combine and classify entities as a single relation
        decoder = Sequential()
        decoder.add(Merge([encoder_R, encoder_L], mode='concat'))
        # decoder.add(Dense(100, activation='sigmoid'))
        decoder.add(Dense(nb_classes, activation='softmax'))

        # compile the final model
        decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.classifier = decoder

    def train(self, notes, epochs=100):
        '''
        obtains entity pairs and tlink labels from every note passed, and uses them to train the network.
        '''

        tlinklabels = []
        left_path_vectors = []
        right_path_vectors = []

        print 'Loading word embeddings...'
        word_vectors = word2vec.Word2Vec.load_word2vec_format(os.environ["TEA_PATH"]+'/GoogleNews-vectors-negative300.bin', binary=True)

        print 'Extracting dependency paths...'
        for i, note in enumerate(notes):
            # get tlink lables
            tlinklabels += note.get_tlink_labels()

            # retrieve tlinks from the note and properly format them
            left_ids, right_ids = _get_token_id_subpaths(note)
            left_path = note.get_tokens_from_ids(left_ids)
            right_path = note.get_tokens_from_ids(right_ids)

            # get the vectors for every word in the left path
            left_vecs = []
            for word in left_path:
                left_vecs.append(word_vectors[word])

            # get the vectors for every word in the right path
            right_vecs = []
            for word in right_path:
                right_vecs.append(word_vectors[word])

            left_path_vectors.append(left_vecs)
            right_path_vectors.append(right_vecs)

        # cast data to numpy arrays
        XL = numpy.asarray(left_path_vectors, dtype='float32')
        XR = numpy.asarray(right_path_vectors, dtype='float32')

        # reformat labels so that they can be used by the NN
        labels = _pre_process_labels(tlinklabels)
        Y = to_categorical(labels,7)

        # train the network
        print 'Training network...'
        self.classifier.fit([XL, XR], Y, nb_epoch=epochs)


        test = self.classifier.predict_classes([XL, XR])

        T = 0
        F = 0
        outs = 0
        for true, pred in zip(labels, test):
            if true == pred:
                T += 1
            else:
                F += 1
            if true == 0:
                outs += 0

        print "T: ", T, "F: ", F, "outs: ", outs

    def predict(self, notes):
        pass

def _get_token_id_subpaths(note):
    '''
    extract ids for the tokens in each half of the shortest dependency path between each token in each relation
    '''
    # TODO: for now we only look at the first token in a given entity. Eventually, we should get all tokens in the entity

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
    # print tmp_note.pre_processed_text[2][16]

    test.train([tmp_note])

    # labels = tmp_note.get_tlink_labels()
    # labels = _pre_process_labels(labels)
    # _labels = to_categorical(labels,7)
    # print len(labels)
    # print labels
    # input1 = np.random.random((len(labels),300, 16))
    # input2 = np.random.random((len(labels),300, 16))
    # # labels = np.random.randint(7, size=(10000,1))
    # test.classifier.fit([input1,input2], _labels, nb_epoch=100)
    # print test.classifier.predict_classes([input1,input2])
    # print labels
    pass
