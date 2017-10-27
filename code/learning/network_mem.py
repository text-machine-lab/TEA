from __future__ import print_function
import os
import sys
import numpy as np
np.random.seed(1337)
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Reshape, LSTM, Dense, concatenate, Merge, MaxPooling1D, TimeDistributed, Flatten, Permute, Input, Dropout, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.optimizers import Adam
from word2vec import load_word2vec_binary
from collections import deque

from network import Network
from ntm_models import get_untrained_model, get_ntm_model, get_ntm_model2, get_ntm_model3
from ntm_models import LABELS, DENSE_LABELS, EMBEDDING_DIM, MAX_LEN
TYPE_MARKERS = {'_INTRA_': 0, '_CROSS_': 1, '_DCT_': -1}

class NetworkMem(Network):
    def __init__(self):
        super(NetworkMem, self).__init__()

    def slice_data(self, generator, batch_size):
        """Slice data into equal batch sizes
        Left-over small chunks will be augmented with random samples
        """
        while True:
            feed = generator.next() # XL, XR, type_markers, y, <pair_index>
            data = feed[:4]
            N = len(data[-1])
            i = 0
            while (i+1) * batch_size <= N:
                if len(feed) == 5: # has pair_index, test data
                    if (i + 1) * batch_size == N:
                        marker = -1  # end of note
                    else:
                        marker = 0
                    # we need to add a dummy dimension to make the data 4D
                    out_data = [np.expand_dims(item[i*batch_size:(i+1)*batch_size], axis=0) for item in data]
                    yield [out_data[0], out_data[1], out_data[2]], out_data[3], feed[-1], marker
                else:
                    out_data = [np.expand_dims(item[i*batch_size:(i+1)*batch_size], axis=0) for item in data]
                    yield [out_data[0], out_data[1], out_data[2]], out_data[3]
                i += 1

            left_over = N % batch_size
            if left_over != 0:
                to_add = batch_size - left_over
                indexes_to_add = np.random.choice(N, to_add) # randomly sample more instances
                indexes = np.concatenate((np.arange(i*batch_size, N), indexes_to_add))
                if len(feed) == 5:
                    marker = left_over
                    out_data = [np.expand_dims(item[indexes], axis=0) for item in data]
                    yield [out_data[0], out_data[1], out_data[2]], out_data[3], feed[-1], marker
                else:
                    out_data = [np.expand_dims(item[indexes], axis=0) for item in data]
                    yield [out_data[0], out_data[1], out_data[2]], out_data[3]


    def generate_training_input(self, notes, pair_type, max_len=15, multiple=1, nolink_ratio=None, no_ntm=False):
        """
        Input data generator. Each step generates all data from one note file.
        Training data will be in narrative order
        """

        # data tensor for left and right SDP subpaths
        XL = None
        XR = None

        if self.word_vectors is None:
            print('Loading word embeddings...')
            self.word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin',
                                                verbose=0)
        data_q = deque()
        count = 0
        count_none = 0
        for note in notes:

            XL, XR, id_pairs, type_markers = self._extract_path_representations(note, self.word_vectors, pair_type)
            type_markers = np.expand_dims(np.array([TYPE_MARKERS[item] for item in type_markers]), axis=-1) # convert to 2D

            if not id_pairs:
                print("No pair found:", note.annotated_note_path)
                continue

            pos_case_indexes = []
            neg_case_indexes = []
            note_labels = []

            if DENSE_LABELS:
                id_to_labels = note.get_id_to_denselabels()  # use TimeBank-Dense labels
            else:
                id_to_labels = note.id_to_labels

            if not id_to_labels:
                continue

            for index, pair in enumerate(id_pairs):
                if pair in id_to_labels and id_to_labels[pair] != 'None': #TimeBank Dense could have None labels
                    pos_case_indexes.append(index)
                else:
                    neg_case_indexes.append(index)
                note_labels.append(id_to_labels.get(pair, 'None'))
            note_labels = np.array(note_labels)

            if nolink_ratio is not None:
                np.random.shuffle(neg_case_indexes)
                n_samples = min(len(neg_case_indexes), int(nolink_ratio * len(pos_case_indexes)))
                neg_case_indexes = neg_case_indexes[0:n_samples]
                if not neg_case_indexes:
                    training_indexes = np.array(pos_case_indexes, dtype=np.int32)
                else:
                    training_indexes = np.concatenate([pos_case_indexes, neg_case_indexes])
                XL = XL[training_indexes, :, :]
                XR = XR[training_indexes, :, :]
                note_labels = note_labels[training_indexes]
                type_markers = type_markers[training_indexes]

            if XL is not None and XL.shape[0] != 0:
                XL = pad_sequences(XL, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                XR = pad_sequences(XR, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
            else:
                continue

            labels = to_categorical(self._convert_str_labels_to_int(note_labels), len(LABELS))

            if no_ntm:
                data_q.append(([XL, XR, type_markers], labels))
            else:
                data_q.append((XL, XR, type_markers, labels))

            for label in labels:
                if label[12] == 1:
                    count_none += 1
                else:
                    count += 1

        print("Positive instances:", count)
        print("Negative instances:", count_none)
        if nolink_ratio is not None:
            assert count_none <= count * nolink_ratio

        while True:
            for i in range(multiple):  # read the document multiple times
                yield data_q[0]
            data_q.rotate(-1)

    def generate_test_input(self, notes, pair_type, max_len=15, multiple=1, no_ntm=False, reverse_pairs=True):

        if self.word_vectors is None:
            print('Loading word embeddings...')
            self.word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin',
                                                     verbose=0)

        data_q = deque()
        count_none = 0
        count = 0
        for i, note in enumerate(notes):
            pair_index = {}  # record note id and all the used entity pairs

            XL, XR, id_pairs, type_markers = self._extract_path_representations(note, self.word_vectors, pair_type)
            type_markers = np.expand_dims(np.array([TYPE_MARKERS[item] for item in type_markers]), axis=-1)

            if DENSE_LABELS:
                id_to_labels = note.id_to_denselabels  # use TimeBank-Dense labels
            else:
                id_to_labels = note.id_to_labels

            if id_to_labels:
                note_labels = []
                for index, pair in enumerate(id_pairs):  # id pairs that have tlinks
                    label_from_file = id_to_labels.get(pair, 'None')
                    note_labels.append(label_from_file)

                labels = np.array(self._convert_str_labels_to_int(note_labels), dtype='int16')

                for index, pair in enumerate(id_pairs):
                    pair_index[(i, pair)] = index # map pairs to their index in data array, i is index for notes

                if XL is not None and XL.shape[0] != 0:
                    XL = pad_sequences(XL, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                    XR = pad_sequences(XR, maxlen=max_len, dtype='float32', padding='post', truncating='pre', value=0.)
                else:
                    continue

                if no_ntm:
                    data_q.append(([XL, XR, type_markers], labels, pair_index, note.annotated_note_path))
                else:
                    data_q.append((XL, XR, type_markers, labels, pair_index))

                for label in labels:
                    if label == 12: count_none += 1
                    else: count += 1

        print("Positive instances:", count)
        print("Negative instances:", count_none)

        while True:
            for i in range(multiple):
                yield data_q[0]
            data_q.rotate(-1)

    def train_model(self, model=None, no_ntm=False, epochs=100, steps_per_epoch=10, validation_steps=10, input_generator=None,
                    val_generator=None, weight_classes=False,
                    encoder_dropout=0.5, decoder_dropout=0.5, input_dropout=0.5, LSTM_size=128, dense_size=128,
                    max_len='auto', nb_classes=13, callbacks=[], batch_size=300):

        # infer maximum sequence length
        if max_len == 'auto':
            max_len = None

        if model is None:
            if no_ntm:
                model = get_untrained_model(encoder_dropout=encoder_dropout, decoder_dropout=decoder_dropout,
                                                 input_dropout=input_dropout, LSTM_size=LSTM_size, dense_size=dense_size,
                                                 max_len=max_len, nb_classes=len(LABELS))

            else:
                model = get_ntm_model3(batch_size=batch_size, m_depth=256, n_slots=128, ntm_output_dim=128, shift_range=3, max_len=15, read_heads=2, write_heads=1, nb_classes=13)
        # train the network
        print('Training network...')

        training_history = model.fit_generator(input_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                               callbacks=callbacks, validation_data=val_generator,
                                               validation_steps=validation_steps, class_weight=None, max_q_size=300, workers=1)

        return model, training_history.history

    def predict(self, model, data_generator, batch_size=300, evaluation=True, smart=True, no_ntm=False):
        predictions = []
        scores = []
        pair_indexes = {}
        true_labels = None
        probs_in_note = None
        labels_in_note = []
        end_of_note = False
        print("not using NTM", no_ntm)

        while True:
            if no_ntm:
                X, y, pair_index, path_name = data_generator.next()
                print("file name to predict", path_name)
                marker = -1
            else:
                X, y, pair_index, marker = data_generator.next()

            note_index = pair_index.keys()[0][0]
            if end_of_note and note_index == 0: # all data consumed
                break
            pair_indexes.update(pair_index)

            probs = model.predict(X, batch_size=batch_size) # keras functional API model predicts probs
            if not no_ntm:
                y = y[0, :]  # remove the dummy dimension
                probs = probs[0, :]

            if marker > 0: # end of note with leftover
                y = y[:marker]
                probs = probs[:marker]
            labels = [np.argmax(x) for x in probs]

            if probs_in_note is None:
                probs_in_note = probs
                labels_in_note = labels
            else:
                probs_in_note = np.concatenate([probs_in_note, probs])
                labels_in_note += labels

            if marker != 0: # end of note
                end_of_note = True
                if smart:
                    labels_in_note, pair_index = self.smart_predict(labels_in_note, probs_in_note, pair_index, type='int')
                predictions += labels_in_note
                scores.append(probs_in_note)
                probs_in_note = None
                labels_in_note = []

            if true_labels is None:
                true_labels = y
            else:
                true_labels = np.concatenate([true_labels, y])

        if evaluation:
            Network.class_confusion(predictions, true_labels, len(LABELS))
        if len(scores) > 1:
            scores = np.concatenate(scores, axis=0)
        elif len(scores) == 1:
            scores = scores[0]
        return predictions, scores, pair_indexes
