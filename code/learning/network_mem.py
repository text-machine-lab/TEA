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

sys.path.append(os.path.join(os.environ["TEA_PATH"], '..', 'ntm_keras'))
from ntm import NeuralTuringMachine as NTM
from ntm import controller_input_output_shape as controller_shape

LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IBEFORE", "IAFTER", "IS_INCLUDED", "INCLUDES",
          "DURING", "BEGINS", "BEGUN_BY", "ENDS", "ENDED_BY", "None"]
EMBEDDING_DIM = 300
DENSE_LABELS = True


def get_lstm_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=300):

    controller = Sequential()
    controller.name = 'LSTM'

    controller.add(LSTM(units=controller_output_dim,  # 2x if using Bidrectional
                        kernel_initializer='random_normal',
                        bias_initializer='random_normal',
                        activation='linear',
                        stateful=True,  # must be true, because in controller every step is a batch?
                        return_sequences=False,  # does not matter because for controller the sequence len is 1?
                        implementation=2,  # best for gpu. other ones also might not work.
                        batch_input_shape=(batch_size, max_steps, controller_input_dim)))
    #
    # controller.add(MaxPooling1D(pool_size=max_steps, padding='same'))
    # controller.add(Flatten())  # (1, controller_output_dim)
    # controller.add(Dropout(0.5))

    controller.summary()
    controller.compile(loss='binary_crossentropy', optimizer=Adam(lr=.0005, clipnorm=10.), metrics=['binary_accuracy'])

    return controller

def get_dense_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=300):

    controller = Sequential()
    controller.name = 'Dense'

    controller.add(Dense(units=controller_output_dim,
                        activation='relu',
                        bias_initializer='zeros',
                        input_shape=(controller_input_dim,)))

    controller.summary()
    controller.compile(loss='binary_crossentropy', optimizer=Adam(lr=.0005, clipnorm=10.), metrics=['binary_accuracy'])

    return controller

class NetworkMem(Network):
    def __init__(self):
        super(NetworkMem, self).__init__()

    def get_untrained_model(self, encoder_dropout=0, decoder_dropout=0, input_dropout=0, LSTM_size=32, dense_size=256,
                            max_len=15, nb_classes=13):

        raw_input_l = Input(shape=(max_len, EMBEDDING_DIM))  # (steps, EMBEDDING_DIM)
        raw_input_r = Input(shape=(max_len, EMBEDDING_DIM))

        input_l = Dropout(input_dropout)(raw_input_l)
        input_r = Dropout(input_dropout)(raw_input_r)

        ## option 1: two branches
        # encoder_l = LSTM(LSTM_size, return_sequences=True)(input_l)
        # # encoder_l = LSTM(LSTM_size, return_sequences=True)(encoder_l)
        # encoder_l = MaxPooling1D(pool_size=max_len)(encoder_l)  # (1, LSTM_size)
        # encoder_l = Flatten()(encoder_l)
        # encoder_r = LSTM(LSTM_size, return_sequences=True)(input_r)
        # # encoder_r = LSTM(LSTM_size, return_sequences=True)(encoder_r)
        # encoder_r = MaxPooling1D(pool_size=max_len)(encoder_r)  # (1, LSTM_size)
        # encoder_r = Flatten()(encoder_r)
        # encoder = concatenate([encoder_l, encoder_r])  # (2*LSTM_size)

        ## option 2: no branch
        input = concatenate([input_l, input_r], axis=-2)
        encoder = Bidirectional(LSTM(LSTM_size, return_sequences=True))(input)
        encoder = Bidirectional(LSTM(LSTM_size, return_sequences=True), merge_mode='sum')(encoder)
        encoder = MaxPooling1D(pool_size=max_len)(encoder)  # (1, 2*LSTM_size)
        encoder = Flatten()(encoder)

        hidden = Dense(dense_size, activation='relu')(encoder)
        hidden = Dropout(decoder_dropout)(hidden)
        softmax = Dense(nb_classes, activation='softmax')(hidden)

        model = Model(inputs=[raw_input_l, raw_input_r], outputs=[softmax])

        # compile the final model
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_ntm_model(self, batch_size=100, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15, read_heads=1, write_heads=1, nb_classes=13,
                      input_dropout=0.5):

        left_branch = Sequential()
        left_branch.add(Dropout(input_dropout, batch_input_shape=(1, batch_size, max_len, EMBEDDING_DIM)))
        left_branch.add(TimeDistributed(LSTM(128, return_sequences=True)))
        left_branch.add(TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))) #(1, batch_size, 1, 128)
        left_branch.add(Reshape((batch_size, -1))) #(1, batch_size, 128)

        right_branch = Sequential()
        right_branch.add(Dropout(input_dropout, batch_input_shape=(1, batch_size, max_len, EMBEDDING_DIM)))
        right_branch.add(TimeDistributed(LSTM(128, return_sequences=True)))
        right_branch.add(TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same')))
        right_branch.add(Reshape((batch_size, -1)))  # (1, batch_size, 128)

        model = Sequential()
        model.add(Merge([left_branch, right_branch], mode='concat', concat_axis=-1))  # (1, batch_size, 256)
        # model.add(Dropout(0.5, batch_input_shape=(batch_size, max_len, 2*EMBEDDING_DIM)))

        controller_input_dim, controller_output_dim = controller_shape(256, ntm_output_dim, m_depth,
                                                                       n_slots, shift_range, read_heads, write_heads)

        # we feed in controller (# documents, # pairs, data_dim)
        # so max_steps here is # pairs
        controller = get_lstm_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=batch_size)
        # controller = get_dense_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=batch_size)

        model.name = "NTM_-_" + controller.name

        ntm = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                  read_heads=read_heads, write_heads=write_heads, controller_model=controller,
                  return_sequences=True, input_shape=(batch_size, 256),
                  activation='sigmoid', batch_size=1)

        model.add(ntm)
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(nb_classes, activation='softmax')))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    def slice_data(self, generator, batch_size):
        """Slice data into equal batch sizes
        Left-over small chunks will be augmented with random samples
        """
        while True:
            feed = generator.next() # x, y, <pair_index>
            data = feed[:3]
            N = len(data[-1])
            i = 0
            while (i+1) * batch_size <= N:
                if len(feed) == 4: # has pair_index, test data
                    if (i + 1) * batch_size == N:
                        marker = -1  # end of note
                    else:
                        marker = 0
                    # we need to add a dummy dimension to make the data 4D
                    out_data = [np.expand_dims(item[i*batch_size:(i+1)*batch_size], axis=0) for item in data]
                    yield [out_data[0], out_data[1]], out_data[2], feed[-1], marker
                else:
                    out_data = [np.expand_dims(item[i*batch_size:(i+1)*batch_size], axis=0) for item in data]
                    yield [out_data[0], out_data[1]], out_data[2]
                i += 1

            left_over = N % batch_size
            if left_over != 0:
                to_add = batch_size - left_over
                indexes_to_add = np.random.choice(N, to_add) # randomly sample more instances
                indexes = np.concatenate((np.arange(i*batch_size, N), indexes_to_add))
                if len(feed) == 4:
                    marker = left_over
                    out_data = [np.expand_dims(item[indexes], axis=0) for item in data]
                    yield [out_data[0], out_data[1]], out_data[2], feed[-1], marker
                else:
                    out_data = [np.expand_dims(item[indexes], axis=0) for item in data]
                    yield [out_data[0], out_data[1]], out_data[2]


    def generate_training_input(self, notes, pair_type, max_len=15, nolink_ratio=None, no_ntm=False):
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
        for note in notes:

            XL, XR, id_pairs = self._extract_path_representations(note, self.word_vectors, pair_type)

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

            for index, pair in enumerate(id_pairs):
                if pair in id_to_labels:
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

            if XL is not None and XL.shape[0] != 0:
                XL = pad_sequences(XL, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                XR = pad_sequences(XR, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
            else:
                continue

            labels = to_categorical(self._convert_str_labels_to_int(note_labels), len(LABELS))

            if no_ntm:
                data_q.append(([XL, XR], labels))
            else:
                data_q.append((XL, XR, labels))

        while True:
            yield data_q[0]
            data_q.rotate(-1)

    def generate_test_input(self, notes, pair_type, max_len=15, no_ntm=False):

        if self.word_vectors is None:
            print('Loading word embeddings...')
            self.word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin',
                                                     verbose=0)

        data_q = deque()
        for i, note in enumerate(notes):
            pair_index = {}  # record note id and all the used entity pairs

            XL, XR, id_pairs = self._extract_path_representations(note, self.word_vectors, pair_type)

            if DENSE_LABELS:
                id_to_labels = note.id_to_denselabels  # use TimeBank-Dense labels
            else:
                id_to_labels = note.id_to_labels

            if id_to_labels:
                note_labels = []
                index_to_reverse = []
                for index, pair in enumerate(id_pairs):  # id pairs that have tlinks

                    label_from_file = id_to_labels.get(pair, 'None')
                    opposite_from_file = id_to_labels.get((pair[1], pair[0]), 'None')
                    if label_from_file == 'None' and opposite_from_file != 'None':
                        index_to_reverse.append(index)
                        note_labels.append(opposite_from_file)  # save the opposite lable first, reverse later
                    else:
                        note_labels.append(label_from_file)

                labels = self._convert_str_labels_to_int(note_labels)
                labels_to_reverse = [labels[x] for x in index_to_reverse]
                reversed = self.reverse_labels(labels_to_reverse)
                print(note.annotated_note_path)
                print("{} labels augmented".format(len(reversed)))

                labels = np.array(labels, dtype='int16')
                index_to_reverse = np.array(index_to_reverse)
                if index_to_reverse.any():
                    labels[index_to_reverse] = reversed

                for index, pair in enumerate(id_pairs):
                    pair_index[(i, pair)] = index # map pairs to their index in data array, i is index for notes

                if XL is not None and XL.shape[0] != 0:
                    XL = pad_sequences(XL, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                    XR = pad_sequences(XR, maxlen=max_len, dtype='float32', padding='post', truncating='pre', value=0.)
                else:
                    continue

                if no_ntm:
                    data_q.append(([XL, XR], labels, pair_index))
                else:
                    data_q.append((XL, XR, labels, pair_index))

        while True:
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
                model = self.get_untrained_model(encoder_dropout=encoder_dropout, decoder_dropout=decoder_dropout,
                                                 input_dropout=input_dropout, LSTM_size=LSTM_size, dense_size=dense_size,
                                                 max_len=max_len, nb_classes=len(LABELS))

            else:
                model = self.get_ntm_model(batch_size=batch_size, m_depth=256, n_slots=128, ntm_output_dim=128, shift_range=3, max_len=15, read_heads=1, write_heads=1, nb_classes=13)
        # train the network
        print('Training network...')

        training_history = model.fit_generator(input_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                               callbacks=callbacks, validation_data=val_generator,
                                               validation_steps=validation_steps, class_weight=None, max_q_size=300, workers=1)

        return model, training_history.history

    def predict(self, model, data_generator, batch_size=300, evaluation=True, smart=True, no_ntm=False):
        predictions = []
        true_labels = None
        probs_in_note = None
        labels_in_note = []
        end_of_note = False

        while True:
            if no_ntm:
                X, y, pair_index = data_generator.next()
                marker = -1
            else:
                X, y, pair_index, marker = data_generator.next()

            note_index = pair_index.keys()[0][0]
            if end_of_note and note_index == 0: # all data consumed
                break

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
                probs_in_note = None
                labels_in_note = []

            if true_labels is None:
                true_labels = y
            else:
                true_labels = np.concatenate([true_labels, y])

        if evaluation:
            Network.class_confusion(predictions, true_labels, len(LABELS))
        return predictions
