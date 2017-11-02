from __future__ import print_function
import os
import sys
import time
import copy
import numpy as np
np.random.seed(1337)

from sklearn.metrics import precision_recall_fscore_support
from network import Network
from ntm_models import *

TYPE_MARKERS = {'_INTRA_': 0, '_CROSS_': 1, '_DCT_': -1}

class NetworkMem(Network):
    def __init__(self, no_ntm=False, nb_training_files=None):
        super(NetworkMem, self).__init__()
        self.counter = 0
        self.nb_training_files = nb_training_files
        self.nb_test_files = None
        self.test_data_collection = []
        self.no_ntm = no_ntm

    def slice_data(self, feed, batch_size, shift=0):
        """Slice data into equal batch sizes
        Left-over small chunks will be augmented with random samples
        """
        # while True:
            # feed = generator.next() # XL, XR, type_markers, y, <pair_index>

        if self.no_ntm: # for no_ntm models, no need to slice
            yield feed
        else:
            data = feed[:4]
            # circshift training data, so each time it starts with a different batch
            # if shift value is not a multiple of batch size, then the batches will be changing too
            if shift > 0 and len(feed) == 4:
                roll = shift * self.counter / self.nb_training_files #  accumulate shift after each epoch
                data = [np.roll(item, roll, axis=0) for item in data]
                self.counter += 1

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
        self.nb_training_files = len(notes)
        self.no_ntm = no_ntm

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
                if label[LABELS.index('None')] == 1:
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

        self.nb_test_files = len(notes)

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
                    if label == LABELS.index('None'): count_none += 1
                    else: count += 1

        print("Positive instances:", count)
        print("Negative instances:", count_none)

        while True:
            for i in range(multiple):
                yield data_q[0]
            data_q.rotate(-1)

    def train_model(self, model=None, no_ntm=False, epochs=100, steps_per_epoch=10, validation_steps=10,
                    input_generator=None, val_generator=None, weight_classes=False,
                    encoder_dropout=0.5, decoder_dropout=0.5, input_dropout=0.5, LSTM_size=128, dense_size=128,
                    max_len='auto', nb_classes=13, callbacks={}, batch_size=300, has_auxiliary=False):

        # infer maximum sequence length
        if max_len == 'auto':
            max_len = None

        if model is None:
            if no_ntm:
                model = get_untrained_model2(encoder_dropout=encoder_dropout, decoder_dropout=decoder_dropout,
                                            input_dropout=input_dropout, LSTM_size=LSTM_size, dense_size=dense_size,
                                            max_len=max_len, nb_classes=len(LABELS))

            else:
                model = get_ntm_model3_1(batch_size=batch_size, m_depth=256, n_slots=128, ntm_output_dim=128,
                                       shift_range=3, max_len=15, read_heads=2, write_heads=1, nb_classes=len(LABELS), has_auxiliary=has_auxiliary)
        # train the network
        print('Training network...')
        training_history = []
        best_result = None
        epochs_over_best = 0

        for epoch in range(epochs):
            epoch_history = []
            start = time.time()
            for n in range(self.nb_training_files):
                note_data = input_generator.next()
                for sliced in self.slice_data(note_data, batch_size=batch_size, shift=batch_size+11):
                    x = sliced[0]
                    y = sliced[1]

                    if has_auxiliary:
                        assert len(y.shape) in (2, 3)
                        if len(y.shape) == 3:
                            assert y.shape[0] == 1 # must be 3D with a dummy dimension
                            y_aux = np.zeros((y.shape[0], y.shape[1], 2))
                            for i, item in enumerate(y[0]):
                                if item[-1] == 1:
                                    y_aux[0, i, -1] = 1
                                else:
                                    y_aux[0, i, 0] = 1
                        else:
                            y_aux = np.zeros((y.shape[0], 2))
                            for i, item in enumerate(y):
                                if item[-1] == 1:
                                    y_aux[i, -1] = 1
                                else:
                                    y_aux[i, 0] = 1
                        y = [y, y_aux]

                    history = model.fit(x, y, batch_size=32, epochs=1, verbose=0, validation_split=0.0,
                              validation_data=None, shuffle=False, class_weight=None, sample_weight=None,
                              initial_epoch=0)
                    # print(history.history)
                    epoch_history.append(history.history)

                    # training_history = model.fit_generator(input_generator, steps_per_epoch=steps_per_epoch, epochs=1,
                    #                                        callbacks=callbacks, validation_data=val_generator,
                    #                                        validation_steps=validation_steps, class_weight=None, max_q_size=300,
                    #                                        workers=1)

                # reset states after a note file is processed
                model.reset_states()
                if 'main_out_acc' in history.history:
                    main_affix = 'main_out_'
                else:
                    main_affix = ''
                acc = np.mean([h[main_affix+'acc'][0] for h in epoch_history])
                loss = np.mean([h[main_affix+'loss'][0] for h in epoch_history])
                    
                if has_auxiliary:
                    aux_acc = np.mean([h['aux_out_acc'][0] for h in epoch_history])
                    # aux_loss = np.mean([h['aux_out_loss'][0] for h in epoch_history])
                    sys.stdout.write("epoch %d after training file %d/%d--- -%ds - main_loss : %.4f - main_acc : %.4f - aux_acc : %.4f\r" % (
                    epoch + 1, n, self.nb_training_files, int(time.time() - start), loss, acc, aux_acc))
                else:
                    sys.stdout.write("epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f\r" % (
                    epoch + 1, n, self.nb_training_files, int(time.time() - start), loss, acc))
                sys.stdout.flush()

            training_history.append({'acc': acc, 'loss': loss})
            print("\n\nepoch finished... evaluating on val data...")
            # evaluate after each epoch
            eval = self.predict(model, val_generator, batch_size=batch_size, evaluation=True, smart=False, no_ntm=no_ntm, has_auxiliary=has_auxiliary)

            if callbacks['earlystopping'].monitor == 'loss':
                if best_result is None or best_result > loss:
                    best_result = loss
                    epochs_over_best = 0
                    model.save_weights(callbacks['checkpoint'].filepath)
                else:
                    epochs_over_best += 1
            elif callbacks['earlystopping'].monitor == 'acc':
                if best_result is None or best_result < acc:
                    best_result = acc
                    epochs_over_best = 0
                    model.save_weights(callbacks['checkpoint'].filepath)
                else:
                    epochs_over_best += 1
            else:
                if best_result is None or best_result < eval[0]:  # just use evaluation accuracy
                    best_result = eval[0]
                    epochs_over_best = 0
                    model.save_weights(callbacks['checkpoint'].filepath)
                else:
                    epochs_over_best += 1
            print("%d epochs over the best result" % epochs_over_best)
            if epochs_over_best > callbacks['earlystopping'].patience:
                break

        print("Fisnished training. Data rolled %d times" %self.counter)
        return model, training_history

    def predict(self, model, data_generator, batch_size=300, evaluation=True, smart=True, no_ntm=False, has_auxiliary=False):
        predictions = []
        scores = []
        pair_indexes = {}
        true_labels = None
        probs_in_note = None
        labels_in_note = []
        end_of_note = False
        # print("not using NTM", no_ntm)

        if len(self.test_data_collection) > 0:
            feeder = self.test_data_collection
            self.nb_test_files = len(feeder)
        else:
            feeder = data_generator

        all_notes_consumed = False
        for note_data in feeder:
            if all_notes_consumed:
                break
            if not isinstance(feeder, list):
                self.test_data_collection.append(note_data)  # store data for fast retrieval
            model.reset_states()
            for sliced in self.slice_data(note_data, batch_size=batch_size, shift=0):
                if no_ntm:
                    X, y, pair_index, path_name = sliced
                    # print("file name to predict", path_name)
                    marker = -1
                else:
                    X, y, pair_index, marker = sliced

                note_index = pair_index.keys()[0][0]
                if end_of_note and note_index == 0: # all data consumed
                    all_notes_consumed = True
                pair_indexes.update(pair_index)

                probs = model.predict(X, batch_size=batch_size) # keras functional API model predicts probs
                if has_auxiliary:
                    probs = probs[0]  # use main output only
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
            return precision_recall_fscore_support(true_labels, predictions, average='micro')
        if len(scores) > 1:
            scores = np.concatenate(scores, axis=0)
        elif len(scores) == 1:
            scores = scores[0]
        return predictions, scores, pair_indexes

    def reverse_labels(self, labels):
        # LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IS_INCLUDED", "INCLUDES", "None"] # TimeBank Dense labels
        processed_labels = []

        for label in labels:
            if label in self.label_reverse_map:
                processed_labels.append(self.label_reverse_map[label])
                continue

            if label == 0:
                processed_labels.append(0)
            elif label == 1:
                processed_labels.append(2)
            elif label == 2:
                processed_labels.append(1)
            elif label == 3:
                processed_labels.append(4)
            elif label == 4:
                processed_labels.append(3)
            else:  # label for unlinked pairs (should have int 0)
                processed_labels.append(5)

            self.label_reverse_map[label] = processed_labels[-1]

        return processed_labels
