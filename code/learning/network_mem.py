from __future__ import print_function
import time
import copy
import numpy as np
np.random.seed(1337)
import math

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support
from network import Network
from word2vec import load_word2vec_binary, build_vocab
from ntm_models import *
from code.learning.time_ref import predict_timex_rel, TimeRefNetwork
from code.learning.break_cycle import modify_tlinks
from collections import deque
# import torch

TYPE_MARKERS = {'_INTRA_': 0, '_CROSS_': 1, '_DCT_': -1}

class NetworkMem(Network):
    def __init__(self, no_ntm=False, nb_training_files=None):
        super(NetworkMem, self).__init__()
        self.roll_counter = 0
        self.nb_training_files = nb_training_files
        self.nb_test_files = None
        self.test_data_collection = []
        self.no_ntm = no_ntm
        self.infersent = None
        self.training_passes = 1
        self.test_passes = 1

    def build_wordvectors(self, notes):
        print("Building word vectors from %d note files" % len(notes))
        if not notes:
            self.word_vectors = {}
        else:
            self.word_vectors = build_vocab([item.original_text for item in notes],
                                            os.environ["TEA_PATH"] + 'embeddings/glove.840B.300d.txt')
            self.word_vectors['_EVENT_'] = np.ones(EMBEDDING_DIM)
            self.word_vectors['_TIMEX_'] = - np.ones(EMBEDDING_DIM)
            self.word_vectors['UKN'] = np.random.uniform(-0.5, 0.5, EMBEDDING_DIM)

    def slice_data(self, feed, batch_size, shift=0):
        """Slice data into equal batch sizes
        Left-over small chunks will be augmented with random samples
        """

        data = feed[:7]

        if batch_size == 0: # special case, no slicing but expand the dimensions
            out_data = [np.expand_dims(item, axis=0) for item in data]
            if len(feed) == 8:
                marker = -1
                yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4], out_data[5]], out_data[6], feed[-1], marker
            else:
                yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4], out_data[5]], out_data[6]
        else:

            # circshift training data, so each time it starts with a different batch
            # if shift value is not a multiple of batch size, then the batches will be changing too
            if shift > 0 and len(feed) == 7:  # for training data only
                roll = shift * self.roll_counter / self.nb_training_files #  accumulate shift after each epoch
                data = [np.roll(item, roll, axis=0) for item in data]
                self.roll_counter += 1

            N = len(data[-1])
            i = 0
            while (i+1) * batch_size <= N:
                # we need to add a dummy dimension to make the data 4D
                out_data = [np.expand_dims(item[i * batch_size:(i + 1) * batch_size], axis=0) for item in data]
                if len(feed) == 8: # has pair_index, test data
                    if (i + 1) * batch_size == N:
                        marker = -1  # end of note
                    else:
                        marker = 0
                    yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4], out_data[5]], out_data[6], feed[-1], marker
                else:
                    # randomize training data within a batch (micro-randomization)
                    # rng_state = np.random.get_state()
                    # for data_index in range(6):
                    #     np.random.shuffle(out_data[data_index])
                    #     np.random.set_state(rng_state)

                    yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4], out_data[5]], out_data[6]
                i += 1

            left_over = N % batch_size

            # if left_over != 0:
            #     out_data = [np.expand_dims(item[i * batch_size:N], axis=0) for item in data]
            #     if len(feed) == 8:
            #         marker = left_over
            #         yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4], out_data[5]], out_data[6], feed[-1], marker
            #     else:
            #         yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4], out_data[5]], out_data[6]

            # for leftover, we add some random samples to make it a full batch
            if left_over != 0:
                to_add = batch_size - left_over
                indexes_to_add = np.random.choice(N, to_add) # randomly sample more instances
                indexes = np.concatenate((np.arange(i*batch_size, N), indexes_to_add))
                out_data = [np.expand_dims(item[indexes], axis=0) for item in data]
                if len(feed) == 8:
                    marker = left_over
                    yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4], out_data[5]], out_data[6], feed[-1], marker
                else:
                    yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4], out_data[5]], out_data[6]


    def generate_training_input(self, notes, pair_type, max_len=16, multiple=1, nolink_ratio=None, no_ntm=False):
        """
        Input data generator. Each step generates all data from one note file.
        Training data will be in narrative order
        """
        print("Generating training data...")
        # data tensor for left and right SDP subpaths
        XL = None
        XR = None
        self.nb_training_files = len(notes)
        self.no_ntm = no_ntm
        self.training_passes = multiple

        assert self.word_vectors
        # print('Loading word embeddings for training data...')
        # training_vocab = build_vocab([item.original_text for item in notes],
        #                                     os.environ["TEA_PATH"] + 'embeddings/glove.840B.300d.txt')
        # if self.word_vectors is None:
        #     self.word_vectors = training_vocab
        # else:
        #     self.word_vectors.update(training_vocab)

        data_q = deque()
        count = 0
        count_none = 0
        for note in notes:

            XL, XR, id_pairs, type_markers = self._extract_path_representations(note, self.word_vectors, pair_type, use_shortest=True)
            type_markers = np.expand_dims(np.array([TYPE_MARKERS[item] for item in type_markers]), axis=-1) # convert to 2D
            # context = self.extract_sentence_representations(note, id_pairs)
            context_L, context_R, _, _ = self._extract_path_representations(note, self.word_vectors, pair_type, use_shortest=False)
            time_differences = self.extract_timex_representations(note, id_pairs)  # (pairs, 3)

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
                XL = XL[training_indexes]
                XR = XR[training_indexes]
                note_labels = note_labels[training_indexes]
                type_markers = type_markers[training_indexes]
                context_L = context_L[training_indexes]
                context_R = context_R[training_indexes]
                time_differences = time_differences[training_indexes]

            if XL is not None and XL.shape[0] != 0:
                XL = pad_sequences(XL, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                XR = pad_sequences(XR, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                context_L = pad_sequences(context_L, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                context_R = pad_sequences(context_R, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
            else:
                continue

            labels = to_categorical(self._convert_str_labels_to_int(note_labels), len(LABELS))

            data_q.append([XL, XR, type_markers, context_L, context_R, time_differences, labels])

            for label in labels:
                if label[LABELS.index('None')] == 1:
                    count_none += 1
                else:
                    count += 1

        print("Positive instances:", count)
        print("Negative instances:", count_none)
        if nolink_ratio is not None:
            assert count_none <= count * nolink_ratio

        sequence = range(self.nb_training_files)
        while True:
            np.random.shuffle(sequence) # shuffle training data
            for index in sequence:
                data = copy.copy(data_q[index])
                data_copy = copy.copy(data)
                for m in range(multiple-1):
                    for i, item in enumerate(data_copy):
                        data[i] = np.concatenate([data[i], item], axis=0)
                yield data

    def generate_test_input(self, notes, pair_type, max_len=16, multiple=1, no_ntm=False, reverse_pairs=True):

        print("\nGenerating test data...")
        self.nb_test_files = len(notes)
        self.test_notes = notes
        self.test_passes = multiple

        assert self.word_vectors
        # print('Loading word embeddings for test data...')
        # test_vocab = build_vocab([item.original_text for item in notes],
        #                                     os.environ["TEA_PATH"] + 'embeddings/glove.840B.300d.txt')
        # if self.word_vectors is None:
        #     self.word_vectors = test_vocab
        # else:
        #     self.word_vectors.update(test_vocab)

        data_q = deque()
        count_none = 0
        count = 0
        for i, note in enumerate(notes):
            pair_index = {}  # record note id and all the used entity pairs

            XL, XR, id_pairs, type_markers = self._extract_path_representations(note, self.word_vectors, pair_type, use_shortest=True)
            type_markers = np.expand_dims(np.array([TYPE_MARKERS[item] for item in type_markers]), axis=-1)
            # context = self.extract_sentence_representations(note, id_pairs)
            context_L, context_R, _, _ = self._extract_path_representations(note, self.word_vectors, pair_type, use_shortest=False)
            time_differences = self.extract_timex_representations(note, id_pairs)  # (pairs, 3)

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
                    pair_index[(i, pair)] = index # map pairs to their index in data array (within a note), i is id for notes

                if XL is not None and XL.shape[0] != 0:
                    XL = pad_sequences(XL, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                    XR = pad_sequences(XR, maxlen=max_len, dtype='float32', padding='post', truncating='pre', value=0.)
                    context_L = pad_sequences(context_L, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                    context_R = pad_sequences(context_R, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                else:
                    continue

                data_q.append([XL, XR, type_markers, context_L, context_R, time_differences, labels, pair_index])

                for label in labels:
                    if label == LABELS.index('None'): count_none += 1
                    else: count += 1

        print("Positive instances:", count)
        print("Negative instances:", count_none)

        while True:
            data = copy.copy(data_q[0])
            data_copy = copy.copy(data)
            for m in range(multiple-1):
                for i, item in enumerate(data_copy[:7]):  # pair_index not changed. so it applies to one multiple only
                    data[i] = np.concatenate([data[i], item], axis=0)
            yield data
            data_q.rotate(-1)

    def extract_timex_representations(self, note, id_pairs):
        time_ref = TimeRefNetwork(note)

        def get_time_interval(time_id):
            if time_id[0] != 't':
                return [-1, 0, 0]  # -1 means the interval is not real
            if time_id not in time_ref.timex_elements:
                # some cannot be found because they are "DURATION" not "DATA"
                # print("timex id not found:", note.annotated_note_path, time_id)
                return [-1, 0, 0]
            time_pair = time_ref.transform_value(time_ref.timex_elements[time_id]['value'])
            if time_pair is None:
                return [-1, 0, 0]
            else:
                return [1, time_pair[0], time_pair[1]]

        time_differences = []
        for pair in id_pairs:
            interval_l = get_time_interval(pair[0])
            interval_r = get_time_interval(pair[1])
            if interval_l[0] == -1 or interval_r[0] == -1:
                time_differences.append([-1, 0, 0])
            else:
                time_differences.append([1, interval_l[1]-interval_r[1], interval_l[2]-interval_r[2]])
        return np.sign(time_differences)

        # timex_values_l = []
        # timex_values_r = []
        # for pair in id_pairs:
        #     timex_values_l.append(get_time_interval(pair[0]))
        #     timex_values_r.append(get_time_interval(pair[1]))
        #
        # return np.array(timex_values_l), np.array(timex_values_r)

    def load_raw_model(self, no_ntm, fit_batch_size=10):
        if no_ntm:
            model = get_pre_ntm_model(group_size=None, nb_classes=len(LABELS), input_dropout=0.3, max_len=MAX_LEN,
                                      embedding_matrix=self.get_embedding_matrix())
        else:
            model = get_ntm_hiddenfeed(batch_size=fit_batch_size, group_size=None, m_depth=512, n_slots=128,
                                     ntm_output_dim=512, shift_range=3, max_len=MAX_LEN, read_heads=1, write_heads=1,
                                     nb_classes=len(LABELS), embedding_matrix=self.get_embedding_matrix())
        return model

    def train_model(self, model=None, no_ntm=False, epochs=100, steps_per_epoch=10, input_generator=None,
                    val_generator=None, weight_classes=False, encoder_dropout=0.5, decoder_dropout=0.5,
                    input_dropout=0.5, LSTM_size=128, dense_size=128, max_len='auto', nb_classes=13,
                    callbacks={}, batch_size=300, has_auxiliary=False):

        # learning rate schedule
        def step_decay(epoch):
            # initial_lrate = 0.0002
            initial_lrate = 0.0002
            drop = 0.5
            epochs_drop = 5  # get half in every epochs_drop steps
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        fit_batch_size = 10  # group batches

        if model is None:
            model = self.load_raw_model(no_ntm, fit_batch_size)

        print('Training network...')
        training_history = []
        best_result = None
        epochs_over_best = 0
        batch_sizes = [batch_size/2, batch_size*3/4, batch_size, batch_size*3/2, batch_size*2]
        for epoch in range(epochs):
            #print("check weights", model.get_weights()[8][0:5])
            lr = step_decay(epoch)
            print("set learning rate %f" %lr)
            model.optimizer.lr.assign(lr)
            #K.set_value(model.optimizer.lr, lr)

            epoch_history = []
            start = time.time()
            for n in range(self.nb_training_files):
                note_data = input_generator.next()
                batch_size = np.random.choice(batch_sizes)
                X = None
                y = None
                for sliced in self.slice_data(note_data, batch_size=batch_size, shift=batch_size + 7):
                    if X is None:
                        X = sliced[0]
                        y = sliced[1]
                    else:
                        X = [np.concatenate([X[i], item], axis=0) for i, item in enumerate(sliced[0])]
                        y = np.concatenate([y, sliced[1]], axis=0)

                extra = y.shape[0] % fit_batch_size
                nb_fit_batches = y.shape[0]/fit_batch_size + (extra != 0)
                to_add = fit_batch_size - extra
                indexes_to_add = np.random.choice(y.shape[0], to_add)  # randomly sample more instances
                for index in indexes_to_add:
                    for i in range(len(X)):
                        X[i] = np.concatenate([X[i], X[i][index][None,:]], axis=0)
                    y = np.concatenate([y, y[index][None, :]])

                for m in range(nb_fit_batches):
                    X_b = [item[m*fit_batch_size: (m+1)*fit_batch_size] for item in X]
                    y_b = y[m*fit_batch_size: (m+1)*fit_batch_size]
                    if has_auxiliary:
                        assert len(y.shape) in (2, 3)
                        y_aux = y_b

                        y_b = [y_b, y_aux]

                    history = model.fit(X_b, y_b, batch_size=fit_batch_size, epochs=1, verbose=0, validation_split=0.0,
                              validation_data=None, shuffle=False, class_weight=None, sample_weight=None,
                              initial_epoch=0)
                    # print(history.history)
                    epoch_history.append(history.history)

                # reset states after a note file is processed
                model.reset_states()
                if 'main_out_categorical_accuracy' in history.history:
                    main_affix = 'main_out_'
                else:
                    main_affix = ''
                try:
                    acc = np.mean([h[main_affix+'categorical_accuracy'][0] for h in epoch_history])
                except KeyError:
                    print("KeyError. categorical_accuracy is not found. Correct keys:", epoch_history[-1].keys())
                loss = np.mean([h[main_affix+'loss'][0] for h in epoch_history])

                if has_auxiliary:
                    try:
                        aux_acc = np.mean([h['aux_out_categorical_accuracy'][0] for h in epoch_history])
                    except KeyError:
                        aux_acc = np.mean([h['pre_ntm_categorical_accuracy'][0] for h in epoch_history])
                    sys.stdout.write("epoch %d after training file %d/%d--- -%ds - main_loss : %.4f - main_acc : %.4f - aux_acc : %.4f\r" % (
                    epoch + 1, n, self.nb_training_files, int(time.time() - start), loss, acc, aux_acc))
                else:
                    sys.stdout.write("epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f\r" % (
                    epoch + 1, n, self.nb_training_files, int(time.time() - start), loss, acc))
                sys.stdout.flush()

            sys.stdout.write("\n")
            training_history.append({'categorical_accuracy': acc, 'loss': loss})
            if epoch <= 8:
                continue
            print("\n\nepoch finished... evaluating on val data...")
            if val_generator is not None:
                # evaluate after each epoch
                evalu = self.predict(model, val_generator, batch_size=0, fit_batch_size=fit_batch_size, evaluation=True, smart=False, no_ntm=no_ntm, has_auxiliary=has_auxiliary)

            if 'earlystopping' in callbacks:
                if callbacks['earlystopping'].monitor == 'loss':
                    if best_result is None or best_result > loss:
                        best_result = loss
                        epochs_over_best = 0
                        model.save_weights(callbacks['checkpoint'].filepath)
                    else:
                        epochs_over_best += 1
                elif callbacks['earlystopping'].monitor == 'categorical_accuracy':
                    if best_result is None or best_result < acc:
                        best_result = acc
                        epochs_over_best = 0
                        model.save_weights(callbacks['checkpoint'].filepath)
                    else:
                        epochs_over_best += 1
                else:
                    if best_result is None or best_result < evalu[0]:  # just use evaluation accuracy
                        best_result = evalu[0]
                        epochs_over_best = 0
                        model.save_weights(callbacks['checkpoint'].filepath)
                    else:
                        epochs_over_best += 1
                print("%d epochs over the best result" % epochs_over_best)
                if epochs_over_best > callbacks['earlystopping'].patience:
                    break

        print("Fisnished training. Data rolled %d times" %self.roll_counter)
        return model, training_history

    def predict(self, model, data_generator, batch_size=0, fit_batch_size=10, evaluation=True, smart=True, no_ntm=False, has_auxiliary=False, pruning=False):
        predictions = []
        scores = []
        pair_indexes = {}
        true_labels = []
        true_labels_in_note = None
        probs_in_note = None
        labels_in_note = []
        end_of_note = False

        if len(self.test_data_collection) > 0:
            feeder = self.test_data_collection
            self.nb_test_files = len(feeder)
        else:
            feeder = data_generator

        all_pred_timex_labels = None
        all_notes_consumed = False
        for i, note_data in enumerate(feeder):
            if all_pred_timex_labels is None:  # get timex labels from rule-based system
                all_pred_timex_labels, _, all_timex_pairs = get_timex_predictions(self.test_notes)

            if all_notes_consumed:
                break

            model.reset_states()
            for sliced in self.slice_data(note_data, batch_size=batch_size, shift=0):
                X, y, pair_index, marker = sliced
                X = [np.repeat(item, fit_batch_size, axis=0) for item in X]  # have to do this because the batch size must be fixed

                note_index = pair_index.keys()[0][0]
                if end_of_note and note_index == 0: # all data consumed
                    all_notes_consumed = True
                    break
                # pair_indexes.update(pair_index)

                # probs = model.predict(X, batch_size=1)
                probs = model.predict_on_batch(X)
                if has_auxiliary:
                    probs = probs[0]  # use main output only
                y = y[0, :]  # remove the dummy dimension
                probs = probs[0, :]

                if self.test_passes > 1:
                    data_len = len(y)/self.test_passes
                    probs = probs[-data_len:]  # only use the last pass
                    y = y[-data_len:]

                if marker > 0: # end of note with leftover
                    y = y[:marker]
                    probs = probs[:marker]
                labels = [np.argmax(x) for x in probs]

                if probs_in_note is None:
                    probs_in_note = probs
                    true_labels_in_note = y
                    labels_in_note = labels
                else:
                    probs_in_note = np.concatenate([probs_in_note, probs])
                    true_labels_in_note = np.concatenate([true_labels_in_note, y])
                    labels_in_note += labels

                if marker != 0: # end of note
                    end_of_note = True

                    # replace timex pairs with rule-based results
                    for key, value in pair_index:
                        if key in all_timex_pairs:
                            index = all_timex_pairs.index(key)
                            timex_label_rule = all_pred_timex_labels[index]  # result from rule-based system
                            if timex_label_rule != LABELS.index('None'):
                                labels_in_note[value] = timex_label_rule
                                probs_in_note[value, timex_label_rule] = 1

                    if smart:
                        labels_in_note, pair_index = self.smart_predict(labels_in_note, probs_in_note, pair_index, type='int')
                        labels_in_note, used_indexes, pair_index = self.remove_redudant_pairs(labels_in_note, pair_index)
                        probs_in_note = probs_in_note[used_indexes]
                        true_labels_in_note = true_labels_in_note[used_indexes]

                    if pruning:
                        print("Pruning note #%d out of %d %s" %(i+1, len(self.test_notes), self.test_notes[i].annotated_note_path))
                        # pred_timex_labels, true_timex_labels, timex_pairs = get_timex_predictions([self.test_notes[i]])
                        pairs_to_prune = []
                        labels_to_prune = []
                        scores_to_prune = []
                        true_labels_after_prune = []
                        for key in pair_index:  # key is (i, (pair))
                            pairs_to_prune.append(key)
                            labels_to_prune.append(labels_in_note[pair_index[key]])
                            scores_to_prune.append(max(probs_in_note[pair_index[key]]))
                            true_labels_after_prune.append(true_labels_in_note[pair_index[key]])
                        # pairs_to_prune += timex_pairs
                        # labels_to_prune += pred_timex_labels
                        # timex_scores = [0.2 if label==LABELS.index('None') else 1.0 for label in pred_timex_labels]
                        # scores_to_prune += timex_scores
                        pruned_labels_in_note = modify_tlinks([pair for note_id, pair in pairs_to_prune],
                                                              NetworkMem.convert_int_labels_to_str(labels_to_prune),
                                                              scores_to_prune)
                        pruned_labels_in_note = NetworkMem.convert_str_labels_to_int(pruned_labels_in_note)
                        # true_labels_after_prune += true_timex_labels

                        predictions += pruned_labels_in_note
                        scores += scores_to_prune
                        true_labels += true_labels_after_prune
                    else:
                        predictions += labels_in_note
                        scores.append(probs_in_note)
                        true_labels.append(true_labels_in_note)

                    pair_indexes.update(pair_index)
                    probs_in_note = None
                    true_labels_in_note = None
                    labels_in_note = []

            if not isinstance(feeder, list):
                self.test_data_collection.append(note_data)  # store data for fast retrieval

        if not pruning:
            if len(scores) > 1:
                scores = np.concatenate(scores, axis=0)
                true_labels = np.concatenate(true_labels, axis=0)
            elif len(scores) == 1:
                scores = scores[0]
                true_labels = true_labels[0]
        if evaluation:
            Network.class_confusion(predictions, true_labels, len(LABELS))
            return precision_recall_fscore_support(true_labels, predictions, average='micro')

        return predictions, scores, true_labels, pair_indexes

    def remove_redudant_pairs(self, labels, pair_index):
        new_pair_index = {}
        for key in pair_index:
            note_id, pair = key
            if key in new_pair_index or (note_id, (pair[1], pair[0])) in new_pair_index:
                continue
            else:
                new_pair_index[key] = pair_index[key]

        indexes = sorted([v for k, v in new_pair_index.iteritems()])
        new_labels = [labels[i] for i in indexes]
        old_indexes_to_new = dict(zip(indexes, range(len(indexes))))
        for key in new_pair_index:
            new_pair_index[key] = old_indexes_to_new[new_pair_index[key]] # map old indexes to new indexes in the label list

        return new_labels, np.array(indexes), new_pair_index


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

    # def get_sentence_embeddings(self, sentences):
    #     if self.infersent is None:
    #         assert self.word_vectors is not None
    #         encoder_path = '/home/ymeng/projects/InferSent/encoder'
    #         sys.path.append(encoder_path)
    #         self.infersent = torch.load(os.path.join(encoder_path, 'infersent.allnli.pickle'), map_location=lambda storage, loc: storage)
    #         self.infersent.set_glove_path(os.environ["TEA_PATH"] + 'embeddings/glove.840B.300d.txt')
    #         self.infersent.build_vocab([''], tokenize=True) # must do this to get the infersent.word_vec object
    #         self.infersent.word_vec = self.word_vectors
    #
    #     embeddings = self.infersent.encode(sentences, tokenize=True)
    #     return embeddings

    def extract_sentence_representations(self, note, id_pairs):
        sentences = []
        for sent_num in note.pre_processed_text:
            sentence = ' '.join([x['token'] for x in note.pre_processed_text[sent_num]])
            sentences.append(sentence)

        sentence_embs = self.get_sentence_embeddings(sentences)

        context = []
        for src_id, target_id in id_pairs:
            left_sentence = note.id_to_sent[src_id]
            right_sentence = note.id_to_sent[target_id]

            if left_sentence is not None:
                left_context = sentence_embs[left_sentence-1]  # sentence ids starts with 1
            else:
                left_context = np.zeros(4096)

            if left_sentence == right_sentence or right_sentence is None:  # 't0' can be on the right only
                context.append(left_context)
            else:
                right_context = sentence_embs[right_sentence-1]  # sentence ids starts with 1
                context.append(np.mean([left_context, right_context], axis=0))

        return np.array(context)

    @staticmethod
    def convert_str_labels_to_int(labels):
        for i, label in enumerate(labels):
            if label == "IDENTITY":
                labels[i] = "SIMULTANEOUS"
            elif label not in LABELS:
                labels[i] = "None"

        return [LABELS.index(x) for x in labels]

    @staticmethod
    def convert_int_labels_to_str(labels):
        '''
        convert ints to tlink labels so network output can be understood
        '''

        return [LABELS[s] if s < 12 else "None" for s in labels]

    def get_embedding_matrix(self, notes=None):
        if self.word_vectors is None:
            print('Loading word embeddings...')
            self.word_vectors = build_vocab([item.original_text for item in notes],
                                            os.environ["TEA_PATH"] + 'embeddings/glove.840B.300d.txt')
            self.word_vectors['_EVENT_'] = np.ones(EMBEDDING_DIM)
            self.word_vectors['_TIMEX_'] = - np.ones(EMBEDDING_DIM)

        self.word_indexes = {}
        embedding_matrix = np.random.uniform(low=-0.5, high=0.5, size=(len(self.word_vectors) + 1, EMBEDDING_DIM))
        for index, word in enumerate(sorted(self.word_vectors.keys())):
            self.word_indexes[word] = index + 1
            embedding_vector = self.word_vectors.get(word, None)
            embedding_matrix[index+1] = embedding_vector
        embedding_matrix[0] = np.zeros(EMBEDDING_DIM)  # used for mask/padding

        return embedding_matrix

    def _extract_path_representations(self, note, word_vectors, pair_type, use_shortest=True):

        if use_shortest:
            id_pair_to_path_words = self._extract_path_words(note, pair_type)
        else:
            id_pair_to_path_words = self._extract_context_words(note, pair_type)

        left_vecs = []
        right_vecs = []

        # get the word index for every word in the left path
        # must sort it to match the labels correctly
        sorted_pairs = Network.sort_id_pairs(note, id_pair_to_path_words.keys())
        type_markers = []
        for id_pair in sorted_pairs:
            type_marker = id_pair_to_path_words[id_pair][-1]
            assert type_marker in ('_INTRA_', '_CROSS_', '_DCT_')
            type_markers.append(type_marker)

            path = id_pair_to_path_words[id_pair][0]
            left_word_indexes = [self.word_indexes[word] if word in self.word_indexes else self.word_indexes['UKN'] for word in path]
            assert left_word_indexes
            left_vecs.append(left_word_indexes)

            path = id_pair_to_path_words[id_pair][1]
            right_word_indexes = [self.word_indexes[word] if word in self.word_indexes else self.word_indexes['UKN'] for word in path]
            # assert right_word_indexes
            right_vecs.append(right_word_indexes)

        return np.array(left_vecs), np.array(right_vecs), sorted_pairs, type_markers


def get_timex_predictions(notes):
    timex_labels, timex_pair_index = predict_timex_rel(notes)
    true_timex_labels = []
    pred_timex_labels = []
    timex_pairs = []
    for i, note in enumerate(notes):

        id_to_labels = note.id_to_denselabels  # If augmented, this is bidirectional, even for t0 pairs
        processed = {}  # used to remove redundant pairs

        # The id pairs in timex_pair_index are exactly the same as in note.timex_pairs
        # For TimeBank-Dense data, only labeled pairs are included
        for pair in note.timex_pairs:
            if pair in processed: continue
            if (i, pair) in timex_pair_index:
                timex_pairs.append((i, pair))
                pred_timex_labels.append(timex_labels[timex_pair_index[(i, pair)]])

                if pair in id_to_labels:
                    true_timex_labels.append(id_to_labels[pair])
                else:
                    true_timex_labels.append(LABELS.index("None"))
                    print("Timex pair in %s not found in true labels:" % note.annotated_note_path, pair)
            else:
                print("Timex pair in %s not found in timex_pair_index:" % note.annotated_note_path, pair)

            processed[pair] = 1
            processed[(pair[1], pair[0])] = 1
    pred_timex_labels = NetworkMem.convert_str_labels_to_int(pred_timex_labels)
    true_timex_labels = NetworkMem.convert_str_labels_to_int(true_timex_labels)

    return pred_timex_labels, true_timex_labels, timex_pairs
