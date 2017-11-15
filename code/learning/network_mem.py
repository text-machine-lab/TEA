from __future__ import print_function
import os
import sys
import time
import copy
import numpy as np
np.random.seed(1337)
import math

from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import LearningRateScheduler
from network import Network
from word2vec import load_word2vec_binary, build_vocab
from ntm_models import *
from code.learning.time_ref import predict_timex_rel
from code.learning.break_cycle import modify_tlinks
import torch

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

    def slice_data(self, feed, batch_size, shift=0):
        """Slice data into equal batch sizes
        Left-over small chunks will be augmented with random samples
        """
        # while True:
            # feed = generator.next() # XL, XR, type_markers, context_L, context_R, y, <pair_index>, only test data has pair_index

        if self.no_ntm: # for no_ntm models, no need to slice
            yield feed
        else:
            data = feed[:6]

            # circshift training data, so each time it starts with a different batch
            # if shift value is not a multiple of batch size, then the batches will be changing too
            if shift > 0 and len(feed) == 6:
                roll = shift * self.roll_counter / self.nb_training_files #  accumulate shift after each epoch
                data = [np.roll(item, roll, axis=0) for item in data]
                self.roll_counter += 1

            N = len(data[-1])
            i = 0
            while (i+1) * batch_size <= N:
                if len(feed) == 7: # has pair_index, test data
                    if (i + 1) * batch_size == N:
                        marker = -1  # end of note
                    else:
                        marker = 0
                    # we need to add a dummy dimension to make the data 4D
                    out_data = [np.expand_dims(item[i*batch_size:(i+1)*batch_size], axis=0) for item in data]
                    yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4]], out_data[5], feed[-1], marker
                else:
                    out_data = [np.expand_dims(item[i*batch_size:(i+1)*batch_size], axis=0) for item in data]
                    # randomize training data within a batch (micro-randomization)
                    rng_state = np.random.get_state()
                    for data_index in range(6):
                        np.random.shuffle(out_data[data_index])
                        np.random.set_state(rng_state)

                    yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4]], out_data[5]
                i += 1

            left_over = N % batch_size
            if left_over != 0:
                to_add = batch_size - left_over
                indexes_to_add = np.random.choice(N, to_add) # randomly sample more instances
                indexes = np.concatenate((np.arange(i*batch_size, N), indexes_to_add))
                if len(feed) == 7:
                    marker = left_over
                    out_data = [np.expand_dims(item[indexes], axis=0) for item in data]
                    yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4]], out_data[5], feed[-1], marker
                else:
                    out_data = [np.expand_dims(item[indexes], axis=0) for item in data]
                    yield [out_data[0], out_data[1], out_data[2], out_data[3], out_data[4]], out_data[5]


    def generate_training_input(self, notes, pair_type, max_len=15, multiple=1, nolink_ratio=None, no_ntm=False):
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

        if self.word_vectors is None:
            print('Loading word embeddings...')
            # self.word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin',
            #                                     verbose=0)
            self.word_vectors = build_vocab([item.original_text for item in notes], os.environ["TEA_PATH"] + 'embeddings/glove.840B.300d.txt')
        data_q = deque()
        count = 0
        count_none = 0
        for note in notes:

            XL, XR, id_pairs, type_markers = self._extract_path_representations(note, self.word_vectors, pair_type, use_shortest=True)
            type_markers = np.expand_dims(np.array([TYPE_MARKERS[item] for item in type_markers]), axis=-1) # convert to 2D
            # context = self.extract_sentence_representations(note, id_pairs)
            context_L, context_R, _, _ = self._extract_path_representations(note, self.word_vectors, pair_type, use_shortest=False)

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
                # context = context[training_indexes]
                context_L = context_L[training_indexes]
                context_R = context_R[training_indexes]

            if XL is not None and XL.shape[0] != 0:
                XL = pad_sequences(XL, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                XR = pad_sequences(XR, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                context_L = pad_sequences(context_L, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
                context_R = pad_sequences(context_R, maxlen=max_len, dtype='float32', padding='pre', truncating='post', value=0.)
            else:
                continue

            labels = to_categorical(self._convert_str_labels_to_int(note_labels), len(LABELS))

            if no_ntm:
                data_q.append([[XL, XR, type_markers, context_L, context_R], labels])
            else:
                data_q.append([XL, XR, type_markers, context_L, context_R, labels])

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
                if no_ntm:
                    for m in range(multiple-1):
                        for i, item in enumerate(data_copy[0]):
                            data[0][i] = np.concatenate([data[0][i], item], axis=0)
                        data[1] = np.concatenate([data[1], data_copy[1]])
                else:
                    for m in range(multiple-1):
                        for i, item in enumerate(data_copy):
                            data[i] = np.concatenate([data[i], item], axis=0)
                yield data

    def generate_test_input(self, notes, pair_type, max_len=15, multiple=1, no_ntm=False, reverse_pairs=True):

        print("Generating test data...")
        self.nb_test_files = len(notes)
        self.test_notes = notes
        self.test_passes = multiple

        if self.word_vectors is None:
            print('Loading word embeddings...')
            # self.word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin',
            #                                          verbose=0)
            self.word_vectors = build_vocab([item.original_text for item in notes],
                                            os.environ["TEA_PATH"] + 'embeddings/glove.840B.300d.txt')

        data_q = deque()
        count_none = 0
        count = 0
        for i, note in enumerate(notes):
            pair_index = {}  # record note id and all the used entity pairs

            XL, XR, id_pairs, type_markers = self._extract_path_representations(note, self.word_vectors, pair_type, use_shortest=True)
            type_markers = np.expand_dims(np.array([TYPE_MARKERS[item] for item in type_markers]), axis=-1)
            # context = self.extract_sentence_representations(note, id_pairs)
            context_L, context_R, _, _ = self._extract_path_representations(note, self.word_vectors, pair_type, use_shortest=False)

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

                if no_ntm:
                    data_q.append([[XL, XR, type_markers, context_L, context_R], labels, pair_index, note.annotated_note_path])
                else:
                    data_q.append([XL, XR, type_markers, context_L, context_R, labels, pair_index])

                for label in labels:
                    if label == LABELS.index('None'): count_none += 1
                    else: count += 1

        print("Positive instances:", count)
        print("Negative instances:", count_none)

        while True:
            data = copy.copy(data_q[0])
            data_copy = copy.copy(data)
            if no_ntm:
                for m in range(multiple-1):
                    for i, item in enumerate(data_copy[0]):
                        data[0][i] = np.concatenate([data[0][i], item], axis=0)
                    data[1] = np.concatenate([data[1], data_copy[1]])
            else:
                for m in range(multiple-1):
                    for i, item in enumerate(data_copy[:6]):  # pair_index not changed. so it applies to one multiple only
                        data[i] = np.concatenate([data[i], item], axis=0)
            yield data
            data_q.rotate(-1)

    def train_model(self, model=None, no_ntm=False, epochs=100, steps_per_epoch=10, validation_steps=10,
                    input_generator=None, val_generator=None, weight_classes=False,
                    encoder_dropout=0.5, decoder_dropout=0.5, input_dropout=0.5, LSTM_size=128, dense_size=128,
                    max_len='auto', nb_classes=13, callbacks={}, batch_size=300, has_auxiliary=False):


        # learning rate schedule
        def step_decay(epoch):
            initial_lrate = 0.002
            drop = 0.5
            epochs_drop = 4
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        # infer maximum sequence length
        if max_len == 'auto':
            max_len = None

        if model is None:
            if no_ntm:
                model = get_untrained_model4_2(encoder_dropout=encoder_dropout, decoder_dropout=decoder_dropout,
                                            input_dropout=input_dropout, LSTM_size=LSTM_size, dense_size=dense_size,
                                            max_len=max_len, nb_classes=len(LABELS))

            else:
                model = get_ntm_model4_1(batch_size=batch_size, m_depth=256, n_slots=128, ntm_output_dim=128,
                                       shift_range=3, max_len=15, read_heads=2, write_heads=1, nb_classes=len(LABELS), has_auxiliary=has_auxiliary)
        # train the network
        print('Training network...')
        training_history = []
        best_result = None
        epochs_over_best = 0

        for epoch in range(epochs):
            lr = step_decay(epoch)
            print("set learning rate %f" %lr)
            model.optimizer.lr.assign(lr)

            epoch_history = []
            start = time.time()
            for n in range(self.nb_training_files):
                note_data = input_generator.next()
                for sliced in self.slice_data(note_data, batch_size=batch_size, shift=0):
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

                    history = model.fit(x, y, batch_size=1, epochs=1, verbose=0, validation_split=0.0,
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
                try:
                    acc = np.mean([h[main_affix+'categorical_accuracy'][0] for h in epoch_history])
                except KeyError:
                    print("KeyError. categorical_accuracy is not found. Correct keys:", epoch_history[-1].keys())
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

            training_history.append({'categorical_accuracy': acc, 'loss': loss})
            print("\n\nepoch finished... evaluating on val data...")
            # evaluate after each epoch
            evalu = self.predict(model, val_generator, batch_size=batch_size, evaluation=True, smart=False, no_ntm=no_ntm, has_auxiliary=has_auxiliary)

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

    def predict(self, model, data_generator, batch_size=300, evaluation=True, smart=True, no_ntm=False, has_auxiliary=False, combine_timex=False):
        predictions = []
        scores = []
        pair_indexes = {}
        true_labels = []
        true_labels_in_note = None
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
        for i, note_data in enumerate(feeder):
            # predictions_in_note = []
            # scores_in_note = []
            if all_notes_consumed:
                break

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
                    break
                # pair_indexes.update(pair_index)

                probs = model.predict(X, batch_size=batch_size) # keras functional API model predicts probs
                if has_auxiliary:
                    probs = probs[0]  # use main output only
                if not no_ntm:
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
                    if smart:
                        labels_in_note, pair_index = self.smart_predict(labels_in_note, probs_in_note, pair_index, type='int')
                        labels_in_note, used_indexes, pair_index = self.remove_redudant_pairs(labels_in_note, pair_index)
                        probs_in_note = probs_in_note[used_indexes]
                        true_labels_in_note = true_labels_in_note[used_indexes]

                    if combine_timex:
                        print("combining timex pairs for note #%d out of %d %s" %(i+1, len(self.test_notes), self.test_notes[i].annotated_note_path))
                        pred_timex_labels, true_timex_labels, timex_pairs = get_timex_predictions([self.test_notes[i]])
                        pairs_to_prune = []
                        labels_to_prune = []
                        scores_to_prune = []
                        true_labels_after_prune = []
                        for key in pair_index:  # key is (i, (pair))
                            pairs_to_prune.append(key)
                            labels_to_prune.append(labels_in_note[pair_index[key]])
                            scores_to_prune.append(max(probs_in_note[pair_index[key]]))
                            true_labels_after_prune.append(true_labels_in_note[pair_index[key]])
                        pairs_to_prune += timex_pairs
                        labels_to_prune += pred_timex_labels
                        timex_scores = [0.2 if label==LABELS.index('None') else 1.0 for label in pred_timex_labels]
                        scores_to_prune += timex_scores
                        pruned_labels_in_note = modify_tlinks([pair for note_id, pair in pairs_to_prune],
                                                              NetworkMem.convert_int_labels_to_str(labels_to_prune),
                                                              scores_to_prune)
                        pruned_labels_in_note = NetworkMem.convert_str_labels_to_int(pruned_labels_in_note)
                        true_labels_after_prune += true_timex_labels

                        predictions += pruned_labels_in_note
                        scores += scores_to_prune
                        true_labels += true_labels_after_prune
                    else:
                        predictions += labels_in_note
                        scores.append(probs_in_note)
                        true_labels.append(true_labels_in_note)

                    probs_in_note = None
                    true_labels_in_note = None
                    labels_in_note = []

            if not isinstance(feeder, list):
                self.test_data_collection.append(note_data)  # store data for fast retrieval

        if not combine_timex:
            if len(scores) > 1:
                scores = np.concatenate(scores, axis=0)
                true_labels = np.concatenate(true_labels, axis=0)
            elif len(scores) == 1:
                scores = scores[0]
                true_labels = true_labels[0]
        if evaluation:
            Network.class_confusion(predictions, true_labels, len(LABELS))
            return precision_recall_fscore_support(true_labels, predictions, average='micro')

        return predictions, scores, true_labels #, pair_indexes

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

    def get_sentence_embeddings(self, sentences):
        if self.infersent is None:
            assert self.word_vectors is not None
            encoder_path = '/home/ymeng/projects/InferSent/encoder'
            sys.path.append(encoder_path)
            self.infersent = torch.load(os.path.join(encoder_path, 'infersent.allnli.pickle'), map_location=lambda storage, loc: storage)
            self.infersent.set_glove_path(os.environ["TEA_PATH"] + 'embeddings/glove.840B.300d.txt')
            self.infersent.build_vocab([''], tokenize=True) # must do this to get the infersent.word_vec object
            self.infersent.word_vec = self.word_vectors

        embeddings = self.infersent.encode(sentences, tokenize=True)
        return embeddings

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