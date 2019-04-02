from __future__ import print_function
import time
import copy
import numpy as np
# np.random.seed(1337)
import math
import sys
import pickle
import torch

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support
from src.learning.network_mem import NetworkMem, get_timex_predictions
from word2vec import load_word2vec_binary, build_vocab
from src.learning.models_pytorch import PairRelation, EMBEDDING_DIM, DENSE_LABELS, MAX_LEN, LABELS
from src.learning.time_ref import predict_timex_rel, TimeRefNetwork
from src.learning.break_cycle import modify_tlinks
from collections import deque
from sklearn.metrics import classification_report


TYPE_MARKERS = {'_INTRA_': 0, '_CROSS_': 1, '_DCT_': -1}
BATCH_SIZE = 10  # group batches

class NetworkMemPT(NetworkMem):
    def __init__(self, no_ntm=False, nb_training_files=None, model_path=None, device=None):
        super(NetworkMemPT, self).__init__()
        self.roll_counter = 0
        self.nb_training_files = nb_training_files
        self.nb_test_files = None
        self.test_data_collection = []
        self.no_ntm = no_ntm
        self.model_path = model_path
        self.infersent = None
        self.training_passes = 1
        self.test_passes = 1
        self.positive_only = False
        self.word_indexes = {}
        self.device = device


    def load_raw_model(self, no_ntm, fit_batch_size=10, batch_size=80):
        # n_slots = max(1, batch_size / 2)
        n_slots = batch_size

        embedding_matrix = self.get_embedding_matrix()
        vocab_size = len(embedding_matrix)
        if no_ntm:
            model = PairRelation(vocab_size, nb_classes=6, input_dropout=0.5, word_embeddings=self.get_embedding_matrix()).to(self.device)
        # else:
        #     model = get_ntm_hiddenfeed(batch_size=fit_batch_size, input_dropout=0.1, group_size=None, m_depth=512, n_slots=n_slots,
        #                              ntm_output_dim=1024, shift_range=3, max_len=MAX_LEN, read_heads=1, write_heads=1,
        #                              nb_classes=len(LABELS), embedding_matrix=self.get_embedding_matrix(), model_path=self.model_path)

        return model

    def train_model(self, model=None, no_ntm=False, epochs=100, steps_per_epoch=10, input_generator=None,
                    val_generator=None, weight_classes=False, callbacks={}, batch_size=300, has_auxiliary=False):

        # from keras.optimizers import RMSprop
        # learning rate schedule
        def step_decay(epoch):
            drop = 0.5
            if no_ntm:
                initial_lrate = 0.001
                epochs_drop = 20
            else:
                initial_lrate = 0.002
                epochs_drop = 10  # get half in every epochs_drop steps
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        fit_batch_size = BATCH_SIZE

        if model is None:
            model = self.load_raw_model(no_ntm, fit_batch_size=fit_batch_size, batch_size=batch_size)
        print("loaded raw model")

        if DENSE_LABELS:
            eval_batch = 40 #0
        else:
            eval_batch = 160

        print('Training network...')
        training_history = []
        best_result = None
        epochs_over_best = 0
        # batch_sizes = [int(batch_size/2), int(batch_size*3/4), batch_size, int(batch_size*3/2), int(batch_size*2)]
        batch_size *= 2
        for epoch in range(epochs):
            lr = step_decay(epoch)
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = lr

            epoch_history = []
            start = time.time()
            for n in range(self.nb_training_files):
                note_data = next(input_generator)
                # batch_size = np.random.choice(batch_sizes)
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
                nb_fit_batches = int(y.shape[0]/fit_batch_size) + (extra != 0)
                to_add = fit_batch_size - extra
                indexes_to_add = np.random.choice(y.shape[0], to_add)  # randomly sample more instances
                for index in indexes_to_add:
                    for i in range(len(X)):
                        X[i] = np.concatenate([X[i], X[i][index][None,:]], axis=0)
                        X[i] = torch.from_numpy(X[i]).long().to(self.device)
                    y = np.concatenate([y, y[index][None, :]])
                y = np.argmax(y, axis=-1)  # change one-hot to indexes
                y = torch.from_numpy(y).long().to(self.device)

                for m in range(nb_fit_batches):
                    X_b = [item[m*fit_batch_size: (m+1)*fit_batch_size] for item in X]
                    y_b = y[m*fit_batch_size: (m+1)*fit_batch_size]

                    # print("size:", y_b.shape)
                    history = model.fit(X_b, y_b, epochs=1)  # history = (loss, acc)

                    # print(history.history)
                    epoch_history.append(history)

                loss = np.mean([h[0] for h in epoch_history])
                acc = np.mean([h[1] for h in epoch_history])

                sys.stdout.write("epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f\r" % (
                epoch + 1, n + 1, self.nb_training_files, int(time.time() - start), loss, acc))
                sys.stdout.flush()

            sys.stdout.write("\n")
            training_history.append({'categorical_accuracy': acc, 'loss': loss})

            if no_ntm and epoch <= 20:
                continue
            elif not no_ntm and epoch < 15:
                continue
            print("\n\nepoch finished... evaluating on val data...")
            if val_generator is not None:
                # evaluate after each epoch
                evalu = self.predict(model, val_generator, batch_size=eval_batch, fit_batch_size=fit_batch_size, evaluation=True, smart=False, has_auxiliary=has_auxiliary)

        print("Fisnished training. Data rolled %d times" %self.roll_counter)
        return model, training_history

    def predict(self, model, data_generator, batch_size=0, fit_batch_size=5, evaluation=True, smart=True, has_auxiliary=False, pruning=False):
        predictions = []
        scores = []
        pair_indexes = {}
        true_labels = []
        true_labels_in_note = None
        probs_in_note = None
        labels_in_note = []
        end_of_note = False
        model.eval()

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

            # model.reset_states()
            for sliced in self.slice_data(note_data, batch_size=batch_size, shift=0):
                X, y, pair_index, marker = sliced
                # X = [np.repeat(item, fit_batch_size, axis=0) for item in X]  # have to do this because the batch size must be fixed
                X = [torch.from_numpy(item).long().to(self.device) for item in X]
                y = torch.from_numpy(y).long().to(self.device)

                note_index = list(pair_index.keys())[0][0]
                if end_of_note and note_index == 0: # all data consumed
                    all_notes_consumed = True
                    break
                # pair_indexes.update(pair_index)

                # probs = model.predict(X, batch_size=1)
                probs = model.predict(X)
                y = y[0, :]  # remove the dummy dimension
                # probs = probs[0, :]

                if self.test_passes > 1:
                    data_len = int(len(y)/self.test_passes)
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

                    if self.positive_only:
                        pos_indexes = [ind for ind, label in enumerate(true_labels_in_note) if label != LABELS.index('None') ]
                        true_labels_in_note = [true_labels_in_note[ind] for ind in pos_indexes]
                        labels_in_note = [labels_in_note[ind] for ind in pos_indexes]

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

                    offset = len(pair_indexes)
                    new_pair_index = {}
                    for k in pair_index:
                        new_pair_index[k] = pair_index[k] + offset
                    pair_indexes.update(new_pair_index)
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
            class_confusion(predictions, true_labels, len(LABELS))
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

        indexes = sorted([v for k, v in new_pair_index.items()])
        new_labels = [labels[i] for i in indexes]
        old_indexes_to_new = dict(zip(indexes, range(len(indexes))))
        for key in new_pair_index:
            new_pair_index[key] = old_indexes_to_new[new_pair_index[key]] # map old indexes to new indexes in the label list

        return new_labels, np.array(indexes), new_pair_index

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


def class_confusion(predicted, actual, nb_classes):
    '''
    print confusion matrix for two lists of labels.
    A given index in both lists should correspond to the same data sample
    '''

    confusion = np.zeros((nb_classes, nb_classes), dtype='int16')

    for true, pred in zip(actual, predicted):
        # build confusion matrix
        confusion[true, pred] += 1

    pickle.dump(confusion, open('evaluation.pkl', 'wb'))
    # print confusion matrix
    print("confusion matrix")
    print("rows: actual labels.  columns: predicted labels.")
    for i, row in enumerate(confusion):
        print(i, ": ", row)

    print(classification_report(actual, predicted, digits=3))