import os
import pickle
import copy
import sys

import numpy as np
np.random.seed(1337)
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Merge, MaxPooling1D, TimeDistributed, Flatten, Masking, Input, Dropout, Permute
from keras.regularizers import l2

from word2vec import load_word2vec_binary
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

# LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IBEFORE", "IAFTER", "IS_INCLUDED", "INCLUDES",
#           "DURING", "BEGINS", "BEGUN_BY", "ENDS", "ENDED_BY", "None"]
LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IS_INCLUDED", "INCLUDES", "None"] # TimeBank Dense labels
EMBEDDING_DIM = 300
DENSE_LABELS = True

class Network(object):
    def __init__(self):
        #self.id_to_path = {}
        self.label_to_int = {}
        self.int_to_label = {}
        self.label_reverse_map = {} # map BEFORE to AFTER etc., in int label
        self.word_vectors = None

    def get_untrained_model(self, encoder_dropout=0, decoder_dropout=0, input_dropout=0, reg_W=0, reg_B=0, reg_act=0, LSTM_size=32, dense_size=100, maxpooling=True, data_dim=EMBEDDING_DIM, max_len=22, nb_classes=7):
        '''
        Creates a neural network with the specified conditions.
        params:
            encoder_dropout: dropout rate for LSTM encoders (NOT dropout for LSTM internal gates)
            decoder_dropout: dropout rate for decoder
            reg_W: lambda value for weight regularization
            reg_b: lambda value for bias regularization
            reg_act: lambda value for activation regularization
            LSTM_size: number of units in the LSTM layers
            maxpooling: pool over LSTM output at each timestep, or just take the output from the final LSTM timestep
            data_dim: dimension of the input data
            max_len: maximum length of an input sequence (this should be found based on the training data)
            nb_classes: number of classes present in the training data
        '''

        # create regularization objects if needed
        if reg_W != 0:
            W_reg = l2(reg_W)
        else:
            W_reg = None

        if reg_B != 0:
            B_reg = l2(reg_B)
        else:
            B_reg = None

        if reg_act != 0:
            act_reg = l2(reg_act)
        else:
            act_reg = None

        # encode the first entity
        encoder_L = Sequential()

        encoder_L.add(Dropout(input_dropout, input_shape=(max_len, data_dim)))

        # with maxpooling
        if maxpooling:
            encoder_L.add(LSTM(LSTM_size, return_sequences=True, inner_activation="sigmoid"))
            if encoder_dropout != 0:
                encoder_L.add(TimeDistributed(Dropout(encoder_dropout)))
            encoder_L.add(MaxPooling1D(pool_length=max_len))
            encoder_L.add(Flatten())

        # without maxpooling
        else:
            encoder_L.add(Masking(mask_value=0.))
            encoder_L.add(LSTM(LSTM_size, return_sequences=False, inner_activation="sigmoid"))
            if encoder_dropout != 0:
                encoder_L.add(Dropout(encoder_dropout))

        # encode the second entity
        encoder_R = Sequential()

        encoder_R.add(Dropout(input_dropout, input_shape=(max_len, data_dim)))

        # with maxpooling
        if maxpooling:
            encoder_R.add(LSTM(LSTM_size, return_sequences=True, inner_activation="sigmoid"))
            if encoder_dropout != 0:
                encoder_R.add(TimeDistributed(Dropout(encoder_dropout)))
            encoder_R.add(MaxPooling1D(pool_length=max_len))
            encoder_R.add(Flatten())

        else:
        # without maxpooling
            encoder_R.add(Masking(mask_value=0.))
            encoder_R.add(LSTM(LSTM_size, return_sequences=False, inner_activation="sigmoid"))
            if encoder_dropout != 0:
                encoder_R.add(Dropout(encoder_dropout))

        # combine and classify entities as a single relation
        decoder = Sequential()
        decoder.add(Merge([encoder_R, encoder_L], mode='concat'))
        decoder.add(Dense(dense_size, W_regularizer=W_reg, b_regularizer=B_reg, activity_regularizer=act_reg, activation='relu'))
        if decoder_dropout != 0:
            decoder.add(Dropout(decoder_dropout))
        decoder.add(Dense(nb_classes, W_regularizer=W_reg, b_regularizer=B_reg, activity_regularizer=act_reg, activation='softmax'))

        # compile the final model
        decoder.summary()
        decoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return decoder

    def train_model(self, notes, model=None, epochs=5, training_input=None, val_input=None, no_val=False, weight_classes=False, batch_size=256,
        encoder_dropout=0, decoder_dropout=0, input_dropout=0, reg_W=0, reg_B=0, reg_act=0, LSTM_size=32, dense_size=100,
        maxpooling=True, data_dim=200, max_len='auto', nb_classes=13, callbacks=[]):
        '''
        obtains entity pairs and tlink labels from every note passed, and uses them to train the network.
        params:
            notes: timeML files to train on
            epochs: number of training epochs to perform
            training_input: provide processed training matrices directly, rather than rebuilding the matrix from notes every time. Useful for training multipul models with the same data.
                Formatted as a tuple; (XL, XR, labels)
            weight_classes: rather or not to use class weighting
            batch_size: size of training batches to use
            max_len: either an integer specifying the maximum input sequence length, or 'auto', which infer maximum length from the training data
        All other parameters feed directly into get_untrained_model(), and are described there.
        '''
        if training_input == None:
            XL, XR, labels = self._get_training_input(notes)
        else:
            XL, XR, labels = training_input

        # # remove zero labels (there are too many of them)
        # # TODO: maybe just remove some of them, not all
        # print XL.shape
        # labels = np.array(labels)
        # pos_indexes = np.where(labels>0)[0]
        # print len(labels)
        # print max(pos_indexes)
        # XL = XL[pos_indexes,:,:]
        # XR = XR[pos_indexes,:,:]
        # labels = labels[pos_indexes]

        # reformat labels so that they can be used by the NN
        Y = to_categorical(labels,nb_classes)

        # use weighting to assist with the imbalanced data set problem
        if weight_classes:
            class_weights = self.get_uniform_class_weights(Y)
        else:
            class_weights = None

        # infer maximum sequence length
        if max_len == 'auto':
            max_len = XL.shape[-2]
        # pad input to reach max_len
        else:
            filler = np.zeros((1, max_len, 1))
            XL, _ = Network._pad_to_match_dimensions(XL, filler, 1, pad_left=True)
            XR, _ = Network._pad_to_match_dimensions(XR, filler, 1, pad_left=True)

        if model is None:
            model = self.get_untrained_model(encoder_dropout=encoder_dropout, decoder_dropout=decoder_dropout, input_dropout=input_dropout, reg_W=reg_W, reg_B=reg_B, reg_act=reg_act, LSTM_size=LSTM_size, dense_size=dense_size,
            maxpooling=maxpooling, data_dim=data_dim, max_len=max_len, nb_classes=nb_classes)

        # train the network
        print 'Training network...'
        if no_val:
            validation_split = 0.0
            validation_data = None
        elif val_input is None:
            # split off validation data with 20 80 split (this way we get the same validation data every time we use this data sample, and can test on it after to get a confusion matrix)
            V_XL = XL[:(XL.shape[0]/5),:,:]
            V_XR = XR[:(XR.shape[0]/5),:,:]
            V_Y  = Y [:( Y.shape[0]/5),:]
            V_labels = labels[:(Y.shape[0]/5)]

            XL = XL[(XL.shape[0]/5):,:,:]
            XR = XR[(XR.shape[0]/5):,:,:]
            Y  = Y [( Y.shape[0]/5):,:]

            validation_split = 0.2
            validation_data = ([V_XL, V_XR], V_Y)
        else:
            validation_split = 0 # will be overwritten by val data
            V_XL, V_XR, V_labels, V_pair_index = val_input
            V_Y = to_categorical(V_labels, nb_classes)
            filler = np.zeros((1, max_len, 1))
            V_XL, _ = Network._pad_to_match_dimensions(V_XL, filler, 1, pad_left=True)
            V_XR, _ = Network._pad_to_match_dimensions(V_XR, filler, 1, pad_left=True)
            validation_data = ([V_XL, V_XR], V_Y)

        training_history = model.fit([XL, XR], Y, nb_epoch=epochs, validation_split=validation_split, class_weight=class_weights,
                                         batch_size=batch_size, validation_data=validation_data, callbacks=callbacks)

        test = model.predict_classes([V_XL, V_XR])
        Network.class_confusion(test, V_labels, nb_classes)

        if val_input is not None:
            try:
                print "Trying smart predict..."
                probs = model.predict_proba([V_XL, V_XR])
                smart_test, pair_index = self.smart_predict(test, probs, V_pair_index, type='int')
                Network.class_confusion(smart_test, V_labels, nb_classes)
            except KeyError:
                print "cannot perform smart predicting"

        return model, training_history.history

    def single_predict(self, notes, model, pair_type, test_input=None, predict_prob=False):
        '''
        predict using a trained single pass model
        '''

        if test_input is not None:
            XL, XR, _labels, pair_index = test_input
        else:
            XL, XR, _labels, pair_index = self._get_test_input(notes, pair_type)

        if XL is None:
            return [], [], {}

        # get expected length of model input
        model_input_len = model.input_shape[0][-2]

        if model_input_len > XL.shape[-2]:
            # pad input matrix to fit expected length
            filler = np.zeros((1, model_input_len, 1))
            XL, _ = Network._pad_to_match_dimensions(XL, filler, 1, pad_left=True)
            XR, _ = Network._pad_to_match_dimensions(XR, filler, 1, pad_left=True)
        else:
            XL = Network._strip_to_length(XL, model_input_len, 1)
            XR = Network._strip_to_length(XR, model_input_len, 1)

        print 'Predicting...'
        labels = model.predict_classes([XL, XR])
        if predict_prob:
            probs = model.predict_proba([XL, XR])
        else:
            probs = None

        # if evalu:
        #     Network.class_confusion(labels, gold_labels, 13)

        #return self._convert_int_labels_to_str(labels), probs, pair_index

        # format of pair_index: {(note_index, (e1, e2)) : index}
        return labels, probs, pair_index # int labels

    def smart_predict(self, labels, probs, pair_index, type='int'):
        # int labels
        #labels, probs, pair_index = self.single_predict(notes, model, predict_prob=True)

        proccessed = {}
        label_scores = [0.0 for i in labels]
        for key, index in pair_index.iteritems():
            if key in proccessed:
                continue
            note_index, pair = key
            try:
                label = labels[index]
            except IndexError:
                print "IndexError: labels length %d, index %d" % (len(labels), index)
                assert IndexError

            opposite_key = (note_index, (pair[1], pair[0]))
            if opposite_key not in pair_index: # mainly for dct pairs
                continue
            opposite_index = pair_index[opposite_key]
            opposite_label = labels[opposite_index] # predicted label of the opposite pair

            # # set "no link" with lower priority, because "no link" tends to dominate
            # if label == LABELS.index('None'):
            #     score = 1.0/len(LABELS) # set it to random level
            # else:
            #     score = probs[index, label]
            # if opposite_label == LABELS.index('None'):
            #     opposite_score = 1.0/len(LABELS) # set it to random level
            # else:
            #     opposite_score = probs[opposite_index, opposite_label]

            score = probs[index, label]
            opposite_score = probs[opposite_index, opposite_label]

            if score > opposite_score:
                labels[opposite_index] = self.reverse_labels([label])[0]
                label_scores[index] = score
                label_scores[opposite_index] = score
            else:
                labels[index] = self.reverse_labels([opposite_label])[0]
                label_scores[index] = opposite_score
                label_scores[opposite_index] = opposite_score

            proccessed[key] = 1
            proccessed[opposite_key] = 1
        if type == 'int':
            return labels, pair_index
        return self._convert_int_labels_to_str(labels), pair_index, label_scores

    def _get_training_input(self, notes, pair_type, nolink_ratio=None, presence=False, shuffle=True):
        """if shuffle=False, training data will be in narrative order"""

        # data tensor for left and right SDP subpaths
        XL = None
        XR = None

        if self.word_vectors is None:
            print 'Loading word embeddings...'
            self.word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin', verbose=0)
            # word_vectors = load_word2vec_binary(os.environ["TEA_PATH"]+'/wiki.dim-300.win-8.neg-15.skip.bin', verbose=0)
            #word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/glove.840B.300d.txt', verbose=0)
            # word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/glove.6B.200d.txt', verbose=0)

        # print 'Extracting dependency paths...'
        labels = []
        for i, note in enumerate(notes):

            # get the representation for the event/timex pairs in the note
            # will be 3D tensor with axis zero holding the each pair, axis 1 holding the word embeddings
            # (with length equal to word embedding length), and axis 2 hold each word.
            # del_list is a list of indices for which no SDP could be obtained
            left_vecs, right_vecs, id_pairs, type_markers = self._extract_path_representations(note, self.word_vectors, pair_type)

            # perform a random check, to make sure the data is correctly augmented
            if not id_pairs:
                print "No pair found:", note.annotated_note_path
                continue

            pos_case_indexes = []
            neg_case_indexes = []
            note_labels = []

            if DENSE_LABELS:
                id_to_labels = note.get_id_to_denselabels() # use TimeBank-Dense labels
            else:
                id_to_labels = note.id_to_labels

            if not id_to_labels:
                continue

            for index, pair in enumerate(id_pairs):
                if pair in id_to_labels and id_to_labels[pair] != 'None':
                    pos_case_indexes.append(index)
                else:
                    neg_case_indexes.append(index)
                note_labels.append(id_to_labels.get(pair, 'None'))
            note_labels = np.array(note_labels)

            if nolink_ratio is not None:
                np.random.shuffle(neg_case_indexes)
                n_samples = min(len(neg_case_indexes), int(nolink_ratio * len(pos_case_indexes)) )
                neg_case_indexes = neg_case_indexes[0:n_samples]
                if not neg_case_indexes:
                    training_indexes = np.array(pos_case_indexes, dtype=np.int32)
                else:
                    training_indexes = np.concatenate([pos_case_indexes, neg_case_indexes])
                left_vecs = left_vecs[training_indexes, :, :]
                right_vecs = right_vecs[training_indexes, :, :]
                note_labels = note_labels[training_indexes]

            if labels == []:
                labels = note_labels
            else:
                labels = np.concatenate((labels, note_labels))

            # add the note's data to the combine data matrix
            if XL is None:
                XL = left_vecs
            else:
                XL = Network._pad_and_concatenate(XL, left_vecs, axis=0, pad_left=[1])

            if XR is None:
                XR = right_vecs
            else:
                XR = Network._pad_and_concatenate(XR, right_vecs, axis=0, pad_left=[1])

            # if XL is None:
            #     XL = left_vecs
            #     XL = np.expand_dims(XL, axis=0) # (1, sentences, words, word_dim)
            # else:
            #     XL = Network._pad_and_concatenate(XL, left_vecs, axis=0, pad_left=[2])
            #
            # if XR is None:
            #     XR = right_vecs
            #     XR = np.expand_dims(XR, axis=0)  # (1, sentences, words, word_dim)
            # else:
            #     XR = Network._pad_and_concatenate(XR, right_vecs, axis=0, pad_left=[2])

        # pad XL and XR so that they have the same number of dimensions on the second axis
        # any other dimension mis-matches are caused by actually errors and should not be padded away
        XL, XR = Network._pad_to_match_dimensions(XL, XR, 1, pad_left=True)

        # # extract longest input sequence in the training data, and ensure both matrices
        # input_len = XL.shape[2]

        if presence:
            for i, label in enumerate(labels):
                if label != 0:
                    labels[i] = 1

        if shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(XL)
            np.random.set_state(rng_state)
            np.random.shuffle(XR)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
        del notes
        labels = self._convert_str_labels_to_int(labels)

        return XL, XR, labels

    def _get_test_input(self, notes, pair_type):
        # data tensor for left and right SDP subpaths
        XL = None
        XR = None

        if self.word_vectors is None:
            print 'Loading word embeddings...'
            self.word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin', verbose=0)
            # word_vectors = load_word2vec_binary(os.environ["TEA_PATH"]+'/wiki.dim-300.win-8.neg-15.skip.bin', verbose=0)

        print 'Extracting dependency paths...'
        labels = None
        pair_index = {} # record note id and all the used entity pairs
        index_offset = 0
        for i, note in enumerate(notes):

            # get the representation for the event/timex pairs in the note
            # will be 3D tensor with axis zero holding the each pair, axis 1 holding the word embeddings
            # (with length equal to word embedding length), and axis 2 hold each word.
            # del_list is a list of indices for which no SDP could be obtained
            left_vecs, right_vecs, id_pairs, type_markers = self._extract_path_representations(note, self.word_vectors, pair_type)

            if DENSE_LABELS:
                id_to_labels = note.id_to_denselabels # use TimeBank-Dense labels
            else:
                id_to_labels = note.id_to_labels

            if id_to_labels:
                note_labels = []
                index_to_reverse = []
                for index, pair in enumerate(id_pairs): # id pairs that have tlinks
                    #pair_index[(i, pair)] = index + index_offset

                    label_from_file = id_to_labels.get(pair, 'None')
                    # the id pairs from notes are in both ways, however data file may not labeled in both ways
                    # that is why we need to find the opposite
                    # QA evaluation does not need this. Timebank dense does not need it either.
                    # we may want to augment dev/test files so this may not be needed any more
                    opposite_from_file = id_to_labels.get((pair[1], pair[0]), 'None')
                    if label_from_file == 'None' and opposite_from_file != 'None': # maybe the reversed pair is labeled
                        index_to_reverse.append(index)
                        note_labels.append(opposite_from_file) # save the opposite lable first, reverse later
                    else:
                        note_labels.append(label_from_file)

                note_labels = self._convert_str_labels_to_int(note_labels)
                labels_to_reverse = [note_labels[x] for x in index_to_reverse]
                reversed = self.reverse_labels(labels_to_reverse)
                print note.annotated_note_path
                print "{} labels augmented".format(len(reversed))

                note_labels = np.array(note_labels, dtype='int16')
                index_to_reverse = np.array(index_to_reverse)
                if index_to_reverse.any():
                    note_labels[index_to_reverse] = reversed

                if labels is None:
                    labels = note_labels
                else:
                    labels =np.concatenate((labels, note_labels))

            for index, pair in enumerate(id_pairs):
                pair_index[(i, pair)] = index + index_offset

            index_offset += len(id_pairs)

            # add the note's data to the combine data matrix
            if XL is None:
                XL = left_vecs
            else:
                XL = Network._pad_and_concatenate(XL, left_vecs, axis=0, pad_left=[1])

            if XR is None:
                XR = right_vecs
            else:
                XR = Network._pad_and_concatenate(XR, right_vecs, axis=0, pad_left=[1])

        # pad XL and XR so that they have the same number of dimensions on the second axis
        # any other dimension mis-matches are caused by actually errors and should not be padded away
        if XL is not None:
            XL, XR = Network._pad_to_match_dimensions(XL, XR, 1, pad_left=True)

        return XL, XR, labels, pair_index

    def _extract_path_words(self, note, pair_type):
        id_pair_to_path_words = {}

        assert pair_type in ('intra', 'cross', 'dct', 'all')

        if pair_type in ('intra', 'all'):
            id_pair_to_path = note.get_intra_sentence_subpaths() #key: (src_id, target_id), value: [left_path, right_path]
            marker = '_INTRA_'

            # extract paths of all intra_sentence pairs
            # negative data (no relation) also included
            for id_pair in id_pair_to_path:
                # if id_pair[0][0] == 't' and id_pair[1][0] == 't':
                #     # filter (t, t) pairs, we have a separate model
                #     # this will be redundant after all notes are updated
                #     continue
                left_path, right_path = id_pair_to_path[id_pair]
                left_words = [note.id_to_tok['w'+x[1:]]['token'] for x in left_path]
                right_words = [note.id_to_tok['w'+x[1:]]['token'] for x in right_path]

                # right_words.reverse()  # make right branch top to bottom

                if id_pair[0][0] == 'e':
                    left_words.insert(0, '_EVENT_')
                else:
                    left_words.insert(0, '_TIMEX_')
                if id_pair[1][0] == 'e':
                    right_words.insert(0, '_EVENT_')
                else:
                    left_words.insert(0, '_TIMEX_')

                id_pair_to_path_words[id_pair] = (left_words, right_words, marker)

        if pair_type in ('cross', 'all'):
            id_pair_to_path = note.get_cross_sentence_subpaths()
            marker = '_CROSS_'

            # extract paths of all intra_sentence pairs
            # negative data (no relation) also included

            for id_pair in id_pair_to_path:
                # if id_pair[0][0] == 't' and id_pair[1][0] == 't':
                #     # filter (t, t) pairs, we have a separate model
                #     # this will be redundant after all notes are updated
                #     continue
                left_path, right_path = id_pair_to_path[id_pair]
                left_words = [note.id_to_tok['w'+x[1:]]['token'] for x in left_path]
                right_words = [note.id_to_tok['w'+x[1:]]['token'] for x in right_path]
                # right_words.reverse()  # make right branch top to bottom

                if id_pair[0][0] == 'e':
                    left_words.insert(0, '_EVENT_')
                else:
                    left_words.insert(0, '_TIMEX_')
                if id_pair[1][0] == 'e':
                    right_words.insert(0, '_EVENT_')
                else:
                    left_words.insert(0, '_TIMEX_')

                id_pair_to_path_words[id_pair] = (left_words, right_words, marker)

        if pair_type in ('dct', 'all'): # (event, t0) pairs
            t0_to_path = note.get_t0_subpaths()
            marker = '_DCT_'

            for entity_id in t0_to_path:
                left_path = t0_to_path[entity_id]
                left_words = [note.id_to_tok['w' + x[1:]]['token'] for x in left_path]

                if id_pair[0][0] == 'e':
                    left_words.insert(0, '_EVENT_')
                else:
                    left_words.insert(0, '_TIMEX_')

                right_words = copy.copy(left_words)
                # right_words.reverse() # reverse order of the path, to make two branches
                id_pair_to_path_words[(entity_id, 't0')] = (left_words, right_words, marker)

        return id_pair_to_path_words

    def _extract_context_words(self, note, pair_type):

        def _insert_tag(context, eid):
            word_id = note.id_to_wordIDs[eid][0]
            word = note.id_to_tok[word_id]['token']
            try:
                loc = context.index(word)
                if eid[0] == 'e':
                    context.insert(loc, '_EVENT_')
                else:
                    context.insert(loc, '_TIMEX_')
            except:
                print "cound not find token %s in %s" % (eid, note.annotated_note_path)
                print context
                sys.exit(1)
            return context

        assert pair_type in ('intra', 'cross', 'dct', 'all')
        id_pair_to_context_words = {}

        if pair_type in ('intra', 'all'):
            marker = '_INTRA_'
            intra = note.get_intra_sentence_context()
            for key in intra:
                left_context = intra[key][0]
                left_context = _insert_tag(left_context, key[0])
                right_context = intra[key][1]
                right_context = _insert_tag(right_context, key[1])
                intra[key] = (left_context, right_context, marker)

            id_pair_to_context_words.update(intra)  # key: (src_id, target_id), value: (left_context, right_context, marker)

        if pair_type in ('cross', 'all'):
            marker = '_CROSS_'
            cross = note.get_cross_sentence_context()
            for key in cross:
                left_context = cross[key][0]
                left_context = _insert_tag(left_context, key[0])
                right_context = cross[key][1]
                right_context = _insert_tag(right_context, key[1])
                cross[key] = (left_context, right_context, marker)
            id_pair_to_context_words.update(cross)

        if pair_type in ('dct', 'all'):  # (event, t0) pairs
            marker = '_DCT_'
            entity_to_context_words = note.get_t0_context()
            for entity_id in entity_to_context_words:
                words = entity_to_context_words[entity_id]
                words = _insert_tag(words, entity_id)
                # reversed = copy.copy(words)
                id_pair_to_context_words[(entity_id, 't0')] = [words, ['_TIMEX_', 'now'], marker]

        return id_pair_to_context_words

    @staticmethod
    def sort_id_pairs(note, id_pairs):
        """Sort event and timex ids in narrative order"""
        return sorted(id_pairs, key=lambda x: note.id_to_sent[x[0]]) # sorted by sentence number of the first id

    def _extract_path_representations(self, note, word_vectors, pair_type, use_shortest=True):

        word_vectors['_EVENT_'] = np.ones(EMBEDDING_DIM)
        word_vectors['_TIMEX_'] = - np.ones(EMBEDDING_DIM)

        if use_shortest:
            id_pair_to_path_words = self._extract_path_words(note, pair_type)
        else:
            id_pair_to_path_words = self._extract_context_words(note, pair_type)

        # del list stores the indices of pairs which do not have a SDP so that they can be removed from the labels later
        #del_list = []

        left_vecs = None
        right_vecs = None

        # get the word vectors for every word in the left pathy
        # must sort it to match the labels correctly
        sorted_pairs = Network.sort_id_pairs(note, id_pair_to_path_words.keys())
        type_markers = []
        for id_pair in sorted_pairs:
            type_marker = id_pair_to_path_words[id_pair][-1]
            assert type_marker in ('_INTRA_', '_CROSS_', '_DCT_')
            type_markers.append(type_marker)

            path = id_pair_to_path_words[id_pair][0]
            left_vecs_path = None
            for word in path:
                # try to get embedding for a given word. If the word is not in the vocabulary, use a vector of all 1s.
                try:
                    embedding = word_vectors[word]
                except KeyError:
                    embedding = np.random.uniform(low=-0.5, high=0.5, size=(EMBEDDING_DIM))

                # reshape to 3 dimensions so embeddings can be concatenated together to form the final input values
                embedding = embedding[np.newaxis, np.newaxis, :]
                if left_vecs_path is None:
                    left_vecs_path = embedding
                else:
                    left_vecs_path = np.concatenate((left_vecs_path, embedding), axis=1)

            if left_vecs_path is None:
                embedding = np.random.uniform(low=-0.5, high=0.5, size=(EMBEDDING_DIM))
                left_vecs_path = embedding[np.newaxis, np.newaxis, :]
            # if there were no vectors, the link involves the document creation time or is a cross sentence relation.
            # add index to list to indexes to remove and continue
            # if left_vecs_path is None:
            #     del_list.append(id_pair)
            #     continue
            if left_vecs is None:
                left_vecs = left_vecs_path
            else:
                left_vecs = Network._pad_and_concatenate(left_vecs, left_vecs_path, axis=0, pad_left=[1])

        # get the vectors for every word in the right path
            path = id_pair_to_path_words[id_pair][1]
            right_vecs_path = None
            for word in path:
                # try to get embedding for a given word. If the word is not in the vocabulary, use a vector of all 0s.
                try:
                    embedding = word_vectors[word]
                except KeyError:
                    embedding = np.random.uniform(low=-0.5, high=0.5, size=(EMBEDDING_DIM))

                # reshape to 3 dimensions so embeddings can be concatenated together to form the final input values
                embedding = embedding[np.newaxis, np.newaxis, :]
                if right_vecs_path is None:
                    right_vecs_path = embedding
                else:
                    right_vecs_path = np.concatenate((right_vecs_path, embedding), axis=1)

            if right_vecs_path is None:
                embedding = np.random.uniform(low=-0.5, high=0.5, size=(EMBEDDING_DIM))
                right_vecs_path = embedding[np.newaxis, np.newaxis, :]
            # if there were no vectors, the link involves the document creation time or is a cross sentence relation.
            # # remove label from list and continue to the next path
            # if right_vecs_path is None:
            #     del_list.append(id_pair)
            #     continue
            if right_vecs is None:
                right_vecs = right_vecs_path
            else:
                right_vecs = Network._pad_and_concatenate(right_vecs, right_vecs_path, axis=0, pad_left=[1])


        # print "removed from list: ", len(del_list)

        return left_vecs, right_vecs, sorted_pairs, type_markers

    @staticmethod
    def _pad_and_concatenate(a, b, axis, pad_left=[]):
        '''
        given two tensors, pad with zeros so that all dimensions are equal except the axis to concatentate on, and the concatenate
        pad_left is a list of dimensions to pad the left side on. Other dimensions will recieve right sided padding
        '''
        if b is None:
            return a
        # for each axis that is not to be concatenated on, pad the smaller of the two tensors with zeros so that both have equal sizes
        for i in range(len(a.shape)):
            if i == axis:
                continue
            if i in pad_left:
                a, b = Network._pad_to_match_dimensions(a, b, i, pad_left=True)
            else:
                a, b = Network._pad_to_match_dimensions(a, b, i)

        # concatenate and return the padded matrices
        return np.concatenate((a, b), axis=axis)

    @staticmethod
    def _pad_to_match_dimensions(a, b, axis, pad_left=False):
        '''
        given to tensors and an axis for comparison, pad the smaller of the two with zeros to match the dimensions of the larger
        '''

        # get function to pad correct side
        if pad_left:
            concat = lambda X, _pad_shape, _axis: np.concatenate((np.zeros(tuple(_pad_shape)), X), axis=_axis)
        else:
            concat = lambda X, _pad_shape, _axis: np.concatenate((X, np.zeros(tuple(_pad_shape))), axis=_axis)

        a_axis = a.shape[axis]
        b_axis = b.shape[axis]
        if a_axis > b_axis:
            pad_shape = list(b.shape)
            pad_shape[axis] = a_axis - b_axis
            b = concat(b, pad_shape, axis)
        if b_axis > a_axis:
            pad_shape = list(a.shape)
            pad_shape[axis] = b_axis - a_axis
            a = concat(a, pad_shape, axis)

        return a, b

    @staticmethod
    def _strip_to_length(a, length, axis):

        if a.shape[axis] > length:
            snip = a.shape[axis] - length
            if axis == 1:
                a = a[:, snip:, :]

        return a

    def _get_token_id_subpaths(self, note):
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

            left_path, right_path = note.dependency_paths.get_left_right_subpaths(target_id, source_id)
            left_paths.append(left_path)
            right_paths.append(right_path)

        return left_paths, right_paths

    def _get_context_words(self, note):

        pairs = note.get_tlinked_entities() # entity pairs
        left_context = []
        right_context = []
        labels = []
        for pair in pairs:
            if pair['rel_type'] != 'None':
                source = pair['src_entity'][0]
                target = pair['target_entity'][0]
                if source.get('sentence_num', 0) != target.get('sentence_num', -1): # for now, only consider entities in the same sentence
                    continue

                sentence_num = source['sentence_num']
                labels.append(pair['rel_type'])

                if source['token_offset'] <= target['token_offset']:
                    left = source
                    right = target
                else:
                    left = target
                    right = source
                left_indexes, right_indexes = self._get_context(left['token_offset'], right['token_offset'],
                                                                len(note.pre_processed_text[sentence_num]))
                if source['token_offset'] <= target['token_offset']:
                    # source vector, left means "source" here
                    left_context.append([token['token'] for token in
                                    note.pre_processed_text[sentence_num][left_indexes[0]:left_indexes[1]]])
                    # target vector, right means "target" here
                    right_context.append([token['token'] for token in
                                    note.pre_processed_text[sentence_num][right_indexes[0]:right_indexes[1]]])
                else:
                    right_context.append([token['token'] for token in
                                         note.pre_processed_text[sentence_num][left_indexes[0]:left_indexes[1]]])
                    left_context.append([token['token'] for token in
                                      note.pre_processed_text[sentence_num][right_indexes[0]:right_indexes[1]]])

        return left_context, right_context, labels

    def _get_context_representations(self, note, word_vectors):
        """
        @param note:
        @param word_vectors:
        @return: context word embeddings
        """
        source_context, target_context, labels = self._get_context_words(note)

        # print left_context
        # print right_context
        del_list = []
        left_vecs = None
        for j, path in enumerate(source_context):
            vecs_context = None
            for word in path:
                # try to get embedding for a given word. If the word is not in the vocabulary, use a vector of all 1s.
                try:
                    embedding = word_vectors[word]
                except KeyError:
                    embedding = np.random.uniform(low=-0.5, high=0.5, size=(EMBEDDING_DIM))

                # reshape to 3 dimensions so embeddings can be concatenated together to form the final input values
                embedding = embedding[np.newaxis, :, np.newaxis]
                if vecs_context is None:
                    vecs_context = embedding
                else:
                    vecs_context = np.concatenate((vecs_context, embedding), axis=2)

            # if there were no vectors, the link involves the document creation time or is a cross sentence relation.
            # add index to list to indexes to remove and continue
            if vecs_context is None:
                del_list.append(j)
                continue
            if left_vecs is None:
                left_vecs = vecs_context
            else:
                left_vecs = Network._pad_and_concatenate(left_vecs, vecs_context, axis=0, pad_left=[2])

        right_vecs = None
        for j, path in enumerate(target_context):
            vecs_context = None
            for word in path:
                # try to get embedding for a given word. If the word is not in the vocabulary, use a vector of all 1s.
                try:
                    embedding = word_vectors[word]
                except KeyError:
                    embedding = np.random.uniform(low=-0.5, high=0.5, size=(EMBEDDING_DIM))

                # reshape to 3 dimensions so embeddings can be concatenated together to form the final input values
                embedding = embedding[np.newaxis, :, np.newaxis]
                if vecs_context is None:
                    vecs_context = embedding
                else:
                    vecs_context = np.concatenate((vecs_context, embedding), axis=2)

            # if there were no vectors, the link involves the document creation time or is a cross sentence relation.
            # add index to list to indexes to remove and continue
            if vecs_context is None:
                del_list.append(j)
                continue
            if right_vecs is None:
                right_vecs = vecs_context
            else:
                right_vecs = Network._pad_and_concatenate(right_vecs, vecs_context, axis=0, pad_left=[2])

        return left_vecs, right_vecs, del_list, labels


    def _get_context(self, left_offset, right_offset, max_len):
        l_context_l_edge = max(0, left_offset - 5)
        l_context_r_edge = min(right_offset+1, left_offset + 11) # cover the other entity too
        r_context_l_edge = max(left_offset, right_offset - 10) # cover the other entity too
        r_context_r_edge = min(right_offset + 6, max_len)
        return (l_context_l_edge, l_context_r_edge), (r_context_l_edge, r_context_r_edge)

    def get_uniform_class_weights(self, labels):
        '''
        get a dictionary of weights for each class. Used to combat imbalanced data problems by reducing
        the impact of highly represented classes. Has no effect on equally distributed data
        labels is composed of one-hot vectors.
        '''
        n_samples = len(labels)
        n_classes = len(labels[0])

        weights = n_samples/ (n_classes * np.sum(labels, axis=0))

        return weights

    def _convert_str_labels_to_int(self, labels):
        '''
        convert tlink labels to integers so they can be processed by the network
        '''

        for i, label in enumerate(labels):
            if label == "IDENTITY":
                labels[i] = "SIMULTANEOUS"
            elif label not in LABELS:
                labels[i] = "None"

        return [LABELS.index(x) for x in labels]

    def _convert_int_labels_to_str(self, labels):
        '''
        convert ints to tlink labels so network output can be understood
        '''

        return [LABELS[s] if s < 12 else "None" for s in labels]

    def reverse_labels(self, labels):
        # LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IBEFORE", "IAFTER", "IS_INCLUDED", "INCLUDES",
        #  "DURING", "BEGINS", "BEGUN_BY", "ENDS", "ENDED_BY", "None"]
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
            elif label == 5:
                processed_labels.append(6)
            elif label == 6:
                processed_labels.append(5)
            elif label == 7:
                processed_labels.append(7)
            elif label == 8:
                processed_labels.append(9)
            elif label == 9:
                processed_labels.append(8)
            elif label == 10:
                processed_labels.append(11)
            elif label == 11:
                processed_labels.append(10)
            else:  # label for unlinked pairs (should have int 0)
                processed_labels.append(12)

            self.label_reverse_map[label] = processed_labels[-1]

        return processed_labels

    @staticmethod
    def class_confusion(predicted, actual, nb_classes):
        '''
        print confusion matrix for two lists of labels. A given index in both lists should correspond to the same data sample
        '''

        confusion = np.zeros((nb_classes, nb_classes), dtype='int16')

        for true, pred in zip(actual, predicted):
            # build confusion matrix
            confusion[true, pred] += 1
        pickle.dump(confusion, open('evaluation.pkl', 'w'))
        # print confusion matrix
        print "confusion matrix"
        print "rows: actual labels.  columns: predicted labels."
        for i, row in enumerate(confusion):
            print i, ": ", row

        print classification_report(actual, predicted, digits=3)
        # print "micro scores:"
        # print precision_recall_fscore_support(actual, predicted, average='micro')

if __name__ == "__main__":
    #test = NNModel()
    with open("note.dump") as n:
        tmp_note = pickle.load(n)
#    tmp_note = TimeNote("APW19980418.0210.tml.TE3input", "APW19980418.0210.tml")
    # print tmp_note.pre_processed_text[2][16]
#    with open("note.dump", 'wb') as n:
#        pickle.dump(tmp_note, n)
    #test.train([tmp_note])

    # labels = tmp_note.get_tlink_labels()
    # labels = _convert_str_labels_to_int(labels)
    # _labels = to_categorical(labels,7)
    # print len(labels)
    # print labels
    # input1 = np.random.random((len(labels),EMBEDDING_DIM, 16))
    # input2 = np.random.random((len(labels),EMBEDDING_DIM, 16))
    # # labels = np.random.randint(7, size=(10000,1))
    # test.model.fit([input1,input2], _labels, nb_epoch=100)
    # print test.model.predict_classes([input1,input2])
    # print labels
    pass
