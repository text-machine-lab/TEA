import os
import cPickle
import glob

import numpy as np
#np.random.seed(1337)

from keras.models import Sequential, Graph
from keras.layers import Embedding, LSTM, Dense, Merge, MaxPooling1D, TimeDistributed, Flatten, Masking, Input, Dropout, Permute
from keras.regularizers import l2, activity_l2
from code.learning.word2vec import load_word2vec_binary
from keras.callbacks import ModelCheckpoint, EarlyStopping

from code.learning.network import Network
from code.config import env_paths
from code.notes.TimeNote import TimeNote


class EventNetwork(object):
    def __init__(self, word_vectors=None):
        self.word_vectors = word_vectors

    def get_untrained_model(self, encoder_dropout=0, decoder_dropout=0.5, input_dropout=0.5, reg_W=0, reg_B=0, reg_act=0,
                            LSTM_size=128, dense_size=30, maxpooling=True, data_dim=300, max_len=10, nb_classes=13):
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
            act_reg = activity_l2(reg_act)
        else:
            act_reg = None

        # encode the first entity
        encoder_L = Sequential()

        encoder_L.add(Dropout(input_dropout, input_shape=(data_dim, max_len)))
        encoder_L.add(Permute((2, 1)))

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

        # encode the pos tags and is_predicate
        encoder_R = Sequential()

        encoder_R.add(Dense(3, input_dim=4)) # dicrectly connected to output lay

         # combine and classify entities as a single relation
        decoder = Sequential()
        decoder.add(Merge([encoder_L, encoder_R], mode='concat'))
        decoder.add(
            Dense(dense_size, W_regularizer=W_reg, b_regularizer=B_reg, activity_regularizer=act_reg, activation='sigmoid'))
        if decoder_dropout != 0:
            decoder.add(Dropout(decoder_dropout))
        decoder.add(
            Dense(1, activation='sigmoid'))

        # compile the final model
        decoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return decoder

    def _extract_word_representations(self, word_list):
        """Given a list of words, return the embeddings"""
        if self.word_vectors is None:
            print 'Loading word embeddings...'
            self.word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin', verbose=0)

        vecs = None
        for word in word_list:
            # try to get embedding for a given word. If the word is not in the vocabulary, use a vector of all 1s.
            try:
                embedding = self.word_vectors[word]
            except KeyError:
                embedding = np.random.uniform(low=-0.5, high=0.5, size=(300))

            # reshape to 3 dimensions so embeddings can be concatenated together to form the final input values
            embedding = embedding[np.newaxis, :, np.newaxis]
            if vecs is None:
                vecs = embedding
            else:
                vecs = np.concatenate((vecs, embedding), axis=2)

        return vecs

    def get_input(self, notes, shuffle=True, neg_ratio=3):

        word_vectors = None
        attribute_vectors = None
        labels = []
        for note in notes:
            print "processing file ", note.annotated_note_path
            if hasattr(note, 'event_ids'):
                event_ids = note.event_ids
            else:
                id_chunk_map, event_ids, timex_ids, sentence_chunks = note.get_id_chunk_map()

            # every event tag corresponds to a list of words, pick the first word
            event_wordIDs = [note.id_to_wordIDs[x][0] for x in event_ids]
            max_id = len(note.id_to_tok) # word ids starts with 1

            all_wordIDs = set(['w'+str(x) for x in range(1,max_id+1)])
            nonevent_wordIDs = all_wordIDs - set(event_wordIDs)
            n_neg_samples = min(len(nonevent_wordIDs), neg_ratio*len(event_wordIDs))
            nonevent_wordIDs = list(nonevent_wordIDs)[0:n_neg_samples]

            training_wordIDs = event_wordIDs + nonevent_wordIDs

            for wordID in training_wordIDs:
                word_index = int(wordID[1:]) # wordID example: 'w31'
                left_edge = max(1, word_index - 4)
                right_edge = min(max_id, word_index + 4)

                context_tokens = [note.id_to_tok['w'+str(x)] for x in range(left_edge, right_edge+1)]
                context_words = [x['token'] for x in context_tokens]
                vecs = self._extract_word_representations(context_words)
                if word_vectors is None:
                    word_vectors = vecs
                else:
                    word_vectors = Network._pad_and_concatenate(word_vectors, vecs, axis=0)

                tok = note.id_to_tok[wordID]
                attributes = np.array([tok.get('is_main_verb', False), tok.get('is_predicate', False),
                                       tok['pos']=='V', tok['pos']=='N'])
                attributes = attributes[np.newaxis, :]
                if attribute_vectors is None:
                    attribute_vectors = attributes
                else:
                    attribute_vectors = np.concatenate((attribute_vectors, attributes), axis=0)

                if wordID in event_wordIDs:
                    labels.append(1)
                else:
                    labels.append(0)

        if shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(word_vectors)
            np.random.set_state(rng_state)
            np.random.shuffle(attribute_vectors)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)

        return word_vectors, attribute_vectors, labels

    def get_test_input(self, note):
        """Given a note, return data for every token"""

        max_id = len(note.id_to_tok)  # word ids starts with 1
        word_vectors = None
        attribute_vectors = None

        for sent_num in note.pre_processed_text:
            for tok in note.pre_processed_text[sent_num]:
                wordID = tok['id']
                word_index = int(wordID[1:]) # wordID example: 'w31'
                left_edge = max(1, word_index - 4)
                right_edge = min(max_id, word_index + 4)

                context_tokens = [note.id_to_tok['w'+str(x)] for x in range(left_edge, right_edge+1)]
                context_words = [x['token'] for x in context_tokens]
                vecs = self._extract_word_representations(context_words)
                if word_vectors is None:
                    word_vectors = vecs
                else:
                    word_vectors = Network._pad_and_concatenate(word_vectors, vecs, axis=0)

                attributes = np.array([tok.get('is_main_verb', False), tok.get('is_predicate', False),
                                       tok['pos']=='V', tok['pos']=='N'])
                attributes = attributes[np.newaxis, :]
                if attribute_vectors is None:
                    attribute_vectors = attributes
                else:
                    attribute_vectors = np.concatenate((attribute_vectors, attributes), axis=0)

        return word_vectors, attribute_vectors

    def get_notes(self, annotated_dir, newsreader_dir, save_notes=False):
        annotated_files = sorted(glob.glob(os.path.join(annotated_dir, '*.tml')))

        base_names = [os.path.basename(x) for x in annotated_files]
        note_files = [os.path.join(newsreader_dir, x[0:x.index(".tml")] + '.parsed.pickle') for x in base_names]

        notes = []
        for i, note_file in enumerate(note_files):
            if os.path.exists(note_file):
                notes.append(cPickle.load(open(note_file)))
            else:
                note = TimeNote(annotated_files[i], None) # we do not need tlinks
                notes.append(note)
                if save_notes:
                    cPickle.dump(note, open(note_file, 'w'))
        return notes

    def train_model(self, training_data, validation_data=None, model_destination='./', epochs=500,
                    weight_classes=False, batch_size=256,
                    encoder_dropout=0, decoder_dropout=0.5, input_dropout=0.5, reg_W=0, reg_B=0, reg_act=0,
                    LSTM_size=128, dense_size=30, maxpooling=True, data_dim=300, max_len='auto'):

        XL, XR, Y = training_data
        print "training data shape: ", XL.shape

        # reformat labels so that they can be used by the NN
        #Y = to_categorical(Y, 2)

        # use weighting to assist with the imbalanced data set problem
        if weight_classes:
            N = len(Y)
            n_pos = sum(Y)
            neg_weight = 1.0 * n_pos / N # inversely proportional to frequency
            class_weight = {1: 1-neg_weight, 0: neg_weight}

        # infer maximum sequence length
        if max_len == 'auto':
            max_len = XL.shape[2]
        # pad input to reach max_len
        else:
            filler = np.ones((1, 1, max_len))
            XL, _ = Network._pad_to_match_dimensions(XL, filler, 2, pad_left=True)
            XR, _ = Network._pad_to_match_dimensions(XR, filler, 2, pad_left=True)

        model = self.get_untrained_model(encoder_dropout=encoder_dropout, decoder_dropout=decoder_dropout,
                                         input_dropout=input_dropout, reg_W=reg_W, reg_B=reg_B, reg_act=reg_act,
                                         LSTM_size=LSTM_size, dense_size=dense_size,
                                         maxpooling=maxpooling, data_dim=data_dim, max_len=max_len)

        # split off validation data with 20 80 split (this way we get the same validation data every time we use this data sample, and can test on it after to get a confusion matrix)
        if validation_data is None:
            V_XL = XL[:(XL.shape[0] / 5), :, :]
            V_XR = XR[:(XR.shape[0] / 5), :, :]
            V_Y = Y[:(Y.shape[0] / 5), :]
            #V_labels = labels[:(Y.shape[0] / 5)]

            XL = XL[(XL.shape[0] / 5):, :, :]
            XR = XR[(XR.shape[0] / 5):, :, :]
            Y = Y[(Y.shape[0] / 5):, :]
        else:
            V_XL, V_XR, V_Y = validation_data

        # train the network
        print 'Training network...'
        earlystopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_destination + 'model.h5', monitor='val_acc', save_best_only=True)

        training_history = model.fit([XL, XR], Y, nb_epoch=epochs, validation_split=0, class_weight=class_weight,
                                     batch_size=batch_size, validation_data=([V_XL, V_XR], V_Y),
                                     callbacks=[checkpoint, earlystopping])

        test = model.predict_classes([V_XL, V_XR])

        Network.class_confusion(test, V_Y, 2)

        return model, training_history.history

    def predict(self, model, test_data, predict_prob=False):

        XL, XR = test_data

        # get expected length of model input
        model_input_len = model.input_shape[0][2]

        if model_input_len > XL.shape[2]:
            # pad input matrix to fit expected length
            filler = np.ones((1, 1, model_input_len))
            XL, _ = Network._pad_to_match_dimensions(XL, filler, 2, pad_left=True)
        else:
            XL = Network._strip_to_length(XL, model_input_len, 2)

        print "predicting..."
        labels = model.predict_classes([XL, XR])
        if predict_prob:
            probs = model.predict_proba([XL, XR])
        else:
            probs = None

        return labels, probs


    # def single_predict(self, XL, XR, model, predict_prob=False):
    #     '''
    #     predict using a trained single pass model
    #     '''
    #
    #     # get expected length of model input
    #     model_input_len = model.input_shape[0][2]
    #
    #     if model_input_len > XL.shape[2]:
    #         # pad input matrix to fit expected length
    #         filler = np.ones((1, 1, model_input_len))
    #         XL, _ = Network._pad_to_match_dimensions(XL, filler, 2, pad_left=True)
    #         XR, _ = Network._pad_to_match_dimensions(XR, filler, 2, pad_left=True)
    #     else:
    #         XL = Network._strip_to_length(XL, model_input_len, 2)
    #         XR = Network._strip_to_length(XR, model_input_len, 2)
    #
    #     print 'Predicting...'
    #     labels = model.predict_classes([XL, XR])
    #     if predict_prob:
    #         probs = model.predict_proba([XL, XR])
    #     else:
    #         probs = None
    #
    #
    #     return self._convert_int_labels_to_str(labels), probs

if __name__ == "__main__":
    import json

    training_dir = '../sandbox/training_set/'
    val_dir = '../sandbox/val_set'
    newsreader_dir = './newsreader_annotations/12cls_half_neg/'
    model_dir = './model_destination/event/'

    network = EventNetwork()
    training_notes = network.get_notes(training_dir, newsreader_dir, save_notes=False)
    # # downsample to get a quick check
    # np.random.shuffle(training_notes)
    # training_notes = training_notes[0:50]

    val_notes = network.get_notes(val_dir, newsreader_dir, save_notes=False)
    # downsample to get a quick check
    #val_notes = val_notes[0:5]
    print "all notes loaded"

    training_data = network.get_input(training_notes)
    print "training data loaded"

    val_data = network.get_input(val_notes)
    print "validation data loaded"

    print "all data loaded successfully"

    NN, history = network.train_model(training_data, validation_data=val_data, model_destination=model_dir, weight_classes=True, maxpooling=True)
    architecture = NN.to_json()
    open(model_dir + '.arch.json', "wb").write(architecture)
    NN.save_weights(model_dir + '.weights.h5')
    NN.save(model_dir + 'final_model.h5')
    json.dump(history, open(model_dir + 'training_history.json', 'w'))
