import os
import cPickle as pickle
import json
import copy

import numpy as np
np.random.seed(1337)
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Graph
from keras.layers import Embedding, LSTM, Dense, Merge, MaxPooling1D, TimeDistributed, Flatten, Masking, Input, Dropout, Permute
from keras.regularizers import l2, activity_l2
from sklearn.metrics import classification_report
from keras.optimizers import Adam, SGD
from keras.constraints import maxnorm

import network_sem10

from word2vec import load_word2vec_binary


global ignore_order
ignore_order = False
network_sem10.set_ignore_order(ignore_order)

def get_untrained_model(encoder_dropout=0, decoder_dropout=0, input_dropout=0, reg_W=0, reg_B=0, reg_act=0,
                        LSTM_size=256, dense_size=100, maxpooling=True, data_dim=300, max_len=22):
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
    print "Set encoder dropout ", encoder_dropout

    # with maxpooling
    if maxpooling:
        encoder_L.add(LSTM(LSTM_size, return_sequences=True, inner_activation="hard_sigmoid"))
        if encoder_dropout != 0:
            encoder_L.add(TimeDistributed(Dropout(encoder_dropout)))
        encoder_L.add(Permute((2, 1)))
        encoder_L.add(MaxPooling1D(pool_length=LSTM_size))
        encoder_L.add(Flatten())

    # without maxpooling
    else:
        encoder_L.add(Masking(mask_value=0.))
        encoder_L.add(LSTM(LSTM_size, return_sequences=False, inner_activation="hard_sigmoid"))
        if encoder_dropout != 0:
            encoder_L.add(Dropout(encoder_dropout))

    # encode the second entity
    encoder_R = Sequential()

    encoder_R.add(Dropout(input_dropout, input_shape=(data_dim, max_len)))
    encoder_R.add(Permute((2, 1)))

    # with maxpooling
    if maxpooling:
        encoder_R.add(LSTM(LSTM_size, return_sequences=True, inner_activation="hard_sigmoid"))
        if encoder_dropout != 0:
            encoder_R.add(TimeDistributed(Dropout(encoder_dropout)))
        encoder_R.add(Permute((2, 1)))
        encoder_R.add(MaxPooling1D(pool_length=LSTM_size))
        encoder_R.add(Flatten())

    else:
    # without maxpooling
        encoder_R.add(Masking(mask_value=0.))
        encoder_R.add(LSTM(LSTM_size, return_sequences=False, inner_activation="hard_sigmoid"))
        if encoder_dropout != 0:
            encoder_R.add(Dropout(encoder_dropout))

    # combine and classify entities as a single relation
    decoder = Sequential()
    decoder.add(Merge([encoder_R, encoder_L], mode='concat'))

    decoder.add(Dense(dense_size, W_regularizer=W_reg, b_regularizer=B_reg, activity_regularizer=act_reg, activation='relu', W_constraint=maxnorm(4)))
    if decoder_dropout != 0:
        decoder.add(Dropout(decoder_dropout))
        print "Set decoder dropout ", decoder_dropout
    decoder.add(Dense(1, W_regularizer=None, b_regularizer=B_reg, activity_regularizer=act_reg, activation='sigmoid'))

    # compile the final model
    # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # learning rate 0.001 is the default value
    opt = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    decoder.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return decoder


def _get_training_input(notes, no_none=False, presence=False, shuffle=True):
    XL, XR, labels = network_sem10._get_training_input(notes, shuffle=shuffle)
    binary_labels = [1 if x%10 else 0 for x in labels]
    return XL, XR, binary_labels


def train_model(notes, model=None, epochs=5, training_input=None, test_input=None, weight_classes=False, batch_size=100,
    encoder_dropout=0, decoder_dropout=0, input_dropout=0, reg_W=0, reg_B=0, reg_act=0, LSTM_size=256, dense_size=100,
    maxpooling=True, data_dim=300, max_len='auto', nb_classes=2, callbacks=[]):
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
        XL, XR, Y = _get_training_input(notes)
        print "training input obtained..."
    else:
        XL, XR, Y = training_input

    if test_input:
        test_XL, test_XR, test_Y = test_input

    # reformat labels so that they can be used by the NN
    # Y = to_categorical(labels, 2)
    # if test_input:
    #     test_Y = to_categorical(test_labels, 2)
    # print "labels reformed..."

    # use weighting to assist with the imbalanced data set problem
    if weight_classes:
        class_weights = network_sem10.get_uniform_class_weights(Y)
    else:
        class_weights = None

    # infer maximum sequence length
    if max_len == 'auto':
        max_len = XL.shape[2]
    # pad input to reach max_len
    else:
        filler = np.ones((1,1,max_len))
        XL, _ = network_sem10._pad_to_match_dimensions(XL, filler, 2, pad_left=True)
        XR, _ = network_sem10._pad_to_match_dimensions(XR, filler, 2, pad_left=True)
        if test_input:
            test_XL, _ = network_sem10._pad_to_match_dimensions(test_XL, filler, 2, pad_left=True)
            test_XR, _ = network_sem10._pad_to_match_dimensions(test_XR, filler, 2, pad_left=True)

    if model is None:
        model = get_untrained_model(encoder_dropout=encoder_dropout, decoder_dropout=decoder_dropout, input_dropout=input_dropout, reg_W=reg_W, reg_B=reg_B, reg_act=reg_act, LSTM_size=LSTM_size, dense_size=dense_size,
        maxpooling=maxpooling, data_dim=data_dim, max_len=max_len)
    #else:
    #    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if test_input:
        # Use test data for validation
        V_XL = test_XL
        V_XR = test_XR
        V_Y = test_Y
        #V_labels = test_Y

    else:
        # split off validation data with 20 80 split (this way we get the same validation data every time we use this data sample, and can test on it after to get a confusion matrix)
        V_XL = XL[:(XL.shape[0]/5),:,:]
        V_XR = XR[:(XR.shape[0]/5),:,:]
        V_Y = Y [:len(Y)/5]
        #V_labels = Y[:len(Y)/5]

        XL = XL[(XL.shape[0]/5):,:,:]
        XR = XR[(XR.shape[0]/5):,:,:]
        Y  = Y [len(Y)/5:]

    # train the network
    print 'Training network...'
    training_history = model.fit([XL, XR], Y, nb_epoch=epochs, validation_split=0.0, class_weight=class_weights, batch_size=batch_size, validation_data=([V_XL, V_XR], V_Y), callbacks=callbacks)
    test = model.predict_classes([V_XL, V_XR])

    network_sem10.class_confusion(test, V_Y, nb_classes)

    return model, training_history.history
