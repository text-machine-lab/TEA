from __future__ import print_function
import os
import sys
import numpy as np
# np.random.seed(1337)
from keras.models import Model, Sequential, load_model
from keras.regularizers import l1_l2, l2
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Reshape, LSTM, Dense, concatenate, average, add, MaxPooling1D, TimeDistributed, Flatten, Lambda, Input, Dropout, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop, SGD
from collections import deque
import tensorflow as tf

# from ntm2 import NeuralTuringMachine as NTM
# from ntm2 import SingleKeyNTM as NTM
from ntm2 import SimpleNTM as NTM

# LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IBEFORE", "IAFTER", "IS_INCLUDED", "INCLUDES",
#           "DURING", "BEGINS", "BEGUN_BY", "ENDS", "ENDED_BY", "None"]
LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IS_INCLUDED", "INCLUDES", "None"] # TimeBank Dense labels
EMBEDDING_DIM = 300
DENSE_LABELS = True
MAX_LEN = 16  # max # of words on each branch of path

import keras.backend as K

def my_weighted_loss(onehot_labels, logits):
    """scale loss based on class frequency"""

    # compute weights based on their frequencies
    class_weights = 2000.0 / (6 * np.array([50, 500, 500, 125, 125, 1000]))
    # computer weights based on onehot labels
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss

def my_weighted_loss2(onehot_labels, logits):
    """scale logits based on frequency"""

    # compute weights based on their frequencies
    class_weights = 2000.0 / (6 * np.array([50, 500, 500, 125, 125, 1000]))

    unweighted_softmax = tf.nn.softmax(logits)
    weighted_softmax = tf.multiply(class_weights, unweighted_softmax)
    weighted_softmax /= tf.reduce_sum(weighted_softmax)  # normalize
    loss = tf.losses.absolute_difference(labels=[onehot_labels], predictions=[weighted_softmax])

    return loss

def get_lstm_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=300):
    controller = StatefulController(units=controller_output_dim,
                                    kernel_initializer='random_normal',
                                    bias_initializer='random_normal',
                                    activation='tanh',
                                    stateful=True,  # must be true, because in controller every step is a batch?
                                    return_state=True,
                                    return_sequences=False,
                                    # does not matter because for controller the sequence len is 1?
                                    implementation=2,  # best for gpu. other ones also might not work.
                                    batch_input_shape=(batch_size, max_steps, controller_input_dim))
    controller.build(input_shape=(batch_size, max_steps, controller_input_dim))

    return controller

def get_dense_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=300):
    #TODO: not working properly now

    controller = Sequential()
    controller.name = 'Dense'

    controller.add(Dense(units=controller_output_dim,
                        activation='relu',
                        bias_initializer='zeros',
                        input_shape=(controller_input_dim,)))

    # controller.summary()
    controller.compile(loss='binary_crossentropy', optimizer=Adam(lr=.0005, clipnorm=10.), metrics=['binary_accuracy'])

    return controller

def get_untrained_model(encoder_dropout=0, decoder_dropout=0, input_dropout=0, LSTM_size=32, dense_size=256,
                            max_len=15, nb_classes=13):


    raw_input_l = Input(shape=(max_len, EMBEDDING_DIM))  # (steps, EMBEDDING_DIM)
    raw_input_r = Input(shape=(max_len, EMBEDDING_DIM))

    input_l = Dropout(input_dropout)(raw_input_l)
    input_r = Dropout(input_dropout)(raw_input_r)

    type_input = Input(shape=(1,))
    pair_type = Dense(2)(type_input)

    ## option 1: two branches
    encoder_l = Bidirectional(LSTM(LSTM_size, return_sequences=True), merge_mode='sum')(input_l)
    # encoder_l = LSTM(LSTM_size, return_sequences=True)(encoder_l)
    encoder_l = MaxPooling1D(pool_size=max_len)(encoder_l)  # (1, LSTM_size)
    encoder_l = Flatten()(encoder_l)
    encoder_r = Bidirectional(LSTM(LSTM_size, return_sequences=True), merge_mode='sum')(input_r)
    # encoder_r = LSTM(LSTM_size, return_sequences=True)(encoder_r)
    encoder_r = MaxPooling1D(pool_size=max_len)(encoder_r)  # (1, LSTM_size)
    encoder_r = Flatten()(encoder_r)
    encoder = concatenate([encoder_l, encoder_r, pair_type])  # (2*LSTM_size)

    ## option 2: no branch
    # input = concatenate([input_l, input_r], axis=-2)
    # encoder = Bidirectional(LSTM(LSTM_size, return_sequences=True))(input)
    # encoder = MaxPooling1D(pool_size=max_len)(encoder)  # (1, 2*LSTM_size)
    # encoder = Flatten()(encoder)

    hidden = Dense(dense_size, activation='relu')(encoder)
    hidden = Dropout(decoder_dropout)(hidden)
    softmax = Dense(nb_classes, activation='softmax')(hidden)

    model = Model(inputs=[raw_input_l, raw_input_r, type_input], outputs=[softmax])

    # compile the final model
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def get_untrained_model2(encoder_dropout=0, decoder_dropout=0, input_dropout=0, LSTM_size=32, dense_size=256,
                            max_len=15, nb_classes=13):


    raw_input_l = Input(shape=(max_len, EMBEDDING_DIM))  # (steps, EMBEDDING_DIM)
    raw_input_r = Input(shape=(max_len, EMBEDDING_DIM))

    input_l = Dropout(input_dropout)(raw_input_l)
    input_r = Dropout(input_dropout)(raw_input_r)

    type_input = Input(shape=(1,))
    pair_type = Dense(2)(type_input)

    ## option 1: two branches
    encoder_l = Bidirectional(LSTM(LSTM_size, return_sequences=True, activation='tanh'), merge_mode='sum')(input_l)
    # encoder_l = LSTM(LSTM_size, return_sequences=True)(encoder_l)
    encoder_l = MaxPooling1D(pool_size=max_len)(encoder_l)  # (1, LSTM_size)
    encoder_l = Flatten()(encoder_l)
    encoder_r = Bidirectional(LSTM(LSTM_size, return_sequences=True, activation='tanh'), merge_mode='sum')(input_r)
    # encoder_r = LSTM(LSTM_size, return_sequences=True)(encoder_r)
    encoder_r = MaxPooling1D(pool_size=max_len)(encoder_r)  # (1, LSTM_size)
    encoder_r = Flatten()(encoder_r)
    encoder = concatenate([encoder_l, encoder_r, pair_type])  # (2*LSTM_size)

    ## option 2: no branch
    # input = concatenate([input_l, input_r], axis=-2)
    # encoder = Bidirectional(LSTM(LSTM_size, return_sequences=True))(input)
    # encoder = MaxPooling1D(pool_size=max_len)(encoder)  # (1, 2*LSTM_size)
    # encoder = Flatten()(encoder)

    hidden = Dense(dense_size, activation='tanh')(encoder)
    hidden = Dropout(decoder_dropout)(hidden)

    aux_outlayer = Dense(2, activation='softmax', name="aux_out")(hidden)
    aux_hidden = concatenate([aux_outlayer, hidden])

    outlayer = Dense(nb_classes, activation='softmax', name="main_out")(aux_hidden)

    model = Model(inputs=[raw_input_l, raw_input_r, type_input], outputs=[outlayer, aux_outlayer])

    # compile the final model
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], loss_weights=[1., 2.])
    return model


def get_untrained_model4_2(encoder_dropout=0, decoder_dropout=0, input_dropout=0, LSTM_size=128, dense_size=256,
                            max_len=15, nb_classes=13):


    raw_input_l = Input(shape=(max_len, EMBEDDING_DIM))  # (steps, EMBEDDING_DIM)
    raw_input_r = Input(shape=(max_len, EMBEDDING_DIM))
    raw_context_input_l = Input(shape=(max_len, EMBEDDING_DIM))  # (steps, EMBEDDING_DIM)
    raw_context_input_r = Input(shape=(max_len, EMBEDDING_DIM))

    input_l = Dropout(input_dropout)(raw_input_l)
    input_r = Dropout(input_dropout)(raw_input_r)
    context_input_l = Dropout(input_dropout)(raw_context_input_l)
    context_input_r = Dropout(input_dropout)(raw_context_input_r)

    type_input = Input(shape=(1,))

    encoder_l = Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum')(input_l)
    encoder_l = MaxPooling1D(pool_size=max_len)(encoder_l)  # (1, LSTM_size)
    encoder_l = Flatten()(encoder_l)
    encoder_r = Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum')(input_r)
    encoder_r = MaxPooling1D(pool_size=max_len)(encoder_r)  # (1, LSTM_size)
    encoder_r = Flatten()(encoder_r)

    context_l = Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum')(context_input_l)
    context_l = MaxPooling1D(pool_size=max_len)(context_l)  # (1, LSTM_size)
    context_l = Flatten()(context_l)
    context_r = Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum')(context_input_r)
    context_r = MaxPooling1D(pool_size=max_len)(context_r)  # (1, LSTM_size)
    context_r = Flatten()(context_r)

    left_encoder = concatenate([encoder_l, context_l])
    left_encoder = Dense(128)(left_encoder)
    right_encoder = concatenate([encoder_r, context_r])
    right_encoder = Dense(128)(right_encoder)

    encoder = concatenate([left_encoder, right_encoder])
    # encoder = Dropout(0.5)(encoder)
    encoder = concatenate([encoder, type_input])  # do not dropout pair_type

    hidden = Dense(256, activation='relu')(encoder)
    hidden = Dropout(decoder_dropout)(hidden)
    hidden = concatenate([hidden, type_input])

    # hidden_2 = Dense(128, activation='tanh')(hidden)
    # hidden_2 = Dropout(decoder_dropout)(hidden_2)
    # hidden_2 = concatenate([hidden_2, type_input])  # use pair type again to enforce it

    # outlayer = Dense(nb_classes, activation='softmax', name="main_out")(hidden_2)
    outlayer = Dense(nb_classes, activation='softmax', name="main_out")(hidden)

    model = Model(inputs=[raw_input_l, raw_input_r, type_input, raw_context_input_l, raw_context_input_r], outputs=[outlayer])

    # compile the final model
    # model.summary()
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], loss_weights=[1., 2.])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    return model


def get_untrained_model4_3(encoder_dropout=0, decoder_dropout=0, input_dropout=0, LSTM_size=128, dense_size=256,
                            max_len=15, nb_classes=13):


    raw_input_l = Input(shape=(max_len, EMBEDDING_DIM))  # (steps, EMBEDDING_DIM)
    raw_input_r = Input(shape=(max_len, EMBEDDING_DIM))
    raw_context_input_l = Input(shape=(max_len, EMBEDDING_DIM))  # (steps, EMBEDDING_DIM)
    raw_context_input_r = Input(shape=(max_len, EMBEDDING_DIM))

    input_l = Dropout(input_dropout)(raw_input_l)
    input_r = Dropout(input_dropout)(raw_input_r)
    context_input_l = Dropout(input_dropout)(raw_context_input_l)
    context_input_r = Dropout(input_dropout)(raw_context_input_r)

    type_input = Input(shape=(1,))
    time_diff = Input(shape=(3,))

    encoder_l = Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum')(input_l)
    encoder_l = MaxPooling1D(pool_size=max_len)(encoder_l)  # (1, LSTM_size)
    encoder_l = Flatten()(encoder_l)
    encoder_r = Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum')(input_r)
    encoder_r = MaxPooling1D(pool_size=max_len)(encoder_r)  # (1, LSTM_size)
    encoder_r = Flatten()(encoder_r)

    context_l = Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum')(context_input_l)
    context_l = MaxPooling1D(pool_size=max_len)(context_l)  # (1, LSTM_size)
    context_l = Flatten()(context_l)
    context_r = Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum')(context_input_r)
    context_r = MaxPooling1D(pool_size=max_len)(context_r)  # (1, LSTM_size)
    context_r = Flatten()(context_r)

    left_encoder = concatenate([encoder_l, context_l])
    left_encoder = Dense(128)(left_encoder)
    right_encoder = concatenate([encoder_r, context_r])
    right_encoder = Dense(128)(right_encoder)

    encoder = concatenate([left_encoder, right_encoder, type_input, time_diff])

    hidden = Dense(256, activation='relu')(encoder)
    hidden = Dropout(decoder_dropout)(hidden)
    hidden = concatenate([hidden, type_input, time_diff])

    # hidden_2 = Dense(128, activation='tanh')(hidden)
    # hidden_2 = Dropout(decoder_dropout)(hidden_2)
    # hidden_2 = concatenate([hidden_2, type_input])  # use pair type again to enforce it

    # outlayer = Dense(nb_classes, activation='softmax', name="main_out")(hidden_2)
    outlayer = Dense(nb_classes, activation='softmax', name="main_out")(hidden)

    model = Model(inputs=[raw_input_l, raw_input_r, type_input, raw_context_input_l, raw_context_input_r, time_diff], outputs=[outlayer])

    # compile the final model
    # model.summary()
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], loss_weights=[1., 2.])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    return model


def get_ntm_model4_1(batch_size=100, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=13, input_dropout=0.3, has_auxiliary=False):
    """sequential, no early output"""
    from keras.backend import reverse

    left_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_branch = Dropout(input_dropout)(left_input)
    left_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_branch)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Reshape((batch_size, -1))(left_branch)  # (1, batch_size, 128)

    right_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_branch = Dropout(input_dropout)(right_input)
    right_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_branch)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Reshape((batch_size, -1))(right_branch)  # (1, batch_size, 128)

    left_context_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_context = Dropout(input_dropout)(left_context_input)
    left_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_context)
    left_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_context)
    left_context = Reshape((batch_size, -1))(left_context)  # (1, batch_size, 128)

    right_context_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_context = Dropout(input_dropout)(right_context_input)
    right_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_context)
    right_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_context)
    right_context = Reshape((batch_size, -1))(right_context)  # (1, batch_size, 128)

    type_input = Input(batch_shape=(1, batch_size, 1))
    # type_input = Reshape((batch_size, 1))(type_input)
    # pair_type = TimeDistributed(Dense(2))(type_input)

    concat = concatenate([left_branch, right_branch, type_input, left_context, right_context])
    hidden_lstm = TimeDistributed(Dense(256, activation='relu'))(concat)
    # hidden_lstm = Dropout(0.3)(hidden_lstm)
    hidden_lstm2 = TimeDistributed(Dense(128, activation='tanh'))(hidden_lstm)

    left_encoder = TimeDistributed(Dense(64, activation='tanh'))(left_branch)
    right_encoder = TimeDistributed(Dense(64, activation='tanh'))(right_branch)

    encoder = concatenate([left_encoder, right_encoder, hidden_lstm2])
    # encoder = concatenate([left_encoder, right_encoder, hidden_lstm])
    # encoder = Dropout(0.3)(encoder)
    encoder = concatenate([encoder, type_input])

    # controller_input_dim, controller_output_dim = controller_shape(256+1, ntm_output_dim, m_depth,
    #                                                                n_slots, shift_range, read_heads, write_heads)

    # we feed in controller (# documents, # pairs, data_dim)
    # so max_steps here is # pairs
    # controller = get_lstm_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=batch_size)

    # model.name = "NTM_-_" + controller.name

    # NTM_F = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
    #             read_heads=read_heads, write_heads=write_heads, controller_model=controller,
    #             return_sequences=True, batch_input_shape=(1, batch_size, 256+1), stateful=True,
    #             activation='relu')
    # NTM_B = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
    #             read_heads=read_heads, write_heads=write_heads, controller_model=controller,
    #             return_sequences=True, batch_input_shape=(1, batch_size, 256+1), stateful=True,
    #             activation='relu', go_backwards=True)


    NTM_F = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=False,
                return_sequences=True, input_shape=(batch_size, 256 + 1), batch_size=1, stateful=False,
                activation='linear')
    NTM_B = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=False,
                return_sequences=True, input_shape=(batch_size, 256 + 1), batch_size=1, stateful=False,
                activation='linear', go_backwards=True)

    # ntm_layer = Bidirectional(ntm, merge_mode='ave')(encoder)

    ntm_forward = NTM_F(encoder)
    ntm_backward = NTM_B(encoder)

    # # make a layer to reverse output
    # Reverse = Lambda(lambda x: reverse(x, axes=-2), output_shape=(batch_size, ntm_output_dim))
    # ntm_backward = Reverse(ntm_backward)

    ntm_layer = average([ntm_forward, ntm_backward])
    # ntm_layer = Dropout(0.3)(ntm_layer)

    hidden_ntm = TimeDistributed(Dense(128, activation='relu'))(ntm_layer)
    hidden_ntm = Dropout(0.3)(hidden_ntm)
    hidden_ntm2 = TimeDistributed(Dense(64, activation='tanh'))(hidden_ntm)
    # concat_decoder = concatenate([hidden_lstm2, hidden_ntm2, type_input])
    concat_decoder = concatenate([hidden_ntm2, type_input])
    # concat_decoder = concatenate([hidden_ntm, type_input])

    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(concat_decoder)

    if has_auxiliary:
        auxiliary_outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='aux_out')(hidden_lstm)
        model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input], outputs=[outlayer, auxiliary_outlayer])
        # model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'], loss_weights=[1.5, 1.])
    else:
        model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input], outputs=[outlayer])
        # model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

    return model


def get_ntm_model5_1(batch_size=None, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=13, input_dropout=0.3, has_auxiliary=False):
    """feed softmax to ntm"""
    from keras.backend import reshape
    ReshapeBranch = Lambda(lambda x: reshape(x, (1, -1, 128)))

    left_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_branch = Dropout(input_dropout)(left_input)
    left_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_branch)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    # left_branch = Reshape((-1, 128))(left_branch)  # (1, batch_size, 128)
    left_branch = ReshapeBranch(left_branch)

    right_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_branch = Dropout(input_dropout)(right_input)
    right_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_branch)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    # right_branch = Reshape((-1, 128))(right_branch)  # (1, batch_size, 128)
    right_branch = ReshapeBranch(right_branch)

    left_context_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_context = Dropout(input_dropout)(left_context_input)
    left_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_context)
    left_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_context)
    # left_context = Reshape((-1, 128))(left_context)  # (1, batch_size, 128)
    left_context = ReshapeBranch(left_context)

    right_context_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_context = Dropout(input_dropout)(right_context_input)
    right_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_context)
    right_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_context)
    # right_context = Reshape((-1, 128))(right_context)  # (1, batch_size, 128)
    right_context = ReshapeBranch(right_context)

    type_input = Input(batch_shape=(1, batch_size, 1))
    # type_input = Reshape((batch_size, 1))(type_input)
    # pair_type = TimeDistributed(Dense(2))(type_input)

    concat = concatenate([left_branch, right_branch, type_input, left_context, right_context])
    hidden_lstm = TimeDistributed(Dense(256, activation='relu'))(concat)
    hidden_lstm = Dropout(0.3)(hidden_lstm)
    aux_softmax = TimeDistributed(Dense(nb_classes, activation='softmax'), name='aux_out')(hidden_lstm)

    left_encoder = TimeDistributed(Dense(64, activation='tanh'))(left_branch)
    right_encoder = TimeDistributed(Dense(64, activation='tanh'))(right_branch)

    encoder = concatenate([left_encoder, right_encoder, aux_softmax])
    # encoder = concatenate([left_encoder, right_encoder, hidden_lstm])
    # encoder = Dropout(0.3)(encoder)
    # encoder = concatenate([encoder, type_input])

    NTM_F = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
                return_sequences=True, input_shape=(batch_size, 128 + nb_classes), batch_size=1, stateful=False,
                activation='linear')
    NTM_B = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
                return_sequences=True, input_shape=(batch_size, 128 + nb_classes), batch_size=1, stateful=False,
                activation='linear', go_backwards=True)

    # ntm_layer = Bidirectional(ntm, merge_mode='ave')(encoder)

    ntm_forward = NTM_F(encoder)
    ntm_backward = NTM_B(encoder)

    # # make a layer to reverse output
    # Reverse = Lambda(lambda x: reverse(x, axes=-2), output_shape=(batch_size, ntm_output_dim))
    # ntm_backward = Reverse(ntm_backward)

    ntm_layer = average([ntm_forward, ntm_backward])
    # ntm_layer = Dropout(0.3)(ntm_layer)

    hidden_ntm = TimeDistributed(Dense(128, activation='relu'))(ntm_layer)
    hidden_ntm = Dropout(0.3)(hidden_ntm)
    concat_decoder = concatenate([hidden_ntm, hidden_lstm])
    decoder = TimeDistributed(Dense(128, activation='relu'))(concat_decoder)
    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(decoder)
    # outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(hidden_ntm)

    model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input], outputs=[outlayer, aux_softmax])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'], loss_weights=[1., 0.5])

    return model


def get_ntm_model5_3(batch_size=5, group_size=None, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=13, input_dropout=0.3, embedding_matrix=None, **kwargs):
    """feed softmax to ntm"""

    left_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_input = Input(batch_shape=(batch_size, group_size, max_len))
    left_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    type_input = Input(batch_shape=(batch_size, group_size, 1))
    time_diff_input = Input(batch_shape=(batch_size, group_size, 3))

    # Shared embedding layer
    embedding = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)

    # group_size here is the # of pairs in a group
    left_input_emb = embedding(left_input)
    left_branch = Dropout(input_dropout)(left_input_emb)
    left_branch = TimeDistributed(
        Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_branch)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Lambda(lambda x: K.max(x, axis=-2), name='left_branch')(left_branch)

    right_input_emb = embedding(right_input)
    right_branch = Dropout(input_dropout)(right_input_emb)
    right_branch = TimeDistributed(
        Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_branch)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Lambda(lambda x: K.max(x, axis=-2), name='right_branch')(right_branch)

    left_context_emb = embedding(left_input)
    left_context = Dropout(input_dropout)(left_context_emb)
    left_context = TimeDistributed(
        Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_context)
    left_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_context)
    left_context = Lambda(lambda x: K.max(x, axis=-2), name='left_context')(left_context)

    right_context_emb = embedding(right_context_input)
    right_context = Dropout(input_dropout)(right_context_emb)
    right_context = TimeDistributed(
        Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_context)
    right_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_context)
    right_context = Lambda(lambda x: K.max(x, axis=-2), name='right_context')(right_context)

    concat = concatenate([left_branch, right_branch, type_input, left_context, right_context, time_diff_input])
    hidden_lstm = TimeDistributed(Dense(256, activation='relu'))(concat)
    hidden_lstm = Dropout(0.3)(hidden_lstm)
    hidden_lstm = concatenate([hidden_lstm, type_input, time_diff_input])
    aux_softmax = TimeDistributed(Dense(nb_classes, activation='softmax'), name='aux_out')(hidden_lstm)

    left_encoder = TimeDistributed(Dense(64, activation='tanh'))(left_context)
    right_encoder = TimeDistributed(Dense(64, activation='tanh'))(right_context)

    encoder = concatenate([left_encoder, right_encoder, type_input, aux_softmax])
    # encoder = concatenate([left_encoder, right_encoder, type_input, hidden_lstm])

    # encoder = concatenate([left_encoder, right_encoder, hidden_lstm])
    # encoder = Dropout(0.3)(encoder)
    # encoder = concatenate([encoder, type_input])

    NTM_F = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
                return_sequences=True, batch_size=batch_size, stateful=False, activation='linear')

    NTM_B = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
                return_sequences=True, batch_size=batch_size, stateful=False, activation='linear', go_backwards=True)

    # ntm_layer = Bidirectional(ntm, merge_mode='ave')(encoder)

    ntm_forward = NTM_F(encoder)
    ntm_backward = NTM_B(encoder)

    # # make a layer to reverse output
    # Reverse = Lambda(lambda x: reverse(x, axes=-2), output_shape=(batch_size, ntm_output_dim))
    # ntm_backward = Reverse(ntm_backward)

    ntm_layer = average([ntm_forward, ntm_backward])
    # ntm_layer = Dropout(0.3)(ntm_layer)

    hidden_ntm = TimeDistributed(Dense(256, activation='relu'))(ntm_layer)
    hidden_ntm = Dropout(0.3)(hidden_ntm)
    # concat_decoder = concatenate([hidden_ntm, hidden_lstm])
    # decoder = TimeDistributed(Dense(128, activation='relu'))(concat_decoder)
    # outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(decoder)
    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(hidden_ntm)

    model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input], outputs=[outlayer, aux_softmax])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'], loss_weights=[1., 0.5])

    return model


def get_ntm_model5_4(batch_size=5, group_size=None, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=13, input_dropout=0.3, embedding_matrix=None, **kwargs):
    """Feed hidden layer to ntm"""

    left_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_input = Input(batch_shape=(batch_size, group_size, max_len))
    left_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    type_input = Input(batch_shape=(batch_size, group_size, 1))
    time_diff_input = Input(batch_shape=(batch_size, group_size, 3))

    # Shared embedding layer
    embedding = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)

    # group_size here is the # of pairs in a group
    left_input_emb = embedding(left_input)
    left_branch = Dropout(input_dropout)(left_input_emb)
    left_branch = TimeDistributed(
        Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_branch)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Lambda(lambda x: K.max(x, axis=-2), name='left_branch')(left_branch)

    right_input_emb = embedding(right_input)
    right_branch = Dropout(input_dropout)(right_input_emb)
    right_branch = TimeDistributed(
        Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_branch)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Lambda(lambda x: K.max(x, axis=-2), name='right_branch')(right_branch)

    left_context_emb = embedding(left_input)
    left_context = Dropout(input_dropout)(left_context_emb)
    left_context = TimeDistributed(
        Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_context)
    left_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_context)
    left_context = Lambda(lambda x: K.max(x, axis=-2), name='left_context')(left_context)

    right_context_emb = embedding(right_context_input)
    right_context = Dropout(input_dropout)(right_context_emb)
    right_context = TimeDistributed(
        Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_context)
    right_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_context)
    right_context = Lambda(lambda x: K.max(x, axis=-2), name='right_context')(right_context)

    concat = concatenate([left_branch, right_branch, type_input, left_context, right_context, time_diff_input])
    hidden_lstm = TimeDistributed(Dense(256, activation='relu'))(concat)
    hidden_lstm = Dropout(0.3)(hidden_lstm)
    hidden_lstm = concatenate([hidden_lstm, type_input, time_diff_input])
    aux_softmax = TimeDistributed(Dense(nb_classes, activation='softmax'), name='aux_out')(hidden_lstm)

    left_encoder = TimeDistributed(Dense(64, activation='tanh'))(left_context)
    right_encoder = TimeDistributed(Dense(64, activation='tanh'))(right_context)

    encoder = concatenate([left_encoder, right_encoder, type_input, hidden_lstm])

    NTM_F = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=False, key_range=128,
                return_sequences=True, batch_size=batch_size, stateful=False, activation='relu')

    NTM_B = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=False, key_range=128,
                return_sequences=True, batch_size=batch_size, stateful=False, activation='relu', go_backwards=True)

    # ntm_layer = Bidirectional(ntm, merge_mode='ave')(encoder)

    ntm_forward = NTM_F(encoder)
    ntm_backward = NTM_B(encoder)

    ntm_layer = average([ntm_forward, ntm_backward])

    hidden_ntm = TimeDistributed(Dense(512, activation='relu'))(ntm_layer)
    hidden_ntm = Dropout(0.3)(hidden_ntm)
    # concat_decoder = concatenate([hidden_ntm, hidden_lstm])
    decoder = TimeDistributed(Dense(128, activation='sigmoid'))(hidden_ntm)
    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(decoder)
    # outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(hidden_ntm)

    model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input], outputs=[outlayer, aux_softmax])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'], loss_weights=[1., 0.5])

    return model


def get_lstm_model4_1(batch_size=100, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                           read_heads=1, write_heads=1, nb_classes=13,
                           input_dropout=0.5, **kwargs):
    """LSTM context layer, no auxiliary output"""

    left_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_branch = Dropout(input_dropout)(left_input)
    left_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), merge_mode='sum'))(left_branch)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Reshape((batch_size, -1))(left_branch)  # (1, batch_size, 128)

    right_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_branch = Dropout(input_dropout)(right_input)
    right_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), merge_mode='sum'))(right_branch)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Reshape((batch_size, -1))(right_branch)  # (1, batch_size, 128)

    left_context_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_context = Dropout(input_dropout)(left_context_input)
    left_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), merge_mode='sum'))(left_context)
    left_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_context)
    left_context = Reshape((batch_size, -1))(left_context)  # (1, batch_size, 128)

    right_context_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_context = Dropout(input_dropout)(right_context_input)
    right_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), merge_mode='sum'))(right_context)
    right_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_context)
    right_context = Reshape((batch_size, -1))(right_context)  # (1, batch_size, 128)

    type_input = Input(batch_shape=(1, batch_size, 1))
    # type_input = Reshape((batch_size, 1))(type_input)
    pair_type = TimeDistributed(Dense(2))(type_input)

    concat = concatenate([left_branch, right_branch, pair_type, left_context, right_context])
    hidden_lstm = TimeDistributed(Dense(256, activation='tanh'))(concat)
    hidden_lstm = Dropout(0.5)(hidden_lstm)
    hidden_lstm2 = TimeDistributed(Dense(128, activation='tanh'))(hidden_lstm)

    left_encoder = TimeDistributed(Dense(64, activation='tanh'))(left_branch)
    right_encoder = TimeDistributed(Dense(64, activation='tanh'))(right_branch)

    encoder = concatenate([left_encoder, right_encoder, hidden_lstm2, pair_type])

    context_layer = Bidirectional(LSTM(128, return_sequences=True, stateful=True), merge_mode='ave')(encoder)
    context_layer = Dropout(0.5)(context_layer)
    hidden_context = TimeDistributed(Dense(64, activation='tanh'))(context_layer)

    hidden = concatenate([hidden_lstm2, hidden_context, pair_type])

    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name="main_out")(hidden)
    model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input], outputs=[outlayer])

    # compile the final model
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0002), metrics=['accuracy'])

    return model


def get_pre_ntm_model(group_size=None, nb_classes=13, input_dropout=0.3, max_len=16, embedding_matrix=None, **kwargs):
    """feed softmax to ntm"""

    # Shared embedding layer
    embedding = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)

    # left_input = Input(shape=(group_size, max_len, EMBEDDING_DIM))  # group_size here is the # of pairs in a group
    left_input = Input(shape=(group_size, max_len))
    left_input_emb = embedding(left_input)
    left_branch = Dropout(input_dropout)(left_input_emb)
    # left_branch = Dropout(input_dropout)(left_input)
    left_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_branch)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Lambda(lambda x: K.max(x, axis=-2), name='left_branch')(left_branch)

    # right_input = Input(shape=(group_size, max_len, EMBEDDING_DIM))
    right_input = Input(shape=(group_size, max_len))
    right_input_emb = embedding(right_input)
    right_branch = Dropout(input_dropout)(right_input_emb)
    # right_branch = Dropout(input_dropout)(right_input)
    right_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_branch)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Lambda(lambda x: K.max(x, axis=-2), name='right_branch')(right_branch)

    # left_context_input = Input(shape=(group_size, max_len, EMBEDDING_DIM))
    left_context_input = Input(shape=(group_size, max_len))
    left_context_input_emb = embedding(left_context_input)
    left_context = Dropout(input_dropout)(left_context_input_emb)
    # left_context = Dropout(input_dropout)(left_context_input)
    left_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_context)
    left_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_context)
    left_context = Lambda(lambda x: K.max(x, axis=-2), name='left_context')(left_context)

    # right_context_input = Input(shape=(group_size, max_len, EMBEDDING_DIM))
    right_context_input = Input(shape=(group_size, max_len))
    right_context_input_emb = embedding(right_context_input)
    right_context = Dropout(input_dropout)(right_context_input_emb)
    # right_context = Dropout(input_dropout)(right_context_input)
    right_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_context)
    right_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_context)
    right_context = Lambda(lambda x: K.max(x, axis=-2), name='right_context')(right_context)

    type_input = Input(shape=(group_size, 1))
    time_diff_input = Input(shape=(group_size, 3))

    concat = concatenate([left_branch, right_branch, type_input, left_context, right_context, time_diff_input])
    # hidden_lstm = TimeDistributed(Dense(256, activation='relu'))(concat)
    hidden_lstm = Dense(512, activation='relu')(concat)
    hidden_lstm = Dropout(0.3)(hidden_lstm)
    hidden_lstm = concatenate([hidden_lstm, type_input, time_diff_input], name='hidden_lstm')
    # outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='aux_out')(hidden_lstm)
    hidden_lstm2 = Dense(256, activation='relu')(hidden_lstm)
    hidden_lstm2 = Dropout(0.3)(hidden_lstm2)
    outlayer = Dense(nb_classes, activation='softmax', name='aux_out')(hidden_lstm2)

    model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input], outputs=[outlayer])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

    return model


def load_pretrained_model(trainable=False):
    """load the pre-trained model (independant pair classifier)"""

    pre_ntm_model = load_model(os.path.join(os.environ["TEA_PATH"], 'model_destination/12-11-nontm/all/final_model.h5'))
    # pre_ntm_model = get_pre_ntm_model(group_size=None, nb_classes=len(LABELS), input_dropout=0.3, max_len=MAX_LEN,
    #                                                     embedding_matrix=None)
    # pre_ntm_model.load_weights(os.path.join(os.environ["TEA_PATH"], 'model_destination/12-7-nontm-full/all/best_weights.h5'))
    if not trainable:
        for layer in pre_ntm_model.layers:
            if hasattr(layer, 'layer'):  # layers embedded in wrappers
                layer.layer.trainable = False
            layer.trainable = False
        pre_ntm_model.trainable = False
    pre_ntm_model.compile(loss='mse', optimizer='rmsprop')

    return pre_ntm_model


def get_ntm_model6_baseline(batch_size=32, group_size=10, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=6, input_dropout=0.3, has_auxiliary=False, **kwargs):
    """feed softmax to ntm"""

    left_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_input = Input(batch_shape=(batch_size, group_size, max_len))
    left_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    type_input = Input(batch_shape=(batch_size, group_size, 1))
    time_diff_input = Input(batch_shape=(batch_size, group_size, 3))

    input_list = [left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input]

    pre_ntm_model = load_pretrained_model(trainable=False)
    pre_ntm_model.name = 'pre_ntm'
    print("\nPre-NTM model loaded successfully!\n")
    aux_softmax = pre_ntm_model(input_list)

    encoder_model = load_pretrained_model(trainable=False)
    encoder_model.name = 'encoder'
    # encoder_model.layers[19].outbound_nodes = []
    # encoder_model.layers[20].outbound_nodes = []
    encoder_model.get_layer('left_context').outbound_nodes = []
    encoder_model.get_layer('right_context').outbound_nodes = []
    left_encoder_model = Model(encoder_model.input, encoder_model.get_layer('left_context').output)
    right_encoder_model = Model(encoder_model.input, encoder_model.get_layer('right_context').output)

    left_encoder = left_encoder_model(input_list)
    right_encoder = right_encoder_model(input_list)

    encoder = concatenate([left_encoder, right_encoder, aux_softmax, type_input, time_diff_input])

    hidden_ntm = TimeDistributed(Dense(256, activation='relu'))(encoder)
    hidden_ntm = Dropout(0.5)(hidden_ntm)

    # concat_decoder = concatenate([hidden_ntm, aux_softmax, type_input, time_diff_input])
    # decoder = TimeDistributed(Dense(128, activation='relu'))(concat_decoder)
    decoder = TimeDistributed(Dense(128, activation='tanh'))(hidden_ntm )
    decoder = Dropout(0.3)(decoder)
    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(decoder)

    model = Model(inputs=input_list, outputs=[outlayer, aux_softmax])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'], loss_weights=[1., 0.5])

    return model


def get_ntm_model6(batch_size=32, group_size=10, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=6, input_dropout=0.3, has_auxiliary=False, **kwargs):
    """feed softmax to ntm"""

    left_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_input = Input(batch_shape=(batch_size, group_size, max_len))
    left_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    type_input = Input(batch_shape=(batch_size, group_size, 1))
    time_diff_input = Input(batch_shape=(batch_size, group_size, 3))

    input_list = [left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input]

    pre_ntm_model = load_pretrained_model(trainable=False)
    pre_ntm_model.name = 'pre_ntm'
    print("\nPre-NTM model loaded successfully!\n")
    aux_softmax = pre_ntm_model(input_list)

    encoder_model = load_pretrained_model(trainable=False)
    encoder_model.name = 'encoder'
    # encoder_model.layers[19].outbound_nodes = []
    # encoder_model.layers[20].outbound_nodes = []
    encoder_model.get_layer('left_context').outbound_nodes = []
    encoder_model.get_layer('right_context').outbound_nodes = []
    left_encoder_model = Model(encoder_model.input, encoder_model.get_layer('left_context').output)
    right_encoder_model = Model(encoder_model.input, encoder_model.get_layer('right_context').output)

    left_encoder = left_encoder_model(input_list)
    right_encoder = right_encoder_model(input_list)

    encoder = concatenate([left_encoder, right_encoder, aux_softmax, type_input, time_diff_input])

    NTM_F = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=False, key_range=256,
                return_sequences=True, batch_size=batch_size, stateful=False, activation='relu')
    NTM_B = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=False, key_range=256,
                return_sequences=True, batch_size=batch_size, stateful=False, activation='relu', go_backwards=True)

    # ntm_layer = Bidirectional(ntm, merge_mode='ave')(encoder)

    ntm_forward = NTM_F(encoder)
    ntm_backward = NTM_B(encoder)


    ntm_layer = average([ntm_forward, ntm_backward])
    # ntm_layer = Dropout(0.3)(ntm_layer)

    hidden_ntm = TimeDistributed(Dense(256, activation='relu'))(ntm_layer)
    hidden_ntm = Dropout(0.5)(hidden_ntm)

    # concat_decoder = concatenate([hidden_ntm, aux_softmax, type_input, time_diff_input])
    # decoder = TimeDistributed(Dense(128, activation='relu'))(concat_decoder)
    decoder = TimeDistributed(Dense(128, activation='tanh'))(hidden_ntm )
    decoder = Dropout(0.3)(decoder)
    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(decoder)

    model = Model(inputs=input_list, outputs=[outlayer])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

    return model


def get_ntm_model6_2(batch_size=32, group_size=10, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=6, input_dropout=0.3, has_auxiliary=False, **kwargs):
    """feed softmax to ntm"""

    left_input = Input(batch_shape=(batch_size, group_size, max_len, EMBEDDING_DIM))
    right_input = Input(batch_shape=(batch_size, group_size, max_len, EMBEDDING_DIM))
    left_context_input = Input(batch_shape=(batch_size, group_size, max_len, EMBEDDING_DIM))
    right_context_input = Input(batch_shape=(batch_size, group_size, max_len, EMBEDDING_DIM))
    type_input = Input(batch_shape=(batch_size, group_size, 1))
    time_diff_input = Input(batch_shape=(batch_size, group_size, 3))

    input_list = [left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input]

    pre_ntm_model = load_pretrained_model()
    pre_ntm_model.name = 'pre_ntm'
    print("\nPre-NTM model loaded successfully!\n")
    aux_softmax = pre_ntm_model(input_list)

    encoder_model = load_pretrained_model(trainable=True)
    encoder_model.layers[19].outbound_nodes = []
    encoder_model.layers[20].outbound_nodes = []
    left_encoder_model = Model(encoder_model.input, encoder_model.layers[19].output)
    right_encoder_model = Model(encoder_model.input, encoder_model.layers[20].output)

    left_encoder = left_encoder_model(input_list)
    right_encoder = right_encoder_model(input_list)

    encoder = concatenate([left_encoder, right_encoder, aux_softmax, type_input, time_diff_input])
    print("\nPre-NTM model loaded successfully!\n")

    # NTM_F1 = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
    #             read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
    #             return_sequences=True, batch_size=batch_size, stateful=False, activation='linear')
    # NTM_F1.build((batch_size, group_size, 266))
    NTM_F2 = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                 read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
                 return_sequences=True, batch_size=batch_size, stateful=False, activation='linear')
    NTM_F2.build((batch_size, group_size, 128))

    # NTM_B1 = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
    #             read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
    #             return_sequences=True, batch_size=batch_size, stateful=False, activation='linear', go_backwards=True)
    # NTM_B1.build((batch_size, group_size, 266))
    NTM_B2 = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                 read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
                 return_sequences=True, batch_size=batch_size, stateful=False, activation='linear', go_backwards=True)
    NTM_B2.build((batch_size, group_size, 128))

    # ntm_layer = Bidirectional(ntm, merge_mode='ave')(encoder)

    ntm_forward = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
                return_sequences=True, batch_size=batch_size, stateful=False, activation='linear')(encoder)
    ntm_backward = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                 read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
                 return_sequences=True, batch_size=batch_size, stateful=False, activation='linear', go_backwards=True)(encoder)
    ntm_layer = average([ntm_forward, ntm_backward])

    ntm_forward2 = NTM_F2(ntm_layer)
    ntm_backward2 = NTM_B2(ntm_layer)
    rnn2 = average([ntm_forward2, ntm_backward2])
    # rnn2 = Bidirectional(LSTM(64, return_sequences=True, activation='relu'), merge_mode='sum')(ntm_layer)

    hidden_ntm = TimeDistributed(Dense(128, activation='relu'))(rnn2)
    hidden_ntm = Dropout(0.3)(hidden_ntm)

    # concat_decoder = concatenate([hidden_ntm, aux_softmax, type_input, time_diff_input])
    # decoder = TimeDistributed(Dense(128, activation='relu'))(concat_decoder)
    # decoder = TimeDistributed(Dense(128, activation='relu'))(hidden_ntm)
    # decoder = Dropout(0.3)(decoder)
    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(hidden_ntm)

    model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input], outputs=[outlayer, aux_softmax])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'], loss_weights=[1., 0.5])

    return model


def get_ntm_model6_3(batch_size=32, group_size=10, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=6, input_dropout=0.3, has_auxiliary=False, **kwargs):
    """feed hidden layer after lstm to ntm"""

    left_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_input = Input(batch_shape=(batch_size, group_size, max_len))
    left_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    type_input = Input(batch_shape=(batch_size, group_size, 1))
    time_diff_input = Input(batch_shape=(batch_size, group_size, 3))

    input_list = [left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input]

    prentm_model = load_pretrained_model(trainable=False)
    prentm_model.name = 'pre_ntm'
    print("\nPre-NTM model hidden_lstm loaded successfully!\n")
    prentm_model.layers = prentm_model.layers[0:-3]
    hidden_lstm = prentm_model(input_list)

    encoder_model = load_pretrained_model(trainable=False)
    encoder_model.name = 'encoder'
    encoder_model.get_layer('left_context').outbound_nodes = []
    encoder_model.get_layer('right_context').outbound_nodes = []
    left_encoder_model = Model(encoder_model.input, encoder_model.get_layer('left_context').output)
    right_encoder_model = Model(encoder_model.input, encoder_model.get_layer('right_context').output)
    print("\nPre-NTM model encoder loaded successfully!\n")
    left_encoder = left_encoder_model(input_list)
    right_encoder = right_encoder_model(input_list)

    encoder = concatenate([left_encoder, right_encoder, hidden_lstm, type_input, time_diff_input])

    NTM_F = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=False, key_range=256,
                return_sequences=True, batch_size=batch_size, stateful=False, activation='relu')
    NTM_B = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=False, key_range=256,
                return_sequences=True, batch_size=batch_size, stateful=False, activation='relu', go_backwards=True)

    ntm_forward = NTM_F(encoder)
    ntm_backward = NTM_B(encoder)

    ntm_layer = average([ntm_forward, ntm_backward])
    ntm_layer = Dropout(0.3)(ntm_layer)

    hidden_ntm = TimeDistributed(Dense(512, activation='tanh'))(ntm_layer)
    hidden_ntm = Dropout(0.3)(hidden_ntm)
    decoder = TimeDistributed(Dense(128, activation='sigmoid'))(hidden_ntm )
    decoder = Dropout(0.3)(decoder)

    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(decoder)

    model = Model(inputs=input_list, outputs=[outlayer])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

    return model

def get_pre_ntm_model2(group_size=None, nb_classes=13, input_dropout=0.3, max_len=16, embedding_matrix=None, **kwargs):
    """feed softmax to ntm"""

    # Shared embedding layer
    embedding = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)

    left_input = Input(shape=(group_size, max_len))
    # left_input_emb = embedding(left_input)
    # left_branch = Dropout(input_dropout)(left_input_emb)
    # left_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_branch)
    # left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    # left_branch = Lambda(lambda x: K.max(x, axis=-2), name='left_branch')(left_branch)
    #
    right_input = Input(shape=(group_size, max_len))
    # right_input_emb = embedding(right_input)
    # right_branch = Dropout(input_dropout)(right_input_emb)
    # right_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_branch)
    # right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    # right_branch = Lambda(lambda x: K.max(x, axis=-2), name='right_branch')(right_branch)

    left_context_input = Input(shape=(group_size, max_len))
    left_context_input_emb = embedding(left_context_input)
    left_context = Dropout(input_dropout)(left_context_input_emb)
    left_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_context)
    left_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_context)
    left_context = Lambda(lambda x: K.max(x, axis=-2), name='left_context')(left_context)

    right_context_input = Input(shape=(group_size, max_len))
    right_context_input_emb = embedding(right_context_input)
    right_context = Dropout(input_dropout)(right_context_input_emb)
    right_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_context)
    right_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_context)
    right_context = Lambda(lambda x: K.max(x, axis=-2), name='right_context')(right_context)

    type_input = Input(shape=(group_size, 1))
    time_diff_input = Input(shape=(group_size, 3))

    concat = concatenate([type_input, left_context, right_context, time_diff_input])
    # hidden_lstm = TimeDistributed(Dense(256, activation='relu'))(concat)
    hidden_lstm = Dense(512, activation='relu')(concat)
    hidden_lstm = Dropout(0.3)(hidden_lstm)
    hidden_lstm = concatenate([hidden_lstm, type_input, time_diff_input], name='hidden_lstm')
    # outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='aux_out')(hidden_lstm)
    hidden_lstm2 = Dense(256, activation='relu')(hidden_lstm)
    hidden_lstm2 = Dropout(0.3)(hidden_lstm2)
    outlayer = Dense(nb_classes, activation='softmax', name='aux_out')(hidden_lstm2)

    model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input], outputs=[outlayer])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

    return model