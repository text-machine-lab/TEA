from __future__ import print_function
import os
import sys
import numpy as np
np.random.seed(1337)
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Reshape, LSTM, Dense, concatenate, average, MaxPooling1D, TimeDistributed, Flatten, Lambda, Input, Dropout, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.optimizers import Adam
from word2vec import load_word2vec_binary
from collections import deque

from ntm import NeuralTuringMachine as NTM, StatefulController
from ntm import controller_input_output_shape as controller_shape

# LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IBEFORE", "IAFTER", "IS_INCLUDED", "INCLUDES",
#           "DURING", "BEGINS", "BEGUN_BY", "ENDS", "ENDED_BY", "None"]
LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IS_INCLUDED", "INCLUDES", "None"] # TimeBank Dense labels
EMBEDDING_DIM = 300
DENSE_LABELS = True
MAX_LEN = 15  # max # of words on each branch of path


def get_lstm_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=300):
    controller = StatefulController(units=controller_output_dim,  # 2x if using Bidrectional
                                    kernel_initializer='random_normal',
                                    bias_initializer='random_normal',
                                    activation='linear',
                                    stateful=True,  # must be true, because in controller every step is a batch?
                                    return_state=True,  # actually it does not matter, we pass states in
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


def get_simple_ntm(batch_size=100, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15, read_heads=1, write_heads=1, nb_classes=13,
                  input_dropout=0.5):

    controller_input_dim, controller_output_dim = controller_shape(EMBEDDING_DIM, ntm_output_dim, m_depth,
                                                                   n_slots, shift_range, read_heads, write_heads)

    # we feed in controller (# documents, # pairs, data_dim)
    # so max_steps here is # pairs


    controller = get_lstm_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=batch_size)
    # controller = get_dense_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=batch_size)

    # model.name = "NTM_-_" + controller.name

    ntm = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
              read_heads=read_heads, write_heads=write_heads, controller_model=controller,
              return_sequences=True, input_shape=(max_len, EMBEDDING_DIM),
              activation='relu', batch_size=1)


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


def get_ntm_model(batch_size=100, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                  read_heads=1, write_heads=1, nb_classes=13,
                  input_dropout=0.5):
    left_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_branch = TimeDistributed(LSTM(128, return_sequences=True))(left_input)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Reshape((batch_size, -1))(left_branch)  # (1, batch_size, 128)

    right_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_branch = TimeDistributed(LSTM(128, return_sequences=True))(right_input)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Reshape((batch_size, -1))(right_branch)  # (1, batch_size, 128)

    encoder = concatenate([left_branch, right_branch])
    encoder = Dense(256)(encoder)

    controller_input_dim, controller_output_dim = controller_shape(256, ntm_output_dim, m_depth,
                                                                   n_slots, shift_range, read_heads, write_heads)

    # we feed in controller (# documents, # pairs, data_dim)
    # so max_steps here is # pairs
    controller = get_lstm_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=batch_size)
    # controller = get_dense_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=batch_size)

    # model.name = "NTM_-_" + controller.name

    ntm = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
              read_heads=read_heads, write_heads=write_heads, controller_model=controller,
              return_sequences=True, input_shape=(batch_size, 256),
              activation='relu', batch_size=1)

    ntm_layer = ntm(encoder)
    ntm_layer = Dropout(0.5)(ntm_layer)
    hidden = TimeDistributed(Dense(64, activation='relu'))(ntm_layer)
    hidden = Dropout(0.2)(hidden)
    out_layer = TimeDistributed(Dense(nb_classes, activation='softmax'))(hidden)
    model = Model(inputs=[left_input, right_input], outputs=[out_layer])

    # compile the final model
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def get_ntm_model2(batch_size=100, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                      read_heads=1, write_heads=1, nb_classes=13,
                      input_dropout=0.5, **kwargs):
    """hidden layer after lstm connects to out layer"""

    from keras.backend import reverse

    left_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True), merge_mode='sum'))(left_input)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Reshape((batch_size, -1))(left_branch)  # (1, batch_size, 128)

    right_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True), merge_mode='sum'))(right_input)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Reshape((batch_size, -1))(right_branch)  # (1, batch_size, 128)

    type_input = Input(batch_shape=(1, batch_size, 1))
    # type_input = Reshape((batch_size, 1))(type_input)
    pair_type = TimeDistributed(Dense(2))(type_input)

    concat = concatenate([left_branch, right_branch, pair_type])
    hidden_lstm_1 = TimeDistributed(Dense(128, activation='relu'))(concat)
    hidden_lstm_1 = Dropout(0.5)(hidden_lstm_1)
    aux_out_layer = TimeDistributed(Dense(2, activation='softmax'), name='aux_out')(hidden_lstm_1)
    hidden_lstm_2 = Dense(128, activation='relu')(hidden_lstm_1)

    left_encoder = TimeDistributed(Dense(64, activation='relu'))(left_branch)
    right_encoder = TimeDistributed(Dense(64, activation='relu'))(right_branch)

    encoder = concatenate([left_encoder, hidden_lstm_1, right_encoder])  # (batch, 256)

    controller_input_dim, controller_output_dim = controller_shape(256, ntm_output_dim, m_depth,
                                                                   n_slots, shift_range, read_heads, write_heads)

    # we feed in controller (# documents, # pairs, data_dim)
    # so max_steps here is # pairs
    controller = get_lstm_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=batch_size)

    # model.name = "NTM_-_" + controller.name

    NTM_F = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
              read_heads=read_heads, write_heads=write_heads, controller_model=controller,
              return_sequences=True, batch_input_shape=(1, batch_size, 256), stateful=True,
              activation='relu')
    NTM_B = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                      read_heads=read_heads, write_heads=write_heads, controller_model=controller,
                      return_sequences=True, batch_input_shape=(1, batch_size, 256), stateful=True,
                      activation='relu', go_backwards=True)

    # ntm_layer = Bidirectional(ntm, merge_mode='ave')(encoder)

    ntm_forward = NTM_F(encoder)
    ntm_backward = NTM_B(encoder)

    # make a layer to reverse output
    Reverse = Lambda(lambda x: reverse(x, axes=-2), output_shape=(batch_size, ntm_output_dim))
    ntm_backward = Reverse(ntm_backward)

    ntm_layer = average([ntm_forward, ntm_backward])
    ntm_layer = Dropout(0.5)(ntm_layer)

    hidden_ntm = TimeDistributed(Dense(64, activation='relu'))(ntm_layer)

    hidden = concatenate([hidden_lstm_2, hidden_ntm], axis=-1)

    out_layer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(hidden)
    model = Model(inputs=[left_input, right_input, type_input], outputs=[out_layer, aux_out_layer])

    # compile the final model
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], loss_weights=[1., 1.])

    return model

def get_ntm_model2_1(batch_size=100, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=13, input_dropout=0.5, has_auxiliary=False):
    """sequential, no early output"""
    from keras.backend import reverse

    left_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True), merge_mode='sum'))(left_input)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Reshape((batch_size, -1))(left_branch)  # (1, batch_size, 128)

    right_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True), merge_mode='sum'))(right_input)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Reshape((batch_size, -1))(right_branch)  # (1, batch_size, 128)

    type_input = Input(batch_shape=(1, batch_size, 1))
    # type_input = Reshape((batch_size, 1))(type_input)
    pair_type = TimeDistributed(Dense(2))(type_input)

    concat = concatenate([left_branch, right_branch, pair_type])
    concat = Dropout(0.5)(concat)
    hidden_lstm = TimeDistributed(Dense(128, activation='relu'))(concat)

    controller_input_dim, controller_output_dim = controller_shape(128, ntm_output_dim, m_depth,
                                                                   n_slots, shift_range, read_heads, write_heads)

    # we feed in controller (# documents, # pairs, data_dim)
    # so max_steps here is # pairs
    controller = get_lstm_controller(controller_output_dim, controller_input_dim, batch_size=1, max_steps=batch_size)

    # model.name = "NTM_-_" + controller.name

    NTM_F = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_model=controller,
                return_sequences=True, batch_input_shape=(1, batch_size, 128), stateful=True,
                activation='relu')
    NTM_B = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_model=controller,
                return_sequences=True, batch_input_shape=(1, batch_size, 128), stateful=True,
                activation='relu', go_backwards=True)

    # ntm_layer = Bidirectional(ntm, merge_mode='ave')(encoder)

    ntm_forward = NTM_F(hidden_lstm)
    ntm_backward = NTM_B(hidden_lstm)

    # make a layer to reverse output
    Reverse = Lambda(lambda x: reverse(x, axes=-2), output_shape=(batch_size, ntm_output_dim))
    ntm_backward = Reverse(ntm_backward)

    ntm_layer = average([ntm_forward, ntm_backward])
    ntm_layer = Dropout(0.5)(ntm_layer)

    hidden_ntm = TimeDistributed(Dense(128, activation='relu'))(ntm_layer)
    hidden_ntm = Dropout(0.5)(hidden_ntm)

    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(hidden_ntm)

    if has_auxiliary:
        auxiliary_outlayer = TimeDistributed(Dense(2, activation='softmax'), name='aux_out')(hidden_ntm)
        model = Model(inputs=[left_input, right_input, type_input], outputs=[outlayer, auxiliary_outlayer])
        # model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], loss_weights=[1., 1.])
    else:
        model = Model(inputs=[left_input, right_input, type_input], outputs=[outlayer])
        # model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def get_ntm_model3(batch_size=100, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                           read_heads=1, write_heads=1, nb_classes=13,
                           input_dropout=0.5, **kwargs):


    left_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True), merge_mode='sum'))(left_input)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Reshape((batch_size, -1))(left_branch)  # (1, batch_size, 128)

    right_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True), merge_mode='sum'))(right_input)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Reshape((batch_size, -1))(right_branch)  # (1, batch_size, 128)

    type_input = Input(batch_shape=(1, batch_size, 1))
    # type_input = Reshape((batch_size, 1))(type_input)
    pair_type = TimeDistributed(Dense(2))(type_input)

    concat = concatenate([left_branch, right_branch, pair_type])
    hidden_lstm = TimeDistributed(Dense(128, activation='tanh'))(concat)
    hidden_lstm = Dropout(0.5)(hidden_lstm)

    left_encoder = TimeDistributed(Dense(64, activation='relu'))(left_branch)
    right_encoder = TimeDistributed(Dense(64, activation='relu'))(right_branch)

    encoder = concatenate([left_encoder, hidden_lstm, right_encoder])

    context_layer = Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='ave')(encoder)
    context_layer = Dropout(0.5)(context_layer)
    hidden_context = TimeDistributed(Dense(64, activation='tanh'))(context_layer)

    hidden = concatenate([hidden_lstm, hidden_context])

    out_layer = TimeDistributed(Dense(nb_classes, activation='softmax'))(hidden)
    model = Model(inputs=[left_input, right_input, type_input], outputs=[out_layer])

    # compile the final model
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def get_ntm_model3_1(batch_size=100, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                           read_heads=1, write_heads=1, nb_classes=13,
                           input_dropout=0.5, **kwargs):
    """With auxiliary output"""

    left_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    left_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True), merge_mode='sum'))(left_input)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Reshape((batch_size, -1))(left_branch)  # (1, batch_size, 128)

    right_input = Input(batch_shape=(1, batch_size, max_len, EMBEDDING_DIM))
    right_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True), merge_mode='sum'))(right_input)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Reshape((batch_size, -1))(right_branch)  # (1, batch_size, 128)

    type_input = Input(batch_shape=(1, batch_size, 1))
    # type_input = Reshape((batch_size, 1))(type_input)
    pair_type = TimeDistributed(Dense(2))(type_input)

    concat = concatenate([left_branch, right_branch, pair_type])
    hidden_lstm = TimeDistributed(Dense(128, activation='tanh'))(concat)
    hidden_lstm = Dropout(0.5)(hidden_lstm)
    aux_outlayer = Dense(2, activation='softmax', name="aux_out")(hidden_lstm)

    left_encoder = TimeDistributed(Dense(64, activation='relu'))(left_branch)
    right_encoder = TimeDistributed(Dense(64, activation='relu'))(right_branch)

    encoder = concatenate([left_encoder, hidden_lstm, right_encoder])

    context_layer = Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='ave')(encoder)
    context_layer = Dropout(0.5)(context_layer)
    hidden_context = TimeDistributed(Dense(64, activation='tanh'))(context_layer)

    hidden = concatenate([hidden_lstm, hidden_context, aux_outlayer])

    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name="main_out")(hidden)
    model = Model(inputs=[left_input, right_input, type_input], outputs=[outlayer, aux_outlayer])

    # compile the final model
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], loss_weights=[1., 2.])

    return model