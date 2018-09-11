from __future__ import print_function
import os
import sys
import numpy as np
# np.random.seed(1337)
from keras.models import Model, Sequential, load_model
from keras.regularizers import l1_l2, l2
from keras.layers import Embedding, Reshape, LSTM, Dense, concatenate, average, add, MaxPooling1D, TimeDistributed, Flatten, Lambda, Input, Dropout, Bidirectional
from keras.optimizers import Adam, RMSprop, SGD
from keras.constraints import max_norm
import tensorflow as tf

# from ntm2 import NeuralTuringMachine as NTM
# from ntm2 import SingleKeyNTM as NTM
from ntm2 import SimpleNTM as NTM
import keras.backend as K

EMBEDDING_DIM = 300
DENSE_LABELS = True
MAX_LEN = 16  # max # of words on each branch of path

if DENSE_LABELS:
    LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IS_INCLUDED", "INCLUDES", "None"] # TimeBank Dense labels
else:
    LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IBEFORE", "IAFTER", "IS_INCLUDED", "INCLUDES",
              "BEGINS", "BEGUN_BY", "ENDS", "ENDED_BY", "None"]

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
    hidden_lstm = Dense(1024, activation='relu', kernel_constraint=max_norm(10.))(concat)
    hidden_lstm = Dropout(0.5)(hidden_lstm)
    hidden_lstm = concatenate([hidden_lstm, type_input, time_diff_input], name='hidden_lstm')
    # outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='aux_out')(hidden_lstm)
    hidden_lstm2 = Dense(512, activation='relu', kernel_constraint=max_norm(5.))(hidden_lstm)
    hidden_lstm2 = Dropout(0.5)(hidden_lstm2)
    outlayer = Dense(nb_classes, activation='softmax', name='aux_out')(hidden_lstm2)

    model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input], outputs=[outlayer])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

    return model


def load_pretrained_model(trainable=False):
    """load the pre-trained model (independant pair classifier)"""

    if DENSE_LABELS:
        pre_ntm_model = load_model(os.path.join(os.environ["TEA_PATH"], 'model_destination/7-5-nontm/all/final_model.h5'))
        # pre_ntm_model = load_model(os.path.join(os.environ["TEA_PATH"], 'model_destination/t-test/nontm/4/all/final_model.h5'))

    else:
        pre_ntm_model = load_model(os.path.join(os.environ["TEA_PATH"], 'model_destination/1-8-nontm-tml/all/final_model.h5'))
    if not trainable:
        for layer in pre_ntm_model.layers:
            if hasattr(layer, 'layer'):  # layers embedded in wrappers
                layer.layer.trainable = False
            layer.trainable = False
        pre_ntm_model.trainable = False
    pre_ntm_model.compile(loss='mse', optimizer='rmsprop')

    return pre_ntm_model


def get_baseline(batch_size=32, group_size=10, max_len=15, nb_classes=6, **kwargs):
    """feed softmax to ntm"""

    left_input = Input(shape=(group_size, max_len))
    right_input = Input(shape=(group_size, max_len))
    left_context_input = Input(shape=(group_size, max_len))
    right_context_input = Input(shape=(group_size, max_len))
    type_input = Input(shape=(group_size, 1))
    time_diff_input = Input(shape=(group_size, 3))

    input_list = [left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input]

    prentm_model = load_pretrained_model(trainable=False)
    prentm_model.name = 'pre_ntm'
    config = prentm_model.get_config()
    weights = prentm_model.get_weights()[:-3]
    prentm_model.layers = prentm_model.layers[:-3]
    # need to change configuration for proper save/load operations
    config['layers'] = config['layers'][:-3]
    config['output_layers'] = [['hidden_lstm', 0, 0]]
    config['name'] = 'pre_ntm'
    prentm_model = prentm_model.from_config(config)
    prentm_model.set_weights(weights)
    for layer in prentm_model.layers:
        if hasattr(layer, 'layer'):  # layers embedded in wrappers
            layer.layer.trainable = False
        layer.trainable = False
    prentm_model.trainable = False
    print("\nPre-NTM model hidden_lstm loaded successfully!\n")

    hidden_lstm = prentm_model(input_list)

    encoder_model = load_pretrained_model(trainable=False)
    encoder_model.name = 'encoder'
    encoder_model.get_layer('left_context').outbound_nodes = []
    encoder_model.get_layer('right_context').outbound_nodes = []
    left_encoder_model = Model(encoder_model.input, encoder_model.get_layer('left_context').output)
    right_encoder_model = Model(encoder_model.input, encoder_model.get_layer('right_context').output)

    left_encoder = left_encoder_model(input_list)
    right_encoder = right_encoder_model(input_list)

    encoder = concatenate([left_encoder, right_encoder, hidden_lstm, type_input, time_diff_input])

    hidden_ntm = TimeDistributed(Dense(512, activation='relu'))(encoder)
    hidden_ntm = Dropout(0.3)(hidden_ntm)

    decoder = TimeDistributed(Dense(128, activation='tanh'))(hidden_ntm )
    decoder = Dropout(0.3)(decoder)
    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(decoder)

    model = Model(inputs=input_list, outputs=[outlayer])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

    return model


def get_ntm_softmaxfeed(batch_size=32, group_size=10, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=6, input_dropout=0.3, has_auxiliary=False, **kwargs):
    """feed softmax to ntm"""

    left_input = Input(shape=(group_size, max_len))
    right_input = Input(shape=(group_size, max_len))
    left_context_input = Input(shape=(group_size, max_len))
    right_context_input = Input(shape=(group_size, max_len))
    type_input = Input(shape=(group_size, 1))
    time_diff_input = Input(shape=(group_size, 3))

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


def get_ntm_hiddenfeed(batch_size=32, group_size=10, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=6, input_dropout=0.3, has_auxiliary=False, **kwargs):
    """feed hidden layer after lstm to ntm"""

    # left_input = Input(shape=(group_size, max_len))
    # right_input = Input(shape=(group_size, max_len))
    # left_context_input = Input(shape=(group_size, max_len))
    # right_context_input = Input(shape=(group_size, max_len))
    # type_input = Input(shape=(group_size, 1))
    # time_diff_input = Input(shape=(group_size, 3))

    left_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_input = Input(batch_shape=(batch_size, group_size, max_len))
    left_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    right_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    type_input = Input(batch_shape=(batch_size, group_size, 1))
    time_diff_input = Input(batch_shape=(batch_size, group_size, 3))

    input_list = [left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input]

    prentm_model = load_pretrained_model(trainable=False)
    prentm_model.name = 'pre_ntm'
    # config = prentm_model.get_config()
    # weights = prentm_model.get_weights()[:-3]
    prentm_model.layers = prentm_model.layers[:-3]
    # need to change configuration for proper save/load operations
    # config['layers'] = config['layers'][:-3]
    # config['output_layers'] = [['hidden_lstm', 0, 0]]
    # config['name'] = 'pre_ntm'
    # prentm_model = prentm_model.from_config(config)
    # prentm_model.set_weights(weights)
    # for layer in prentm_model.layers:
    #     if hasattr(layer, 'layer'):  # layers embedded in wrappers
    #         layer.layer.trainable = False
    #     layer.trainable = False

    prentm_model.trainable = False
    print("\nPre-NTM model hidden_lstm loaded successfully!\n")

    hidden_lstm = prentm_model(input_list)
    # hidden_lstm = Dropout(0.5)(hidden_lstm)

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
    # ntm_layer = Dropout(0.3)(ntm_layer)

    hidden_ntm = TimeDistributed(Dense(1024, activation='relu', kernel_constraint=max_norm(10.)))(ntm_layer)
    hidden_ntm = Dropout(0.5)(hidden_ntm)
    decoder = TimeDistributed(Dense(512, activation='relu', kernel_constraint=max_norm(5.)))(hidden_ntm )
    decoder = Dropout(0.5)(decoder)

    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(decoder)
    model = Model(inputs=input_list, outputs=[outlayer])
    # model.summary()
    # stateful controller requires smaller lr
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['categorical_accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['categorical_accuracy'])
    return model


def get_combined_ntm_model(batch_size=32, group_size=10, m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, max_len=15,
                   read_heads=1, write_heads=1, nb_classes=6, input_dropout=0.5, embedding_matrix=None, **kwargs):
    """feed softmax to ntm"""

    # Shared embedding layer
    embedding = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)
    # left_input = Input(batch_shape=(batch_size, group_size, max_len))
    # right_input = Input(batch_shape=(batch_size, group_size, max_len))
    # left_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    # right_context_input = Input(batch_shape=(batch_size, group_size, max_len))
    # type_input = Input(batch_shape=(batch_size, group_size, 1))
    # time_diff_input = Input(batch_shape=(batch_size, group_size, 3))

    left_input = Input(shape=(group_size, max_len))
    left_input_emb = embedding(left_input)
    left_branch = Dropout(input_dropout)(left_input_emb)
    # left_branch = Dropout(input_dropout)(left_input)
    left_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_branch)
    left_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_branch)
    left_branch = Lambda(lambda x: K.max(x, axis=-2), name='left_branch')(left_branch)

    right_input = Input(shape=(group_size, max_len))
    right_input_emb = embedding(right_input)
    right_branch = Dropout(input_dropout)(right_input_emb)
    # right_branch = Dropout(input_dropout)(right_input)
    right_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(right_branch)
    right_branch = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(right_branch)
    right_branch = Lambda(lambda x: K.max(x, axis=-2), name='right_branch')(right_branch)

    left_context_input = Input(shape=(group_size, max_len))
    left_context_input_emb = embedding(left_context_input)
    left_context = Dropout(input_dropout)(left_context_input_emb)
    # left_context = Dropout(input_dropout)(left_context_input)
    left_context = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(left_context)
    left_context = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(left_context)
    left_context = Lambda(lambda x: K.max(x, axis=-2), name='left_context')(left_context)

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
    # hidden_lstm = Dropout(0.5)(hidden_lstm)
    hidden_lstm = concatenate([hidden_lstm, type_input, time_diff_input], name='hidden_lstm')
    encoder = concatenate([left_context, right_context, hidden_lstm, type_input, time_diff_input])

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
    decoder = TimeDistributed(Dense(128, activation='sigmoid'))(hidden_ntm)
    decoder = Dropout(0.3)(decoder)

    outlayer = TimeDistributed(Dense(nb_classes, activation='softmax'), name='main_out')(decoder)
    model = Model(inputs=[left_input, right_input, type_input, left_context_input, right_context_input, time_diff_input], outputs=[outlayer])
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0002), metrics=['categorical_accuracy'])

    return model
