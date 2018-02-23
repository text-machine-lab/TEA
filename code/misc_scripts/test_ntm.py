"""test neural turing machine"""
from __future__ import print_function

import os,sys
import keras
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from datetime import datetime
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN

LOG_PATH_BASE="logs/"     #this is for tensorboard callbacks

sys.path.insert(1, os.path.join(sys.path[0], '..'))
#from learning.ntm2 import SingleKeyNTM as NTM
from learning.ntm2 import SimpleNTM as NTM

def test_model(model, sequence_length=None, verbose=False, batch_size=100):
    # print("input shape", model.input_shape)
    input_dim = model.input_shape[-1]
    output_dim = model.output_shape[-1]
    batch_size = batch_size

    I, V, sw = next(get_sample(batch_size=batch_size, in_bits=input_dim, out_bits=output_dim,
                                        max_size=sequence_length, min_size=sequence_length))
    Y = np.asarray(model.predict(I, batch_size=batch_size))

    if not np.isnan(Y.sum()): #checks for a NaN anywhere
        Y = (Y > 0.5).astype('float64')
        x = V[:, -sequence_length:, :] == Y[:, -sequence_length:, :]
        acc = x.mean() * 100
        if verbose:
            print("the overall accuracy for sequence_length {0} was: {1}".format(sequence_length, x.mean()))
            print("per bit")
            print(x.mean(axis=(0,1)))
            print("per timeslot")
            print(x.mean(axis=(0,2)))
        # check_weights = model.layers[1].get_weights()[7]  # W_k_read
        # print("check weights", check_weights)
    else:
        weights = model.layers[1].get_weights()
        print("post-train weights", weights)
        sys.exit(1)
        # import pudb; pu.db
        acc = 0
    return acc


def train_model(model, epochs=10, min_size=5, max_size=20, callbacks=None, verbose=False, batch_size=100):
    input_dim = model.input_shape[-1]
    output_dim = model.output_shape[-1]
    batch_size = batch_size

    sample_generator = get_sample(batch_size=batch_size, in_bits=input_dim, out_bits=output_dim,
                                            max_size=max_size, min_size=min_size)
    #keras.backend.get_session().run(tf.global_variables_initializer())
    if verbose:
        for j in range(epochs):
            model.fit_generator(sample_generator, steps_per_epoch=10, epochs=j+1, callbacks=callbacks, initial_epoch=j)
            print("currently at epoch {0}".format(j+1))
            model.reset_states()
            for i in [5,10,20,40]:
                test_model(model, sequence_length=i, verbose=True, batch_size=batch_size)
    else:
        model.fit_generator(sample_generator, steps_per_epoch=10, epochs=epochs, callbacks=callbacks)

    print("done training")

# def train_model(model, epochs=10, min_size=5, max_size=20, callbacks=None, verbose=False, batch_size=100):
#     input_dim = model.input_shape[-1]
#     output_dim = model.output_shape[-1]
#     batch_size = batch_size
#
#     sample_generator = get_sample(batch_size=batch_size, in_bits=input_dim, out_bits=output_dim,
#                                                 max_size=max_size, min_size=min_size)
#     if verbose:
#         for j in range(epochs):
#             for step in range(10):
#                 x, y, sample_weight = sample_generator.next()
#                 model.fit(x, y, sample_weight=sample_weight, callbacks=callbacks, epochs=1, verbose=1, validation_split=0.0,)
#             print("currently at epoch {0}".format(j+1))
#             for i in [5,10,20,40]:
#                 test_model(model, sequence_length=i, verbose=True)
#     else:
#         model.fit_generator(sample_generator, steps_per_epoch=10, epochs=epochs, callbacks=callbacks)
#
#     print("done training")


def lengthy_test(model, testrange=[5,10,20,40,80], epochs=100, verbose=True, batch_size=100):
    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = LOG_PATH_BASE + ts + "_-_" + "NTM"
    tensorboard = TensorBoard(log_dir=log_path,
                                write_graph=False, #This eats a lot of space. Enable with caution!
                                # histogram_freq = 1,
                                write_images=True,
                                batch_size = batch_size,
                                write_grads=True)
    model_saver = ModelCheckpoint(log_path + 'best_weights.h5', monitor='loss', save_best_only=True, save_weights_only=True)
    callbacks = [tensorboard, TerminateOnNaN(), model_saver]

    # for i in testrange:
    #     acc = test_model(model, sequence_length=i, verbose=verbose, batch_size=batch_size)
    #     print("Before training, the accuracy for length {0} was: {1}%".format(i,acc))

    # weights = model.layers[1].get_weights()
    # print("pre-train weights", weights)
    train_model(model, epochs=epochs, callbacks=callbacks, verbose=verbose, batch_size=batch_size)

    for i in testrange:
        acc = test_model(model, sequence_length=i, verbose=verbose, batch_size=batch_size)
        print("After training, the accuracy for length {0} was: {1}%".format(i,acc))
    return


def get_sample(batch_size=128, in_bits=10, out_bits=8, max_size=20, min_size=1):
    # in order to be a generator, we start with an endless loop:
    while True:
        # generate samples with random length.
        # there a two flags, one for the beginning of the sequence
        # (only second to last bit is one)
        # and one for the end of the sequence (only last bit is one)
        # every other time those are always zero.
        # therefore the length of the generated sample is:
        # 1 + actual_sequence_length + 1 + actual_sequence_length

        # make flags
        begin_flag = np.zeros((1, in_bits))
        begin_flag[0, in_bits - 2] = 1
        end_flag = np.zeros((1, in_bits))
        end_flag[0, in_bits - 1] = 1

        # initialize arrays: for processing, every sequence must be of the same length.
        # We pad with zeros.
        temporal_length = max_size * 2 + 2
        # "Nothing" on our band is represented by 0.5 to prevent immense bias towards 0 or 1.
        inp = np.ones((batch_size, temporal_length, in_bits)) * 0.5
        out = np.ones((batch_size, temporal_length, out_bits)) * 0.5
        # sample weights: in order to make recalling the sequence much more important than having everything set to 0
        # before and after, we construct a weights vector with 1 where the sequence should be recalled, and small values
        # anywhere else.
        sw = np.ones((batch_size, temporal_length)) * 0.01

        # make actual sequence
        for i in range(batch_size):
            ts = np.random.randint(low=min_size, high=max_size + 1)
            actual_sequence = np.random.uniform(size=(ts, out_bits)) > 0.5
            output_sequence = np.concatenate((np.ones((ts + 2, out_bits)) * 0.5, actual_sequence), axis=0)

            # pad with zeros where only the flags should be one
            padded_sequence = np.concatenate((actual_sequence, np.zeros((ts, 2))), axis=1)
            input_sequence = np.concatenate((begin_flag, padded_sequence, end_flag), axis=0)

            # this embedds them, padding with the neutral value 0.5 automatically
            inp[i, :input_sequence.shape[0]] = input_sequence
            out[i, :output_sequence.shape[0]] = output_sequence
            sw[i, ts + 2: ts + 2 + ts] = 1

        yield inp, out, sw


def get_model(batch_size=100, bidirectional=False):
    from keras.layers import Input

    ntm_output_dim = 8
    ntm_input_dim = ntm_output_dim + 2  # this is the actual input dim of the network, that includes two dims for flags
    m_depth = 20
    n_slots = 128
    # batch_size = 100
    shift_range = 5
    read_heads = 3
    write_heads = 3
    in_bits = 10
    max_size = 20

    clipnorm = 1.

    temporal_length = max_size * 2 + 2

    input = Input(batch_shape=(batch_size, None, in_bits))

    ntm = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
                return_sequences=True, input_shape=(None, ntm_input_dim), batch_size=batch_size,
                stateful=False, activation='tanh')(input)

    if bidirectional == True:
        from keras.layers import Lambda, average
        from keras.backend import reverse

        ntm_backward = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                  read_heads=read_heads, write_heads=write_heads, controller_stateful=True,
                  return_sequences=True, input_shape=(None, ntm_input_dim), batch_size=batch_size,
                  stateful=False, activation='tanh', go_backwards=True)(input)

        # # make a layer to reverse output
        # Reverse = Lambda(lambda x: reverse(x, axes=-2), output_shape=(batch_size, ntm_output_dim))
        # ntm_backward = Reverse(ntm_backward)

        ntm = average([ntm, ntm_backward])

    model = Model(inputs=[input], outputs=[ntm])
    model.name = "NTM_-_" + "LSTM-controller"
    model.summary()

    # sgd = Adam(lr=0.1)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['binary_accuracy'], sample_weight_mode="temporal")
    return model

if __name__ == '__main__':
    batch_size = 100

    model = get_model(batch_size=batch_size, bidirectional=True)
    print("Model built. Starting test...")
    lengthy_test(model, epochs=400, verbose=True, batch_size=batch_size)
