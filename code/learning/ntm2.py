import numpy as np

# import theano
# import theano.tensor as T
# floatX = theano.config.floatX
import tensorflow as tf

from keras.layers.recurrent import Recurrent, GRU, LSTM
from keras.layers import TimeDistributed, Dense, Input
from keras import backend as K
from keras.engine.topology import InputSpec
from keras.activations import get as get_activations
from keras import initializers
from keras.models import Model

from keras.activations import softmax, tanh, sigmoid, hard_sigmoid, relu

def _circulant(leng, n_shifts):
    # This is more or less the only code still left from the original author,
    # EderSantan @ Github.
    # My implementation would probably just be worse.
    # Below his original comment:

    """
    I confess, I'm actually proud of this hack. I hope you enjoy!
    This will generate a tensor with `n_shifts` of rotated versions the
    identity matrix. When this tensor is multiplied by a vector
    the result are `n_shifts` shifted versions of that vector. Since
    everything is done with inner products, everything is differentiable.

    Paramters:
    ----------
    leng: int > 0, number of memory locations
    n_shifts: int > 0, number of allowed shifts (if 1, no shift)

    Returns:
    --------
    shift operation, a tensor with dimensions (n_shifts, leng, leng)
    """
    eye = np.eye(leng)
    shifts = range(n_shifts // 2, -n_shifts // 2, -1)
    C = np.asarray([np.roll(eye, s, axis=1) for s in shifts])
    return K.variable(C.astype(K.floatx()))


def _renorm(x, axis=1):
    return x / (K.sum(x, axis=axis, keepdims=True) + 1e-4)
    # return x / K.sum(x, axis=1, keepdims=True)


def _softmax(x):
    wt = x.flatten(ndim=2)
    w = K.softmax(wt)
    return w.reshape(x.shape)  # T.clip(s, 0, 1)


def _cosine_similar(M, k):
    nk = K.l2_normalize(k, axis=-1)
    nM = K.l2_normalize(M, axis=-1)
    cosine_sim = K.batch_dot(nM, nk)
    # print(tf.Print(cosine_distance, [cosine_distance], message="NaN occured in _cosine_distance"))
    return cosine_sim

def _euclidean_similar(M, k):
    ssd = K.square(M-k[:, None, :])
    distance = K.sum(ssd, axis=-1) + 1e-4
    distance = K.l2_normalize(distance, axis=-1)
    # print('Euclidean distance:\n',tf.Print(distance, [distance], message="Euclidean distance:", first_n=50))
    return K.softmax(1-distance)

def _tensor_mean(tensors):
    """return the mean of a list of tensors"""
    stack = K.stack(tensors, axis=-1)
    tensor_mean = K.mean(stack, axis=-1, keepdims=False)
    return tensor_mean

def get_lstm_controller(controller_output_dim, controller_input_dim, activation='relu', batch_size=1, max_steps=1):
    controller = LSTM(units=controller_output_dim,
                      # kernel_initializer='random_normal',
                      # bias_initializer='random_normal',
                      activation=activation,
                      stateful=True,
                      return_state=True,
                      return_sequences=False, # does not matter because for controller the sequence len is 1?
                      implementation=2,  # best for gpu. other ones also might not work.
                      batch_input_shape=(batch_size, max_steps, controller_input_dim),
                      name='lstm_controller')
    controller.build(input_shape=(batch_size, max_steps, controller_input_dim))
    return controller


def get_dense_controller(controller_output_dim, controller_input_dim, activation='relu', batch_size=1):
    controller = Dense(controller_output_dim, activation=activation, batch_input_shape=(batch_size, controller_input_dim))
    controller.build(input_shape=(batch_size, controller_input_dim))

    return controller


class NeuralTuringMachine(Recurrent):
    """ Neural Turing Machines
    Non obvious parameter:
    ----------------------
    shift_range: int, number of available shifts, ex. if 3, avilable shifts are
                 (-1, 0, 1)
    n_slots: number of memory locations
    m_depth: memory length at each location
    Known issues:
    -------------
    Theano may complain when n_slots == 1.
    """

    def __init__(self, units,
                 n_slots=50,
                 m_depth=20,
                 shift_range=3,
                 # controller_model=None,
                 read_heads=1,
                 write_heads=1,
                 activation='tanh',
                 batch_size=1,
                 # stateful=True, # let super class handle it
                 controller_stateful=True,
                 key_range=None,
                 **kwargs):

        super(NeuralTuringMachine, self).__init__(**kwargs)

        self.output_dim = units
        self.units = units  # this is used in the inherited Recurrent class
        self.n_slots = n_slots
        self.m_depth = m_depth
        self.shift_range = shift_range
        # self.controller = controller_model
        self.controller_stateful = controller_stateful
        self.activation = activation
        self.read_heads = read_heads
        self.write_heads = write_heads
        self.batch_size = batch_size
        # self.stateful = stateful
        self.controller_states = None  # used to store controller states

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        batch_size, input_length, self.input_dim = input_shape
        self.input_spec = [InputSpec(ndim=3)]
        self.input_spec[0] = InputSpec(shape=input_shape)

        controller_input_dim = self.input_dim + self.m_depth  # support multiple heads

        self.state_spec = [InputSpec(shape=(self.batch_size, self.output_dim)),  # output
                           InputSpec(shape=(self.batch_size, self.n_slots * self.m_depth)),  # M
                           InputSpec(shape=(self.batch_size, self.n_slots)),  # Wr
                           InputSpec(shape=(self.batch_size, self.n_slots))]  # Ww

        if self.controller_stateful:
            self.controller = get_lstm_controller(self.output_dim, controller_input_dim, activation=self.activation,
                                              batch_size=batch_size, max_steps=None)
        else:
            self.controller = get_dense_controller(self.output_dim, controller_input_dim, activation=self.activation)
        print("Controller loaded. stateful = %d" %self.controller_stateful)

        # initial memory, state, read and write vecotrs
        # self.clear_M()
        # self.init_h = K.zeros((self.output_dim))
        # self.init_wr = K.random_uniform_variable((self.n_slots,), -1, 1)
        # self.init_ww = K.random_uniform_variable((self.n_slots,), -1, 1)


        glorot_normal = initializers.glorot_normal()
        # Zeros = initializers.Zeros()
        # write
        self.W_e = K.variable(glorot_normal((self.output_dim, self.m_depth)) ) # erase
        self.b_e = K.zeros((self.m_depth,))
        self.W_a = K.variable(glorot_normal((self.output_dim, self.m_depth)) ) # add
        self.b_a = K.zeros((self.m_depth,))

        # get_w  parameters for reading operation
        self.W_k_read = [K.variable(glorot_normal((self.input_dim, self.m_depth))) for _ in range(self.read_heads)]
        self.b_k_read = [K.variable(glorot_normal((self.m_depth, ))) for _ in range(self.read_heads)]
        self.W_c_read = [K.variable(glorot_normal((self.input_dim, 3))) for _ in range(self.read_heads)]  # 3 = beta, g, gamma see eq. 5, 7, 9
        self.b_c_read = [K.zeros((3,)) for _ in range(self.read_heads)]
        self.W_s_read = [K.variable(glorot_normal((self.input_dim, self.shift_range))) for _ in range(self.read_heads)]
        self.b_s_read = [K.zeros((self.shift_range,)) for _ in range(self.read_heads)]  # b_s lol! not intentional

        # get_w  parameters for writing operation
        self.W_k_write = [K.variable(glorot_normal((self.output_dim, self.m_depth))) for _ in range(self.write_heads)]
        self.b_k_write = [K.variable(glorot_normal((self.m_depth, ))) for _ in range(self.write_heads)]
        self.W_c_write = [K.variable(glorot_normal((self.output_dim, 3))) for _ in range(self.write_heads)]  # 3 = beta, g, gamma see eq. 5, 7, 9
        self.b_c_write = [K.zeros((3,)) for _ in range(self.write_heads)]
        self.W_s_write = [K.variable(glorot_normal((self.output_dim, self.shift_range))) for _ in range(self.write_heads)]
        self.b_s_write = [K.zeros((self.shift_range,)) for _ in range(self.write_heads)]

        self.C = _circulant(self.n_slots, self.shift_range)

        # print("controller.trainable_weights", self.controller.trainable_weights)
        self.trainable_weights = self.controller.trainable_weights + [self.W_e, self.b_e, self.W_a, self.b_a] \
            + self.W_k_read + self.b_k_read + self.W_c_read + self.b_c_read +  self.W_s_read + self.b_s_read \
            + self.W_k_write + self.b_k_write + self.W_c_write + self.b_c_write + self.W_s_write + self.b_s_write
            #self.M,
            # self.init_h, self.init_wr, self.init_ww]


        # states list [init_wr, init_ww, init_contoller_output]
        # self.states = self.get_initial_state(None)
        self.states = [None, None, None, None]
        if self.stateful:
            self.reset_states()

        self.built = True

    # def clear_M(self):
    #     self.M = K.ones((self.batch_size, self.n_slots, self.m_depth), name='main_memory') * 0.01

    def get_initial_states(self, inputs):
        # keras has a typo in some version, (states --> state)
        # I included it here only to make it work on different versions
        self.get_initial_state(inputs)

    def get_initial_state(self, inputs):
        # we do not include controller states here

        init_M = K.variable(K.ones((self.batch_size, self.n_slots*self.m_depth), name='main_memory') * 0.001)
        # init_M = K.variable(K.ones((self.batch_size, self.n_slots * self.m_depth)) * 0.001)
        init_contoller_output = K.variable(K.ones((self.batch_size, self.output_dim)) * 0.5, name="init_contoller_output")

        # initial reading weights
        init_wr = np.zeros((self.batch_size, self.n_slots))
        init_wr[:, 0] = 1  # bias term
        init_wr = K.variable(init_wr, name="init_weights_read")
        # intial writing weights
        init_ww = np.zeros((self.batch_size, self.n_slots))
        init_ww[:, 0] = 1  # bias term
        init_ww = K.variable(init_ww, name="init_weights_write")

        return [init_contoller_output, init_M, init_wr, init_ww]

    def _read(self, w, M):
        return K.sum((w[:, :, None] * M),axis=1)  # (batch_size, read_heads, m_depth)

    def _write(self, w, e, a, M):
        Mtilda = M * (1 - w[:, :, None]*e[:, None, :])
        Mout = Mtilda + w[:, :, None]*a[:, None, :]
        return Mout

    def _direct_write(self, w, a, M):
        Mtilda = M * (1 - w[:, :, None])
        Mout = Mtilda + w[:, :, None] * a[:, None, :]
        return Mout

    def _get_content_w(self, beta, k, M):
        # convert beta to (batch_size, 1) to broadcast its value over slots
        num = beta[:, None] * _euclidean_similar(M, k)
        return K.softmax(num)

    def _get_raw_location_w(self, gamma, wc):
        wout = _renorm(wc ** gamma[:, None])
        return wout

    def _get_ungated_location_w(self, s, C, gamma, wc, **kwargs):
        """Do not consider previous state"""
        Cs = K.sum(C[None, :, :, :] * wc[:, None, None, :], axis=3)
        wtilda = K.sum(Cs * s[:, :, None], axis=1)
        wout = _renorm(wtilda ** gamma[:, None])
        return wout

    def _get_gated_location_w(self, g, s, C, gamma, wc, w_tm1):
        # print("g, s, C, gamma, wc, w_tm1", g, s, C, gamma, wc, w_tm1)
        gwc = g[:, None] * wc
        wg = gwc + (1-g[:, None])*w_tm1
        Cs = K.sum(C[None, :, :, :] * wg[:, None, None, :], axis=3)
        wtilda = K.sum(Cs * s[:, :, None], axis=1)
        wout = _renorm(wtilda ** gamma[:, None])
        return wout

    def _get_controller_output(self, inputs, W_k, b_k, W_c, b_c, W_s, b_s, heads=1, W_k2=None, b_k2=None):
        # h:(batch_size, output_dim), Wk:(output_dim, m_depth), b_k:(self.m_depth, )
        if W_k is not None and b_k is not None:
            k = _tensor_mean([K.relu(K.bias_add(K.dot(inputs, W_k[i]), b_k[i])) for i in range(heads)])
            if W_k2 is not None and b_k2 is not None:
                k = _tensor_mean([K.relu(K.bias_add(K.dot(k, W_k2[i]), b_k2[i])) for i in range(heads)])
        else:
            k = None
        c = _tensor_mean([K.bias_add(K.dot(inputs, W_c[i]), b_c[i]) for i in range(heads)])
        beta = K.relu(c[:, 0]) + 10.0
        g = K.sigmoid(c[:, 1])
        gamma = K.relu(c[:, 2]) + 20.0
        s = _tensor_mean([K.softmax(K.bias_add(K.dot(inputs, W_s[i]), b_s[i])) for i in range(heads)])
        return k, beta, g, gamma, s

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return input_shape[0], input_shape[1], self.output_dim
        else:
            return input_shape[0], self.output_dim

    # def get_full_output(self, train=False):
    #     """
    #     This method is for research and visualization purposes. Use it as
    #     X = model.get_input()  # full model
    #     Y = ntm.get_output()    # this layer
    #     F = theano.function([X], Y, allow_input_downcast=True)
    #     [memory, read_address, write_address, rnn_state] = F(x)
    #     if inner_rnn == "lstm" use it as
    #     [memory, read_address, write_address, rnn_cell, rnn_state] = F(x)
    #     """
    #     # input shape: (nb_samples, time (padded with zeros), input_dim)
    #     X = self.get_input(train)
    #     assert K.ndim(X) == 3
    #     if K._BACKEND == 'tensorflow':
    #         if not self.input_shape[1]:
    #             raise Exception('When using TensorFlow, you should define ' +
    #                             'explicitely the number of timesteps of ' +
    #                             'your sequences. Make sure the first layer ' +
    #                             'has a "batch_input_shape" argument ' +
    #                             'including the samples axis.')
    #
    #     mask = self.get_output_mask(train)
    #     if mask:
    #         # apply mask
    #         X *= K.cast(K.expand_dims(mask), X.dtype)
    #         masking = True
    #     else:
    #         masking = False
    #
    #     if self.stateful:
    #         initial_states = self.states
    #     else:
    #         initial_states = self.get_initial_states(X)
    #
    #     states = rnn_states(self.step, X, initial_states,
    #                         go_backwards=self.go_backwards,
    #                         masking=masking)
    #     return states

    def _update_controller(self, inputs, read_vector):
        """Give input and read memory to controller. Get the output of the controller.
        """
        controller_input = K.concatenate([inputs, read_vector])

        if self.controller_stateful:  # use stateful lstm cell
            # now shape changed to (batch, 1, input_size)) # pseudo time step 1
            controller_input = controller_input[:, None, :]
            controller_output_states = self.controller(controller_input,
                                                       initial_state=self.controller_states)  # call with states
            self.controller_states = controller_output_states[1:]
            controller_output = controller_output_states[0]  # first output of controller (RNN)

            if self.controller.output_shape == 3:
                # first output of the list (should be one-element list anyway)
                controller_output = controller_output[:, 0, :]

        else: # use dense controller
            controller_output = self.controller(controller_input)
        # print("controller_output", controller_output)
        return controller_output

    def step(self, x, states):

        # statets list: [init_contoller_output, init_M, init_wr, init_ww]
        # print("states:", states)
        controller_output_tm1, M_tm1, wr_tm1, ww_tm1 = states
        # The step function will slice the second dim (supposed to be timesteps in most cases) and only pass one piece
        # so if we need to keep all dimensions and pass it to next time step we need to reshape it
        M_tm1 = K.reshape(M_tm1, (self.batch_size, self.n_slots, self.m_depth))

        # read
        # read keys (k) are calculated from previous output? Maybe it should rely on current input
        # k_read, beta_read, g_read, gamma_read, s_read = self._get_controller_output(
        #     controller_output_tm1, self.W_k_read, self.b_k_read, self.W_c_read, self.b_c_read,
        #     self.W_s_read, self.b_s_read)
        k_read, beta_read, g_read, gamma_read, s_read = self._get_controller_output(x, self.W_k_read, self.b_k_read,
                                    self.W_c_read, self.b_c_read, self.W_s_read, self.b_s_read, heads=self.read_heads)

        wc_read = self._get_content_w(beta_read, k_read, M_tm1)
        # wr_t = self._get_ungated_location_w(s_read, self.C, gamma_read, wc_read)
        wr_t = self._get_gated_location_w(g_read, s_read, self.C, gamma_read, wc_read, wr_tm1)
        M_read = self._read(wr_t, M_tm1)

        # update controller
        controller_output_t = self._update_controller(x, M_read)

        # write
        k_write, beta_write, g_write, gamma_write, s_write = self._get_controller_output(controller_output_t,
                                    self.W_k_write, self.b_k_write, self.W_c_write, self.b_c_write, self.W_s_write,
                                    self.b_s_write, heads=self.write_heads)
        wc_write = self._get_content_w(beta_write, k_write, M_tm1)
        ww_t = self._get_gated_location_w(g_write, s_write, self.C, gamma_write, wc_write, ww_tm1)
        e = K.sigmoid(K.bias_add(K.dot(controller_output_t, self.W_e), self.b_e))
        a = K.tanh(K.bias_add(K.dot(controller_output_t, self.W_a), self.b_a))
        # M_t = self._write(ww_t, e, a, M_tm1)
        M_t = self._write(ww_t, e, a, M_tm1)

        M_t = K.batch_flatten(M_t)

        # for i, weights in enumerate(self.trainable_weights):
        #     self.trainable_weights[i] = K.clip(weights, -10., 10.)

        return controller_output_t, [controller_output_t, M_t, wr_t, ww_t]

    def reset_states(self, states=None):
        # super(NeuralTuringMachine, self).reset_states()
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')

        self.states = self.get_initial_state(None)
        # else:
        #     for index, (value, state) in enumerate(zip(states, self.states)):
        #         K.set_value(state, value)

        if self.controller_stateful:
            self.controller.reset_states()


    def get_config(self):
        config = {'units' : self.output_dim,
        'n_slots' : self.n_slots,
        'm_depth' : self.m_depth,
        'shift_range' : self.shift_range,
        'controller_model' : self.controller,
        'read_heads' : self.read_heads,
        'write_heads' : self.write_heads,
        'activation' : self.activation,
        'batch_size' : self.batch_size,
        'stateful' : self.stateful,
        'controller_stateful' : self.controller_stateful}

        base_config = super(NeuralTuringMachine, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

class SingleKeyNTM(NeuralTuringMachine):
    """ NTM using the same key function for reading and writing.
        The read head uses input (x) and previous output (h_tm1) to compute keys,
        and compare them with M elements to get reading weights.
        The write head saves the keys in M.
    """
    def __init__(self, units, key_range=None, **kwargs):
        super(SingleKeyNTM, self).__init__(units, **kwargs)
        self.write_heads = self.read_heads

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        batch_size, input_length, self.input_dim = input_shape
        self.input_spec = [InputSpec(ndim=3)]
        self.input_spec[0] = InputSpec(shape=input_shape)

        controller_input_dim = self.input_dim + self.m_depth  # support multiple heads

        if self.controller_stateful:
            self.controller = get_lstm_controller(self.output_dim, controller_input_dim, activation=self.activation,
                                              batch_size=batch_size, max_steps=None)
        else:
            self.controller = get_dense_controller(self.output_dim, controller_input_dim, activation=self.activation)
        print("Controller loaded. stateful = %d" %self.controller_stateful)

        glorot_normal = initializers.glorot_normal()
        # for erase gate
        self.W_e = K.variable(glorot_normal((self.output_dim, self.m_depth)) ) # erase
        self.b_e = K.zeros((self.m_depth,))

        # get_w  parameters for reading and writing operation
        self.W_k = [K.variable(glorot_normal((self.input_dim + self.output_dim, self.m_depth))) for _ in range(self.read_heads)]
        self.b_k = [K.variable(glorot_normal((self.m_depth, ))) for _ in range(self.read_heads)]
        self.W_c = [K.variable(glorot_normal((self.input_dim + self.output_dim, 3))) for _ in range(self.read_heads)]  # 3 = beta, g, gamma see eq. 5, 7, 9
        self.b_c = [K.zeros((3,)) for _ in range(self.read_heads)]
        self.W_s = [K.variable(glorot_normal((self.input_dim + self.output_dim, self.shift_range))) for _ in range(self.read_heads)]
        self.b_s = [K.zeros((self.shift_range,)) for _ in range(self.read_heads)]

        self.C = _circulant(self.n_slots, self.shift_range)

        # print("controller.trainable_weights", self.controller.trainable_weights)
        self.trainable_weights = self.controller.trainable_weights + [self.W_e, self.b_e] \
            + self.W_k + self.b_k + self.W_c + self.b_c +  self.W_s + self.b_s  # each is a weight list

        self.states = [None, None, None, None]
        if self.stateful:
            self.reset_states()

        self.built = True

    def step(self, x, states):

        # statets list: [init_contoller_output, init_M, init_wr, init_ww]
        controller_output_tm1, M_tm1, wr_tm1, ww_tm1 = states
        # The step function will slice the second dim (supposed to be timesteps in most cases) and only pass one piece
        # so if we need to keep all dimensions and pass it to next time step we need to reshape it
        M_tm1 = K.reshape(M_tm1, (self.batch_size, self.n_slots, self.m_depth))

        # read
        inputs_read = K.concatenate([x, controller_output_tm1])
        k_read, beta_read, g_read, gamma_read, s_read = self._get_controller_output(inputs_read, self.W_k, self.b_k,
                                    self.W_c, self.b_c, self.W_s, self.b_s, heads=self.read_heads)
        wc_read = self._get_content_w(beta_read, k_read, M_tm1)
        wr_t = self._get_gated_location_w(g_read, s_read, self.C, gamma_read, wc_read, wr_tm1)
        M_read = self._read(wr_t, M_tm1)

        # update controller
        controller_output_t = self._update_controller(x, M_read)

        # write
        inputs_write = K.concatenate([x, controller_output_t])
        k_write, beta_write, g_write, gamma_write, s_write = self._get_controller_output(inputs_write,
                                    self.W_k, self.b_k, self.W_c, self.b_c, self.W_s,
                                    self.b_s, heads=self.write_heads)
        wc_write = self._get_content_w(beta_write, k_write, M_tm1)
        ww_t = self._get_gated_location_w(g_write, s_write, self.C, gamma_write, wc_write, ww_tm1)
        e = K.sigmoid(K.bias_add(K.dot(controller_output_t, self.W_e), self.b_e))
        a = k_write

        M_t = self._write(ww_t, e, a, M_tm1)
        M_t = K.batch_flatten(M_t)

        # for i, weights in enumerate(self.trainable_weights):
        #     self.trainable_weights[i] = K.clip(weights, -10., 10.)

        return controller_output_t, [controller_output_t, M_t, wr_t, ww_t]

class SimpleNTM(NeuralTuringMachine):
    """ NTM without key function for addressing.
        The read head and write head directly use input (x) as key for addressing.
        The write head saves the keys in M.
    """
    def __init__(self, units, key_range=None, **kwargs):
        super(SimpleNTM, self).__init__(units, **kwargs)
        self.write_heads = self.read_heads
        self.key_range = key_range

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        batch_size, input_length, self.input_dim = input_shape
        self.input_spec = [InputSpec(ndim=3)]
        self.input_spec[0] = InputSpec(shape=input_shape)
        self.output_dim = self.input_dim
        self.units = self.output_dim
        print("output_dim overritten by input_dim!")
        self.m_depth = self.output_dim
        print("m_depth overritten by input_dim!")

        if self.key_range is None:
            self.key_range = self.input_dim

        self.state_spec = [InputSpec(shape=(self.batch_size, self.output_dim)),  # output
                           InputSpec(shape=(self.batch_size, self.n_slots * self.m_depth)),  # M
                           InputSpec(shape=(self.batch_size, self.n_slots * self.key_range))]  # Keys

        # controller_input_dim = self.input_dim + self.m_depth  # support multiple heads
        controller_input_dim = self.input_dim

        if self.controller_stateful:
            self.controller = get_lstm_controller(self.output_dim, controller_input_dim, activation=self.activation,
                                              batch_size=batch_size, max_steps=None)
        else:
            self.controller = get_dense_controller(self.output_dim, controller_input_dim, activation=self.activation)
        print("Controller loaded. stateful = %d" %self.controller_stateful)

        glorot_normal = initializers.glorot_normal()

        # get_w  parameters for reading and writing operation
        self.W_c = [K.variable(glorot_normal((self.input_dim + self.output_dim, 3))) for _ in range(self.read_heads)]  # 3 = beta, g, gamma see eq. 5, 7, 9
        self.b_c = [K.zeros((3,)) for _ in range(self.read_heads)]
        self.W_s = [K.variable(glorot_normal((self.input_dim + self.output_dim, self.shift_range))) for _ in range(self.read_heads)]
        self.b_s = [K.zeros((self.shift_range,)) for _ in range(self.read_heads)]

        self.C = _circulant(self.n_slots, self.shift_range)

        # print("controller.trainable_weights", self.controller.trainable_weights)
        self.trainable_weights = self.controller.trainable_weights \
            + self.W_c + self.b_c +  self.W_s + self.b_s  # each is a weight list

        self.states = [None, None, None]
        if self.stateful:
            self.reset_states()

        self.built = True

    def get_initial_state(self, inputs):
        # we do not include controller states here

        init_M = K.variable(K.ones((self.batch_size, self.n_slots * self.m_depth), name='main_memory') * 0.001)
        init_Keys = K.variable(K.ones((self.batch_size, self.n_slots * self.key_range), name='main_memory') * 0.001)
        init_contoller_output = K.variable(K.ones((self.batch_size, self.output_dim)) * 0.5,
                                           name="init_contoller_output")

        return [init_contoller_output, init_M, init_Keys]

    def _smart_similar(self, Keys, k):
        # sim1 = _euclidean_similar(Keys, k)
        sim1 = _cosine_similar(Keys, k)
        k_permute = K.concatenate([k[:, self.key_range/2:], k[:, :self.key_range/2]], axis=-1)
        # sim2 = _euclidean_similar(Keys, k_permute)
        sim2 = _cosine_similar(Keys, k_permute)
        stacked = K.stack([sim1, sim2], axis=0)

        return K.max(stacked, axis=0)

    def _get_content_w(self, beta, k, M):
        # convert beta to (batch_size, 1) to broadcast its value over slots
        num = beta[:, None] * self._smart_similar(M, k)
        return K.softmax(num)

    def _update_controller(self, inputs, read_vector):
        """Give input and read memory to controller. Get the output of the controller.
        """
        # controller_input = K.concatenate([inputs - read_vector, inputs + read_vector])
        controller_input = K.relu(inputs - read_vector)
        with open('/home/ymeng/projects/TEA/log_nohup/read_vector.debug', 'a') as f:
            t = tf.Print(read_vector, [read_vector], message="read_vector:", first_n=50)
            try:
                t.eval().__repre__()
                f.write(t.eval().__repr__())
            except ValueError:
                f.write(t.__repr__())

        if self.controller_stateful:  # use stateful lstm cell
            # now shape changed to (batch, 1, input_size)) # pseudo time step 1
            controller_input = controller_input[:, None, :]
            controller_output_states = self.controller(controller_input,
                                                       initial_state=self.controller_states)  # call with states
            self.controller_states = controller_output_states[1:]
            controller_output = controller_output_states[0]  # first output of controller (RNN)

            if self.controller.output_shape == 3:
                # first output of the list (should be one-element list anyway)
                controller_output = controller_output[:, 0, :]

        else: # use dense controller
            controller_output = self.controller(controller_input)
        # print("controller_output", controller_output)
        return controller_output

    def step(self, x, states):

        controller_output_tm1, M_tm1, Keys_tm1 = states
        # The step function will slice the second dim (supposed to be timesteps in most cases) and only pass one piece
        # so if we need to keep all dimensions and pass it to next time step we need to reshape it
        M_tm1 = K.reshape(M_tm1, (self.batch_size, self.n_slots, self.m_depth))
        Keys_tm1 = K.reshape(Keys_tm1, (self.batch_size, self.n_slots, self.key_range))

        # read
        inputs_read = K.concatenate([x, controller_output_tm1])
        _, beta_read, g_read, gamma_read, s_read = self._get_controller_output(inputs_read, None, None,
                                    self.W_c, self.b_c, self.W_s, self.b_s, heads=self.read_heads)
        k_read = x[:, :self.key_range]
        # get read weights from keys
        wc_read = self._get_content_w(beta_read, k_read, Keys_tm1)
        wr_t = self._get_raw_location_w(gamma_read, wc_read)
        M_read = self._read(wr_t, M_tm1)

        # update controller
        controller_output_t = self._update_controller(x, M_read)

        # write
        inputs_write = K.concatenate([x, controller_output_t])
        _, beta_write, g_write, gamma_write, s_write = self._get_controller_output(inputs_write,
                                    None, None, self.W_c, self.b_c, self.W_s,
                                    self.b_s, heads=self.write_heads)
        k_write = k_read
        wc_write = self._get_content_w(beta_write, k_write, Keys_tm1)
        ww_t = self._get_ungated_location_w(s_write, self.C, gamma_write, wc_write)

        # M_t = self._direct_write(ww_t, controller_output_t, M_tm1)
        M_t = self._direct_write(ww_t, x, M_tm1)
        M_t = K.batch_flatten(M_t)

        Keys_t = self._direct_write(ww_t, k_write, Keys_tm1)
        Keys_t = K.batch_flatten(Keys_t)

        return controller_output_t, [controller_output_t, M_t, Keys_t]

    # def step(self, x, states):
    #
    #     controller_output_tm1, M_tm1, Keys_tm1 = states
    #
    #     # The step function will slice the second dim (supposed to be timesteps in most cases) and only pass one piece
    #     # so if we need to keep all dimensions and pass it to next time step we need to reshape it
    #     M_tm1 = K.reshape(M_tm1, (self.batch_size, self.n_slots, self.m_depth))
    #     Keys_tm1 = K.reshape(Keys_tm1, (self.batch_size, self.n_slots, self.key_range))
    #
    #     # read
    #     inputs_read = K.concatenate([x, controller_output_tm1])
    #     _, beta_read, g_read, gamma_read, s_read = self._get_controller_output(inputs_read, None, None,
    #                                 self.W_c, self.b_c, self.W_s, self.b_s, heads=self.read_heads)
    #     k_read = x[:, :self.key_range]
    #     # get read weights from keys
    #     wc_read = self._get_content_w(beta_read, k_read, Keys_tm1)
    #     wr_t = self._get_raw_location_w(gamma_read, wc_read)
    #     M_read = self._read(wr_t, M_tm1)
    #
    #     # update controller
    #     controller_output_t = self._update_controller(x, M_read)
    #
    #     # write
    #     inputs_write = K.concatenate([x, controller_output_t])
    #     _, beta_write, g_write, gamma_write, s_write = self._get_controller_output(inputs_write,
    #                                 None, None, self.W_c, self.b_c, self.W_s,
    #                                 self.b_s, heads=self.write_heads)
    #     k_write = k_read
    #     wc_write = self._get_content_w(beta_write, k_write, Keys_tm1)
    #     ww_t = self._get_ungated_location_w(s_write, self.C, gamma_write, wc_write)
    #
    #     M_t = self._direct_write(ww_t, controller_output_t, M_tm1)
    #     M_t = K.batch_flatten(M_t)
    #
    #     Keys_t = self._direct_write(ww_t, k_write, Keys_tm1)
    #     Keys_t = K.batch_flatten(Keys_t)
    #
    #     ntm_output_t = K.concatenate([controller_output_t, M_read])
    #
    #     return ntm_output_t, [controller_output_t, M_t, Keys_t]
    #
    # def compute_output_shape(self, input_shape):
    #     if isinstance(input_shape, list):
    #         input_shape = input_shape[0]
    #     if self.return_sequences:
    #         return (input_shape[0], input_shape[1], self.units + self.m_depth)
    #     else:
    #         return (input_shape[0], self.units + self.m_depth)
