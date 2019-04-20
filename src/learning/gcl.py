import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys
import os

NTM_PATH = os.environ['HOME'] + '/projects/pytorch-ntm/ntm'
if NTM_PATH not in sys.path:
    sys.path.append(NTM_PATH)

from ntm import NTM
from controller import LSTMController
from head import NTMHeadBase, NTMReadHead, NTMWriteHead
from memory import NTMMemory

def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class GCLMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M, K):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        :param K: Number of columns/features in the keys.
        """
        super(GCLMemory, self).__init__()

        self.N = N
        self.M = M
        self.K = K
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('content_bias', torch.Tensor(N, M))
        self.register_buffer('key_bias', torch.Tensor(N, K))

        # Initialize memory bias
        stdev_M = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.content_bias, -stdev_M, stdev_M)
        stdev_K = 1 / (np.sqrt(N + K))
        nn.init.uniform_(self.key_bias, -stdev_K, stdev_K)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.content = self.content_bias.clone().repeat(batch_size, 1, 1)
        self.keys = self.key_bias.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.content).squeeze(1)

    def write(self, w, a_k, a, time_step=0, time_threshold=0):
        """write to memory"""
        if time_step < time_threshold:
            w = torch.zeros(w.shape).to(self.device)
            w[:,time_step % self.N] = 1

        self.prev_content = self.content
        # self.content = torch.Tensor(self.batch_size, self.N, self.M)
        self.content = self.prev_content + w.unsqueeze(-1) * (a.unsqueeze(1) - self.prev_content)

        self.prev_keys = self.keys
        # self.keys = torch.Tensor(self.batch_size, self.N, self.K)
        self.keys = self.prev_keys + w.unsqueeze(-1) * (a_k.unsqueeze(1) - self.prev_keys)

        # print(w, self.keys.shape)
        # print(self.prev_keys)
        # print(self.keys)

    def flip_keys(self, k):
        key_len = k.shape[-1] // 2
        left, right = _split_cols(k, [key_len, key_len])
        return torch.cat([right, left], -1)

    # def address(self, k, β, g, s, γ):
    #     """Use keys to compute addresses (attention vector).
    #
    #     Returns a softmax weighting over the rows of the memory matrix.
    #
    #     :param k: The key vector.
    #     :param β: The key strength (focus).
    #     :param g: Scalar interpolation gate (with previous weighting).
    #     :param s: Shift weighting.
    #     :param γ: Sharpen weighting scalar.
    #     :param w_prev: The weighting produced in the previous time step.
    #     """
    #     # Content focus
    #     # print("received k", k)
    #     # print("self.keys", self.keys)
    #     # print('_____')
    #
    #     wc_1 = self._similarity(k, β)
    #     wc_2 = self._similarity(self.flip_keys(k), β)
    #     wc = F.max_pool1d(torch.stack([wc_1, wc_2], -1), 2).squeeze(-1)
    #     # print(wc_1, torch.max(wc_1))
    #     # print(wc_2, torch.max(wc_2))
    #     # wc = F.softmax(wc, dim=1)
    #     wc = torch.div(wc, wc.sum(dim=-1, keepdim=True))
    #     # print(wc)
    #
    #     # wc = self._similarity(k, β)
    #     # Location focus
    #     # wg = self._interpolate(w_prev, wc, g)
    #     # ŵ = self._shift(wg, s)
    #
    #     ŵ = self._shift(wc, s)
    #     w = self._sharpen(ŵ, γ)
    #
    #     # w = self._sharpen(wc, γ)
    #
    #     return w

    def address(self, k, β, g, s, γ, candidates=5, flip_keys=False):
        """Use keys to compute addresses (attention vector).

        Returns a softmax weighting over the rows of the memory matrix.

        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        # Content focus
        # print("received k", k)
        # print("self.keys", self.keys)
        # print('_____')

        if flip_keys:
            wc_1 = self._similarity(k, β)
            wc_2 = self._similarity(self.flip_keys(k), β)
            wc = F.max_pool1d(torch.stack([wc_1, wc_2], -1), 2).squeeze(-1)
        else:
            wc = self._similarity(k, β)
        # print(wc)

        # pick the non-top candidates
        used_slots, used_indexes = torch.topk(wc, candidates, largest=True)
        batch_size = wc.shape[0]
        masks = []
        for i in range(batch_size):
            # print(i, used_indexes[i])
            mask = torch.zeros(self.N).to(self.device) + 1e-16
            mask[used_indexes[i]] = 1
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        wc = wc * masks

        wc = torch.div(wc, wc.sum(dim=-1, keepdim=True))
        # print(wc)

        # wc = self._similarity(k, β)
        # Location focus
        # wg = self._interpolate(w_prev, wc, g)
        # ŵ = self._shift(wg, s)

        # ŵ = self._shift(wc, s)
        # w = self._sharpen(ŵ, γ)

        # print(w)
        w = self._sharpen(wc, γ)

        return w

    def _similarity(self, k, β):
        k = k.view(self.batch_size, 1, -1)

        # cosine distance
        # w = F.softmax(β * F.cosine_similarity(self.keys + 1e-16, k + 1e-16, dim=-1), dim=1)  # use keys for addressing
        w = F.softmax(β * F.cosine_similarity(self.keys, k, dim=-1), dim=1)

        # euclidean distance
        # distances = F.pairwise_distance(self.keys.permute(0, 2, 1), k.repeat(1, self.N, 1).permute(0, 2, 1), p=2)
        # norm_distances = F.normalize(distances, p=2, dim=1)
        # # print(distances, norm_distances)
        # w = F.softmax(β * (1-norm_distances), dim=1)

        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = torch.zeros(wg.size()).to(self.device)
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        # w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1))
        return w


class GCLReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(GCLReadHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_lengths = [1, 1, 3, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.attn_array = None

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        return torch.zeros(batch_size, self.N).to(self.device)

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def _address_memory(self, k, β, g, s, γ):
        # Handle Activations
        k = k.clone()
        β = F.softplus(β)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)
        # print("read beta", β)
        w = self.memory.address(k, β, g, s, γ, candidates=5, flip_keys=True)
        # print(w)
        return w

    def forward(self, k, embeddings, save_attn=False):
        """NTMReadHead forward function.

        :param embeddings: feeds from controller.
        :param w_prev: previous step state
        """
        o = self.fc_read(embeddings)
        β, g, s, γ = _split_cols(o, self.read_lengths)

        # Read from memory
        w = self._address_memory(k, β, g, s, γ)
        r = self.memory.read(w)

        # save read attention
        # print(save_attn)
        if save_attn:
            if self.attn_array is None:
                self.attn_array = w.data.cpu().numpy()
            else:
                self.attn_array = np.append(self.attn_array, w.data.cpu().numpy(), axis=0)

        # print("read weights", w)
        return r, w

    def save_attn_vectors(self, path):
        np.save(path, self.attn_array)
        self.attn_array = None

class GCLWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(GCLWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        self.write_lengths = [1, 1, 3, 1]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.attn_array = None

    def create_new_state(self, batch_size):
        return torch.zeros(batch_size, self.N).to(self.device)

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def _address_memory(self, k, β, g, s, γ):
        # Handle Activations
        k = k.clone()
        β = F.softplus(β)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)
        # print("write beta", β)
        w = self.memory.address(k, β, g, s, γ, candidates=3, flip_keys=False)
        # print("write weights", w)

        return w

    def forward(self, k, embeddings, t, save_attn=False):
        """NTMWriteHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_write(embeddings)
        β, g, s, γ = _split_cols(o, self.write_lengths)

        # Write to memory
        w = self._address_memory(k, β, g, s, γ)
        self.memory.write(w, k, embeddings, time_step=t, time_threshold=self.N)

        # save write attention
        if save_attn:
            if self.attn_array is None:
                self.attn_array = w.data.cpu().numpy()
            else:
                self.attn_array = np.append(self.attn_array, w.data.cpu().numpy(), axis=0)

        return w

    def save_attn_vectors(self, path):
        np.save(path, self.attn_array)
        self.attn_array = None


class GCL(NTM):

    def __init__(self, num_inputs, num_outputs, controller, memory, heads):
        """Initialize the NTM.

        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`LSTMController`
        :param memory: :class:`GCLMemory`
        :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`

        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(GCL, self).__init__(num_inputs, num_outputs, controller, memory, heads)
        # Save arguments

        self.K = memory.K  # key size
        self.controller_type = type(controller)

    def create_new_state(self, batch_size):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        heads_state = [head.create_new_state(batch_size) for head in self.heads]

        if self.controller_type == LSTMController:
            controller_state = self.controller.create_new_state(batch_size)
            return init_r, controller_state, heads_state

        return init_r, heads_state

    def forward(self, k, x, prev_state, t, save_attn=False):
        """NTM forward function.

        :param x: input vector (batch_size x input_dim)
        :param prev_state: The previous state of the NTM
        """
        # Unpack the previous state
        if self.controller_type == LSTMController:
            prev_reads, prev_controller_state, prev_heads_states = prev_state
        else:
            prev_reads, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        inp = torch.cat([x] + prev_reads, dim=1)

        if self.controller_type == LSTMController:
            # (lstm controller) lstm outout, lstm state
            controller_outp, controller_state = self.controller(inp, prev_controller_state)
        else:
            controller_outp = self.controller(inp)

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        # print(save_attn)
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                r, head_state = head(k, controller_outp, save_attn=save_attn)
                reads += [r]
            else:
                head_state = head(k, controller_outp, t, save_attn=save_attn)
            heads_states += [head_state]

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        # o = F.sigmoid(self.fc(inp2))
        o = F.relu(self.fc(inp2))

        # Pack the current state
        if self.controller_type == LSTMController:
            state = (reads, controller_state, heads_states)
        else:
            state = (reads, heads_states)

        return o, state


class EncapsulatedGCL(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M, K, controller_type='MLP'):
        """Initialize an EncapsulatedNTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        :param K: Number of cols/features in the keys.
        """
        super(EncapsulatedGCL, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_type = controller_type
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M
        self.K = K
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the NTM components
        memory = GCLMemory(N, M, K)
        if controller_type == 'LSTM':
            controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers)
        else:
            controller = MLPController(num_inputs + M*num_heads, controller_size)

        heads = nn.ModuleList([])
        for i in range(num_heads):
            heads += [
                GCLReadHead(memory, controller_size),
                GCLWriteHead(memory, controller_size)
            ]

        self.gcl = GCL(num_inputs, num_outputs, controller, memory, heads).to(self.device)
        self.memory = memory.to(self.device)

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.gcl.create_new_state(batch_size)

    def forward(self, k=None, x=None, bidirectional=False, save_attn=False):
        """self.state updates itself, so no need to feed states with input"""
        if x is None:
            x = torch.zeros(self.batch_size, 1, self.num_inputs).to(self.device)
        if k is None:
            k = torch.zeros(self.batch_size, 1, self.K).to(self.device)

        batch_size = x.shape[0]
        time_steps = x.shape[1]
        self.init_sequence(batch_size)

        if bidirectional:
            # warm up
            for t in range(time_steps):
                o, self.previous_state = self.gcl(k[:, t, :], x[:, t, :], self.previous_state, t, save_attn=False)

            # self.init_sequence(self.batch_size)
            backward_outputs = []
            for t in range(time_steps):
                o, self.previous_state = self.gcl(k[:, time_steps-t-1, :], x[:, time_steps-t-1, :],
                                                  self.previous_state, t+time_steps, save_attn=save_attn)
                backward_outputs.append(o)
            backward_outputs.reverse()
            backward_outputs = torch.stack(backward_outputs, dim=1)

            outputs = []
            for t in range(time_steps):
                o, self.previous_state = self.gcl(k[:, t, :], x[:, t, :], self.previous_state, t + time_steps,
                                                  save_attn=save_attn)
                outputs.append(o)
            outputs = torch.stack(outputs, dim=1)

            outputs = torch.cat([outputs, backward_outputs], -1)

        else: # no warm up
            outputs = []
            for t in range(time_steps):
                o, self.previous_state = self.gcl(k[:, t, :], x[:, t, :], self.previous_state, t,
                                                  save_attn=save_attn)
                outputs.append(o)
            outputs = torch.stack(outputs, dim=1)

        return outputs

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

class MLPController(nn.Module):
    """An NTM controller based on MLP."""
    def __init__(self, num_inputs, num_outputs):
        super(MLPController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        # self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.linear = nn.Linear(num_inputs, num_outputs)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.linear.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x):
        outp = F.relu(self.linear(x))
        return outp