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

    def write(self, w, a_k, a):
        """write to memory"""
        self.prev_content = self.content
        # self.content = torch.Tensor(self.batch_size, self.N, self.M)
        self.content = self.prev_content + torch.matmul(w.unsqueeze(1), (a.unsqueeze(1) - self.prev_content))

        self.prev_keys = self.keys
        # self.keys = torch.Tensor(self.batch_size, self.N, self.K)
        self.keys = self.prev_keys + torch.matmul(w.unsqueeze(1), (a_k.unsqueeze(1) - self.prev_keys))

    def address(self, k, β, g, s, γ):
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
        wc = self._similarity(k, β)

        # Location focus
        # wg = self._interpolate(w_prev, wc, g)
        # ŵ = self._shift(wg, s)
        ŵ = self._shift(wc, s)
        w = self._sharpen(ŵ, γ)

        return w

    def _similarity(self, k, β):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(β * F.cosine_similarity(self.keys + 1e-16, k + 1e-16, dim=-1), dim=1)  # use keys for addressing
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
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w


class GCLReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(GCLReadHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_lengths = [1, 1, 3, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        w = self.memory.address(k, β, g, s, γ)

        return w

    def forward(self, k, embeddings):
        """NTMReadHead forward function.

        :param embeddings: feeds from controller.
        :param w_prev: previous step state
        """
        o = self.fc_read(embeddings)
        β, g, s, γ = _split_cols(o, self.read_lengths)

        # Read from memory
        w = self._address_memory(k, β, g, s, γ)
        r = self.memory.read(w)

        # print("read weights", w)
        return r, w


class GCLWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(GCLWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        self.write_lengths = [1, 1, 3, 1]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        w = self.memory.address(k, β, g, s, γ)
        # print("write weights", w)

        return w

    def forward(self, k, embeddings):
        """NTMWriteHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_write(embeddings)
        β, g, s, γ = _split_cols(o, self.write_lengths)

        # Write to memory
        w = self._address_memory(k, β, g, s, γ)
        self.memory.write(w, k, embeddings)

        return w


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

    def forward(self, k, x, prev_state):
        """NTM forward function.

        :param x: input vector (batch_size x input_dim)
        :param prev_state: The previous state of the NTM
        """
        # Unpack the previous state
        prev_reads, prev_controller_state, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        inp = torch.cat([x] + prev_reads, dim=1)
        # (lstm controller) lstm outout, lstm state
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                r, head_state = head(k, controller_outp)
                reads += [r]
            else:
                head_state = head(k, controller_outp)
            heads_states += [head_state]

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = F.sigmoid(self.fc(inp2))

        # Pack the current state
        state = (reads, controller_state, heads_states)

        return o, state


class EncapsulatedGCL(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M, K):
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
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M
        self.K = K
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the NTM components
        memory = GCLMemory(N, M, K)
        controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers)
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

    def forward(self, k=None, x=None):
        """self.state updates itself, so no need to feed states with input"""
        if x is None:
            x = torch.zeros(self.batch_size, 1, self.num_inputs).to(self.device)
        if k is None:
            k = torch.zeros(self.batch_size, 1, self.K).to(self.device)

        time_steps = x.shape[1]
        outputs = []
        for t in range(time_steps):
            o, self.previous_state = self.gcl(k[:,t,:], x[:,t,:], self.previous_state)
            outputs.append(o)

        return torch.stack(outputs, dim=1)

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params