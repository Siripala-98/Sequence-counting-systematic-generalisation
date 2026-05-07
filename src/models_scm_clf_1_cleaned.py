"""
Model Architectures
====================
This module defines all neural network architectures used in the experiments.
Models are organised into three groups:

1. Vanilla baselines (no counter module)
   - VanillaRNN, VanillaGRU, VanillaLSTM

2. DNNC models (with Discrete Non-Negative Counter)
   - StackRNN, StackGRU, StackLSTM

3. DFRC models (with Discrete Full Range Counter)
   - StackRNN_DFRC, StackGRU_DFRC, StackLSTM_DFRC

4. Pure counter models (counter used as sole recurrent unit, no RNN backbone)
   - RecurrentStack (DNNC-only), RecurrentDFRC (DFRC-only)

All models share the same input/output interface:
    - Input: one-hot encoded bracket token, shape [input_size]
    - Output: binary classification (valid/invalid Dyck-1 sequence)

Counter models split the linear projection into two streams:
    - First hidden_size dimensions → fed into the RNN backbone
    - Last 2 dimensions → fed into the counter module
The outputs are concatenated before the final classification layer.

References:
    El-Naggar, N. (2023). What Really Counts: Theoretical and Practical Aspects
    of Counting Behaviour in Simple RNNs.
    https://github.com/nadineelnaggar/DNNC
"""

import torch
import torch.nn as nn
from dnnc import StackCounterNN
from dfrc import StackCounterDFRC_NN

device = 'cpu'


# =============================================================================
# Vanilla baselines (no counter module)
# =============================================================================

class VanillaRNN(nn.Module):
    """
    Plain RNN baseline — no counter module.

    Architecture: Linear → RNN → Linear → Sigmoid
    Used to establish how well a standard RNN learns to count from data alone,
    without any inductive bias toward discrete counting.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 output_activation='Sigmoid'):
        super(VanillaRNN, self).__init__()
        self.model_name = 'VanillaRNN'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, hidden):
        x = self.fc1(x)
        x, hidden = self.rnn(x.unsqueeze(0).unsqueeze(0), hidden)
        x = self.fc2(x.squeeze())
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x)
        elif self.output_activation == 'Softmax':
            x = self.softmax(x)
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1)
        return x, hidden


class VanillaGRU(nn.Module):
    """
    Plain GRU baseline — no counter module.

    Architecture: Linear → GRU → Linear → Sigmoid
    GRUs have a gating mechanism that gives them more capacity to track long-range
    dependencies than a vanilla RNN, making this a stronger baseline.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 output_activation='Sigmoid'):
        super(VanillaGRU, self).__init__()
        self.model_name = 'VanillaGRU'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, hidden):
        x = self.fc1(x)
        x, hidden = self.gru(x.unsqueeze(0).unsqueeze(0), hidden)
        x = self.fc2(x.squeeze())
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x)
        elif self.output_activation == 'Softmax':
            x = self.softmax(x)
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1)
        return x, hidden


class VanillaLSTM(nn.Module):
    """
    Plain LSTM baseline — no counter module.

    Architecture: Linear → LSTM → Linear → Sigmoid
    LSTMs have explicit memory cells which give them the strongest baseline
    capacity for tracking counts among the three vanilla architectures.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 output_activation='Sigmoid'):
        super(VanillaLSTM, self).__init__()
        self.model_name = 'VanillaLSTM'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, hidden):
        x = self.fc1(x)
        x, hidden = self.lstm(x.unsqueeze(0).unsqueeze(0), hidden)
        x = self.fc2(x.squeeze())
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x)
        elif self.output_activation == 'Softmax':
            x = self.softmax(x)
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1)
        return x, hidden


# =============================================================================
# DNNC models (RNN backbone + Discrete Non-Negative Counter)
# =============================================================================

class StackRNN(nn.Module):
    """
    RNN augmented with a DNNC counter module.

    The linear projection splits into two streams:
        - First hidden_size dims → RNN backbone
        - Last 2 dims → DNNC counter
    Their outputs are concatenated and passed to the classifier.

    This architecture allows the RNN to learn sequence features while the
    counter provides a discrete, exact counting signal.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 stack_input_size=2, stack_output_size=2, output_activation='Sigmoid'):
        super(StackRNN, self).__init__()
        self.model_name = 'StackRNN'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size + stack_input_size)
        self.stack = StackCounterNN()
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size + stack_output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, hidden, stack_state):
        x = self.fc1(x)
        rnn_output, hidden = self.rnn(x[:self.hidden_size].unsqueeze(0).unsqueeze(0), hidden)
        stack_state = self.stack(x[self.hidden_size:].squeeze(), stack_state)
        combined = torch.cat((stack_state, rnn_output.squeeze()), dim=0)
        x = self.fc2(combined)
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x).squeeze()
        elif self.output_activation == 'Softmax':
            x = self.softmax(x).squeeze()
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1).squeeze()
        return x, hidden, stack_state


class StackGRU(nn.Module):
    """
    GRU augmented with a DNNC counter module.

    Same split-stream architecture as StackRNN, with a GRU backbone.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 stack_input_size=2, stack_output_size=2, output_activation='Sigmoid'):
        super(StackGRU, self).__init__()
        self.model_name = 'StackGRU'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size + stack_input_size)
        self.stack = StackCounterNN()
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size + stack_output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, hidden, stack_state):
        x = self.fc1(x)
        gru_output, hidden = self.gru(x[:self.hidden_size].unsqueeze(0).unsqueeze(0), hidden)
        stack_state = self.stack(x[self.hidden_size:].squeeze(), stack_state)
        combined = torch.cat((stack_state, gru_output.squeeze()), dim=0)
        x = self.fc2(combined)
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x).squeeze()
        elif self.output_activation == 'Softmax':
            x = self.softmax(x).squeeze()
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1).squeeze()
        return x, hidden, stack_state


class StackLSTM(nn.Module):
    """
    LSTM augmented with a DNNC counter module.

    Same split-stream architecture as StackRNN, with an LSTM backbone.
    The LSTM's forget gate bias is initialised to 1 to encourage long-term
    memory retention, which aids counting over longer sequences.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 stack_input_size=2, stack_output_size=2, output_activation='Sigmoid'):
        super(StackLSTM, self).__init__()
        self.model_name = 'StackLSTM'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size + stack_input_size)
        self.stack = StackCounterNN()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size + stack_output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self._init_forget_gate()

    def _init_forget_gate(self):
        """Initialise LSTM forget gate bias to 1 for better long-range retention."""
        with torch.no_grad():
            for names in self.lstm._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.lstm, name)
                    n = bias.size(0)
                    bias[n // 4: n // 2].fill_(1)

    def forward(self, x, hidden, stack_state):
        x = self.fc1(x)
        lstm_output, hidden = self.lstm(x[:self.hidden_size].unsqueeze(0).unsqueeze(0), hidden)
        stack_state = self.stack(x[self.hidden_size:].squeeze(), stack_state)
        combined = torch.cat((stack_state, lstm_output.squeeze()), dim=0)
        x = self.fc2(combined)
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x).squeeze()
        elif self.output_activation == 'Softmax':
            x = self.softmax(x).squeeze()
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1).squeeze()
        return x, hidden, stack_state


# =============================================================================
# DFRC models (RNN backbone + Discrete Full Range Counter)
# =============================================================================

class StackRNN_DFRC(nn.Module):
    """
    RNN augmented with a DFRC counter module.

    Identical split-stream architecture to StackRNN, but the counter is
    replaced with the DFRC, which supports negative counts and uses a
    tanh-based category signal for smoother gradient flow.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 stack_input_size=2, stack_output_size=2, output_activation='Sigmoid'):
        super(StackRNN_DFRC, self).__init__()
        self.model_name = 'StackRNN_DFRC'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size + stack_input_size)
        self.stack = StackCounterDFRC_NN()
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size + stack_output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, hidden, stack_state):
        x = self.fc1(x)
        rnn_output, hidden = self.rnn(x[:self.hidden_size].unsqueeze(0).unsqueeze(0), hidden)
        stack_state = self.stack(x[self.hidden_size:].squeeze(), stack_state)
        combined = torch.cat((stack_state, rnn_output.squeeze()), dim=0)
        x = self.fc2(combined)
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x).squeeze()
        elif self.output_activation == 'Softmax':
            x = self.softmax(x).squeeze()
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1).squeeze()
        return x, hidden, stack_state


class StackGRU_DFRC(nn.Module):
    """
    GRU augmented with a DFRC counter module.

    Same split-stream architecture as StackRNN_DFRC, with a GRU backbone.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 stack_input_size=2, stack_output_size=2, output_activation='Sigmoid'):
        super(StackGRU_DFRC, self).__init__()
        self.model_name = 'StackGRU_DFRC'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size + stack_input_size)
        self.stack = StackCounterDFRC_NN()
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size + stack_output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, hidden, stack_state):
        x = self.fc1(x)
        gru_output, hidden = self.gru(x[:self.hidden_size].unsqueeze(0).unsqueeze(0), hidden)
        stack_state = self.stack(x[self.hidden_size:].squeeze(), stack_state)
        combined = torch.cat((stack_state, gru_output.squeeze()), dim=0)
        x = self.fc2(combined)
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x).squeeze()
        elif self.output_activation == 'Softmax':
            x = self.softmax(x).squeeze()
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1).squeeze()
        return x, hidden, stack_state


class StackLSTM_DFRC(nn.Module):
    """
    LSTM augmented with a DFRC counter module.

    Same split-stream architecture as StackRNN_DFRC, with an LSTM backbone.
    This is the best-performing configuration in the experiments, achieving
    99.84% accuracy on long sequences (100 tokens) on the large dataset.

    The LSTM forget gate bias is initialised to 1 as in StackLSTM.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 stack_input_size=2, stack_output_size=2, output_activation='Sigmoid'):
        super(StackLSTM_DFRC, self).__init__()
        self.model_name = 'StackLSTM_DFRC'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size + stack_input_size)
        self.stack = StackCounterDFRC_NN()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size + stack_output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self._init_forget_gate()

    def _init_forget_gate(self):
        """Initialise LSTM forget gate bias to 1 for better long-range retention."""
        with torch.no_grad():
            for names in self.lstm._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.lstm, name)
                    n = bias.size(0)
                    bias[n // 4: n // 2].fill_(1)

    def forward(self, x, hidden, stack_state):
        x = self.fc1(x)
        lstm_output, hidden = self.lstm(x[:self.hidden_size].unsqueeze(0).unsqueeze(0), hidden)
        stack_state = self.stack(x[self.hidden_size:].squeeze(), stack_state)
        combined = torch.cat((stack_state, lstm_output.squeeze()), dim=0)
        x = self.fc2(combined)
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x).squeeze()
        elif self.output_activation == 'Softmax':
            x = self.softmax(x).squeeze()
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1).squeeze()
        return x, hidden, stack_state


# =============================================================================
# Pure counter models (counter as sole recurrent unit)
# =============================================================================

class RecurrentStack(nn.Module):
    """
    Counter-only model using DNNC — no RNN backbone.

    The entire recurrent computation is handled by the DNNC counter.
    This tests whether a discrete counter alone, without a learned hidden state,
    is sufficient for Dyck-1 classification.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 stack_input_size=2, stack_output_size=2, output_activation='Sigmoid',
                 bias=False):
        super(RecurrentStack, self).__init__()
        self.model_name = 'RecurrentStack'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.stack = StackCounterNN()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, stack_state):
        x = self.fc1(x)
        x = self.stack(x.squeeze(), stack_state)
        stack_state = x
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(self.fc2(x)).squeeze()
        elif self.output_activation == 'Softmax':
            x = self.softmax(self.fc2(x)).squeeze()
        elif self.output_activation == 'Clipping':
            x = torch.clamp(self.fc2(x), min=0, max=1).squeeze()
        elif self.output_activation == 'None':
            return x, stack_state
        return x, stack_state


class RecurrentDFRC(nn.Module):
    """
    Counter-only model using DFRC — no RNN backbone.

    Identical to RecurrentStack but uses the DFRC counter, which supports
    negative counts. Tests whether the DFRC's extended range improves
    classification without any RNN backbone to assist.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 stack_input_size=2, stack_output_size=2, output_activation='Sigmoid',
                 bias=False):
        super(RecurrentDFRC, self).__init__()
        self.model_name = 'RecurrentDFRC'
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.stack = StackCounterDFRC_NN()
        self.fc2 = nn.Linear(hidden_size + 2, output_size, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, stack_state):
        x = self.fc1(x)
        x = self.stack(x.squeeze(), stack_state)
        stack_state = x
        if self.output_activation == 'Sigmoid':
            if x.dim() > 0 and (x.dim() > 1 or x.size(0) > 1):
                x = self.sigmoid(self.fc2(x)).squeeze()
            else:
                x = self.sigmoid(self.fc2(x))
        elif self.output_activation == 'Softmax':
            if x.dim() > 0 and (x.dim() > 1 or x.size(0) > 1):
                x = self.softmax(self.fc2(x)).squeeze()
            else:
                x = self.softmax(self.fc2(x))
        elif self.output_activation == 'Clipping':
            if x.dim() > 0 and (x.dim() > 1 or x.size(0) > 1):
                x = torch.clamp(self.fc2(x), min=0, max=1).squeeze()
            else:
                x = torch.clamp(self.fc2(x), min=0, max=1)
        elif self.output_activation == 'None':
            return x, stack_state
        return x, stack_state
