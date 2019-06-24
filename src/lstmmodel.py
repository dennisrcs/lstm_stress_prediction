"""LSTM model used in our experiments"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from torch.nn import functional as F
from torch.autograd import Variable

class LSTMModel(nn.Module):
    """ LSTM with 2 layers followed by two fully-connected layers """

    def __init__(self, input_size, output_size, hidden_size, num_layers, batch_size, sequence_size):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.sequences_size = np.zeros(batch_size) + sequence_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)

        #self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(input_size, input_size) # temporary
        #self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1 = nn.BatchNorm1d(input_size) # temporary
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)

    def forward(self, inputs):
        """ Method used for forward propagation """

        hidden, cell = self.init_hidden()

        # First fully connected layer
        output = inputs.view(-1, inputs.shape[2])
        output = self.fc1(output)
        output = F.relu(self.bn1(output))

        # Changing back to (16, 5, 52) shape
        output = output.view(inputs.shape)

        # LSTM part
        packed_inputs = rnn.pack_padded_sequence(inputs, self.sequences_size, batch_first=True)
        packed_output, (hidden, cell) = self.rnn(packed_inputs, (hidden, cell))
        output, _ = rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Fully-connect part

        # Reshaping 16 x 5 dimensional tensor into 80 dimensional tensor
        output = output.contiguous().view(-1, self.hidden_size)

        # Second fully connected layer
        output = self.fc2(output)
        output = F.relu(self.bn2(output))

        return output

    def init_hidden(self):
        """ Initializess hidden and cell states """
        # initialize hidden state
        hidden = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

        return hidden, cell
