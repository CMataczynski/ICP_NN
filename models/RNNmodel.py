import torch
from torch import nn

from models.models_util import BlockLSTM, BlockFCN


class LSTM(nn.Module):
    def __init__(self, input_size=180, hidden_layer_size=None, output_size=4, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        if hidden_layer_size is None:
            self.hidden_layer_size = output_size
        self.bidirectional = bidirectional
        if bidirectional:
            if hidden_layer_size is not None:
                self.lstm = nn.LSTM(input_size, self.hidden_layer_size // 2,
                                    num_layers=1,
                                    batch_first=True, bidirectional=True)
            else:
                self.lstm = nn.LSTM(input_size, self.hidden_layer_size,
                                    num_layers=1,
                                    batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size, self.hidden_layer_size,
                                num_layers=1,
                                batch_first=True)
        if hidden_layer_size is not None:
            self.linear = nn.Linear(hidden_layer_size, output_size)
        else:
            self.linear = None

    def forward(self, input_seq):
        input_seq = input_seq.unsqueeze(1)
        input_seq = torch.transpose(input_seq, 0, 1)
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        if self.linear is not None:
            predictions = self.linear(torch.reshape(lstm_out[-1], (-1, self.hidden_layer_size)))
        else:
            predictions = lstm_out[-1]
        return predictions


class GRU(nn.Module):
    def __init__(self, input_size=180, hidden_layer_size=None, output_size=4, bidirectional=False):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        if hidden_layer_size is None:
            self.hidden_layer_size = output_size
        self.bidirectional = bidirectional
        if bidirectional:
            if hidden_layer_size is not None:
                self.gru = nn.GRU(input_size, self.hidden_layer_size // 2,
                                  num_layers=2,
                                  batch_first=True, bidirectional=True)
            else:
                self.gru = nn.GRU(input_size, self.hidden_layer_size,
                                  num_layers=2,
                                  batch_first=True, bidirectional=True)
        else:
            self.gru = nn.GRU(input_size, self.hidden_layer_size,
                              num_layers=2,
                              batch_first=True)
        if hidden_layer_size is not None:
            self.linear = nn.Linear(hidden_layer_size, output_size)
        else:
            self.linear = None

    def forward(self, input_seq):
        input_seq = input_seq.unsqueeze(1)
        input_seq = torch.transpose(input_seq, 0, 1)
        gru_out, self.hidden_cell = self.gru(input_seq)
        if self.linear is not None:
            predictions = self.linear(torch.reshape(gru_out[-1], (-1, self.hidden_layer_size)))
        else:
            predictions = gru_out[-1]
        return predictions


class LSTMFCN(nn.Module):
    def __init__(self, time_steps, num_variables=1, lstm_hs=64, channels=[1, 32, 64, 32]):
        super().__init__()
        self.lstm_block = BlockLSTM(time_steps, 1, lstm_hs)
        self.fcn_block = BlockFCN(time_steps)
        self.dense = nn.Linear(channels[-1] + lstm_hs, num_variables)
        self.softmax = nn.LogSoftmax(dim=1)  # nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.lstm_block(x)
        x1 = torch.squeeze(x1)
        x2 = self.fcn_block(x)
        x2 = torch.squeeze(x2)
        x = torch.cat([x1, x2], 1)
        x = self.dense(x)
        y = self.softmax(x)
        return y
