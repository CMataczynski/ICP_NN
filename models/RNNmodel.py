from torch import nn
import torch.nn.functional as F
import torch


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.bidirectional = bidirectional
        if bidirectional:
            self.lstm = nn.LSTM(input_size, hidden_layer_size//2, batch_first=True, bidirectional=True)
            self.hidden_cell = (torch.zeros(2, 1, self.hidden_layer_size//2),
                                torch.zeros(2, 1, self.hidden_layer_size//2))
        else:
            self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                                torch.zeros(1, 1, self.hidden_layer_size))

        self.linear = nn.Linear(hidden_layer_size, output_size)


    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]