from torch import nn
import torch.nn.functional as F
import torch


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=None, output_size=4, bidirectional=False):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        if hidden_layer_size is None:
            self.hidden_layer_size = output_size
        self.bidirectional = bidirectional
        if bidirectional:
            if hidden_layer_size is not None:
                self.lstm = nn.LSTM(input_size, self.hidden_layer_size//2,
                                    num_layers= 2,
                                    batch_first=True, bidirectional=True)
            else:
                self.lstm = nn.LSTM(input_size, self.hidden_layer_size,
                                    num_layers= 2,
                                    batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size, self.hidden_layer_size,
                                num_layers=2,
                                batch_first=True)
        if hidden_layer_size is not None:
            self.linear = nn.Linear(hidden_layer_size,output_size)
        else:
            self.linear = None



    def forward(self, input_seq):
        input_seq = input_seq.view(-1, len(input_seq), 1)
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        if self.linear is not None:
            predictions = self.linear(torch.reshape(lstm_out[-1], (-1, self.hidden_layer_size)))
        else:
            predictions = lstm_out[-1]
        #print(predictions.shape)
        return predictions