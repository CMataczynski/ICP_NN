import torch
from torch import nn

from models.models_util import BlockLSTM, BlockFCN, ResBlock, conv1x1, norm


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=None, output_size=4, bidirectional=False):
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
        input_seq = input_seq.unsqueeze(0)
        # input_seq = torch.transpose(input_seq, 0, 1)
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        if self.linear is not None:
            predictions = self.linear(lstm_out[-1, :, :])
        else:
            predictions = lstm_out[-1, :, :]
        return predictions


class GRU(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=None, output_size=4, bidirectional=False):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        if hidden_layer_size is None:
            self.hidden_layer_size = output_size
        self.bidirectional = bidirectional
        if bidirectional:
            if hidden_layer_size is not None:
                self.gru = nn.GRU(input_size, self.hidden_layer_size // 2,
                                  num_layers=4,
                                  batch_first=True, bidirectional=True)
            else:
                self.gru = nn.GRU(input_size, self.hidden_layer_size,
                                  num_layers=4,
                                  batch_first=True, bidirectional=True)
        else:
            self.gru = nn.GRU(input_size, self.hidden_layer_size,
                              num_layers=4,
                              batch_first=True)
        if hidden_layer_size is not None:
            self.linear = nn.Linear(hidden_layer_size, output_size)
        else:
            self.linear = None

    def forward(self, input_seq):
        # input_seq = torch.transpose(input_seq, 0, 1)
        input_seq = input_seq.unsqueeze(0)
        gru_out, self.hidden_cell = self.gru(input_seq)
        if self.linear is not None:
            predictions = self.linear(gru_out[-1, :, :])
        else:
            predictions = gru_out[-1, :, :]
        return predictions


class LSTMFCN(nn.Module):
    def __init__(self, time_steps, num_variables=1, lstm_hs=64, channels=[1, 32, 64, 32], ae=False):
        super().__init__()
        self.embed = channels[-1] + lstm_hs
        self.lstm_block = BlockLSTM(time_steps * channels[0], 1, lstm_hs)
        self.fcn_block = BlockFCN(time_steps, channels=channels)
        self.ae = ae
        self.classification_layer = nn.Linear(channels[-1] + lstm_hs, num_variables)
        # self.softmax = nn.LogSoftmax(dim=1)  # nn.Softmax(dim=1)

    def embed_size(self):
        return self.embed

    def forward(self, x):
        x_lstm = torch.reshape(x, (x.size(0), -1))
        x_lstm = x_lstm.unsqueeze(1)
        x1 = self.lstm_block(x_lstm)
        x1 = torch.squeeze(x1)
        if len(x1.shape) == 1:
            x1 = x1.view(1, x1.size(0))
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x2 = self.fcn_block(x)
        x2 = torch.squeeze(x2)
        if len(x2.shape) == 1:
            x2 = x2.view(1, x2.size(0))
        x = torch.cat([x1, x2], 1)
        if not self.ae:
            x = self.classification_layer(x)
        # y = self.softmax(x)
        return x


class LSTMFRN(nn.Module):
    def __init__(self, time_steps, num_variables=1, lstm_hs=64, in_channels = 1, ae=False):
        super().__init__()
        self.embed = 64 + lstm_hs
        self.lstm_block = BlockLSTM(time_steps * in_channels, 1, lstm_hs)
        self.downsampling_layers = [
            nn.Conv1d(in_channels, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

        self.feature_layers = [ResBlock(64, 64) for _ in range(6)]
        self.res_extractor = nn.Sequential(*self.downsampling_layers, *self.feature_layers,
                                            norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1))
        self.ae = ae
        self.classification_layer = nn.Linear(64 + lstm_hs, num_variables)
        # self.softmax = nn.LogSoftmax(dim=1)  # nn.Softmax(dim=1)

    def embed_size(self):
        return self.embed

    def forward(self, x):
        x_lstm = torch.reshape(x, (x.size(0), -1))
        x_lstm = x_lstm.unsqueeze(1)
        x1 = self.lstm_block(x_lstm)
        x1 = torch.squeeze(x1)
        if len(x1.shape) == 1:
            x1 = x1.view(1, x1.size(0))
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x2 = self.res_extractor(x)
        x2 = torch.squeeze(x2)
        if len(x2.shape) == 1:
            x2 = x2.view(1, x2.size(0))
        x = torch.cat([x1, x2], 1)
        if not self.ae:
            x = self.classification_layer(x)
        # y = self.softmax(x)
        return x
