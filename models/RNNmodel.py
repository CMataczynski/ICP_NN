from torch import nn
import torch.nn.functional as F
import torch


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
        input_seq = input_seq.view(-1, len(input_seq), self.input_size)
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        if self.linear is not None:
            predictions = self.linear(torch.reshape(lstm_out[-1], (-1, self.hidden_layer_size)))
        else:
            predictions = lstm_out[-1]
        #print(predictions.shape)
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
        input_seq = input_seq.view(-1, len(input_seq), self.input_size)
        gru_out, self.hidden_cell = self.gru(input_seq)
        if self.linear is not None:
            predictions = self.linear(torch.reshape(gru_out[-1], (-1, self.hidden_layer_size)))
        else:
            predictions = gru_out[-1]
        # print(predictions.shape)
        return predictions


class BlockLSTM(nn.Module):
    def __init__(self, time_steps, num_layers, lstm_hs, dropout=0.8, attention=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=time_steps, hidden_size=lstm_hs, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        # input is of the form (batch_size, num_layers, time_steps), e.g. (128, 1, 512)
        x = torch.transpose(x, 0, 1)
        # lstm layer is of the form (num_layers, batch_size, time_steps)
        x, (h_n, c_n) = self.lstm(x)
        # dropout layer input shape (Sequence Length, Batch Size, Hidden Size * Num Directions)
        y = self.dropout(x)
        # output shape is same as Dropout intput
        return y


class BlockFCNConv(nn.Module):
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99, epsilon=0.001, squeeze=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()
    def forward(self, x):
        # input (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
        x = self.conv(x)
        # input (batch_size, out_channel, L_out)
        x = self.batch_norm(x)
        # same shape as input
        y = self.relu(x)
        return y


class BlockFCN(nn.Module):
    def __init__(self, time_steps, channels=[1, 128, 256, 128], kernels=[8, 5, 3], mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps, squeeze=True)
        self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
        output_size = time_steps - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(kernel_size=output_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # apply Global Average Pooling 1D
        y = self.global_pooling(x)
        return y


class LSTMFCN(nn.Module):
    def __init__(self, time_steps, num_variables=1, lstm_hs=256, channels=[1, 128, 256, 128]):
        super().__init__()
        self.lstm_block = BlockLSTM(time_steps, 1, lstm_hs)
        self.fcn_block = BlockFCN(time_steps)
        self.dense = nn.Linear(channels[-1] + lstm_hs, num_variables)
        self.softmax = nn.LogSoftmax(dim=1)  # nn.Softmax(dim=1)


    def forward(self, x):
        # input is (batch_size, time_steps), it has to be (batch_size, 1, time_steps)
        x = x.unsqueeze(1)
        # pass input through LSTM block
        x1 = self.lstm_block(x)
        x1 = torch.squeeze(x1)
        # pass input through FCN block
        x2 = self.fcn_block(x)
        x2 = torch.squeeze(x2)
        # concatenate blocks output
        # x1 = torch.zeros(x1.size())
        x = torch.cat([x1, x2], 1)
        # pass through Linear layer
        x = self.dense(x)
        # x = torch.squeeze(x)
        # pass through Softmax activation
        y = self.softmax(x)
        return y