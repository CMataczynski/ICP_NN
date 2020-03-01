import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1280):
        return input.view(input.size(0), 64, 20)


class BlockConv(nn.Module):
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99, epsilon=0.001, norm=True, stride=1):
        super().__init__()
        self.norm = norm
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        if norm:
            self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.batch_norm(x)
        y = self.relu(x)
        return y


class BlockDeconv(nn.Module):
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99, epsilon=0.001, norm=True, stride=1, sigmoid=False):
        super().__init__()
        self.norm = norm
        self.conv = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        if norm:
            self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        if not sigmoid:
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.batch_norm(x)
        y = self.act(x)
        return y


class BlockLSTM(nn.Module):
    def __init__(self, time_steps, num_layers, lstm_hs, dropout=0.8):
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


class BlockFCN(nn.Module):
    def __init__(self, time_steps, channels=[1, 32, 64, 32], kernels=[8, 5, 3], mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps)
        self.conv2 = BlockConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps)
        self.conv3 = BlockConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
        output_size = time_steps - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(kernel_size=output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # apply Global Average Pooling 1D
        y = self.global_pooling(x)
        return y



class BlockConv2d(nn.Module):
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99, epsilon=0.001, norm=True, stride=1):
        super().__init__()
        self.norm = norm
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        if norm:
            self.batch_norm = nn.BatchNorm2d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.batch_norm(x)
        y = self.relu(x)
        return y
