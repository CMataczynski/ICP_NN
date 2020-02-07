from torch import nn

from models.models_util import BlockConv


class CNN(nn.Module):
    def __init__(self, time_steps, out_features=4, channels=[1, 32, 64, 32], kernels=[8, 5, 3], mom=0.99, eps=0.001):
        super().__init__()
        size = time_steps
        self.conv1 = BlockConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps)
        size = size - (kernels[0] - 1)
        self.conv2 = BlockConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps)
        size = size - (kernels[1] - 1)
        self.conv3 = BlockConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
        size = size - (kernels[2] - 1)
        self.pooling = nn.MaxPool1d(size)
        self.fully_connected = nn.Linear(channels[-1], out_features)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        y = self.fully_connected(x)
        return y
