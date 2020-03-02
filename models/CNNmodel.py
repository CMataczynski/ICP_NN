from torch import nn

from models.models_util import BlockConv, BlockConv2d


class CNN(nn.Module):
    def __init__(self, time_steps, out_features=4, channels=[1, 32, 64, 32], kernels=[9, 5, 3], mom=0.99, eps=0.001):
        super().__init__()
        size = time_steps
        self.conv1 = BlockConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps)
        size = size - (kernels[0] - 1)
        self.conv2 = BlockConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps)
        size = size - (kernels[1] - 1)
        self.conv3 = BlockConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
        size = size - (kernels[2] - 1)
        self.pooling = nn.MaxPool1d(size)
        self.dropout = nn.Dropout(0.5)
        self.fully_connected = nn.Linear(channels[-1], out_features)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        y = self.fully_connected(x)
        return y


class CNN2d(nn.Module):
    def __init__(self, image_shape, out_features=4, channels=[1, 32, 64, 64], kernels=[9, 7, 5], mom=0.99, eps=0.001):
        super().__init__()
        height = image_shape[0]
        width = image_shape[1]
        self.conv1 = BlockConv2d(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps)
        height = height - (kernels[0] - 1)
        width = width - (kernels[0] - 1)
        self.conv2 = BlockConv2d(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps)
        height = height - (kernels[1] - 1)
        width = width - (kernels[1] - 1)
        self.conv3 = BlockConv2d(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
        height = height - (kernels[2] - 1)
        width = width - (kernels[2] - 1)
        self.pooling = nn.MaxPool2d((height, width))
        self.dropout = nn.Dropout(0.5)
        self.fully_connected = nn.Linear(channels[-1], out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        y = self.fully_connected(x)
        return y
