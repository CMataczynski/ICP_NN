from torch import nn

from models.models_util import BlockConv, BlockConv2d, Flatten, BlockDeconv


class CNN(nn.Module):
    def __init__(self, time_steps, out_features=4, channels=[1, 32, 64, 64],
                    kernels=[9, 5, 3], mom=0.99, eps=0.001, ae=False):
        super().__init__()
        dropout_val = 0.3

        self.feature_extractor = nn.Sequential(
            BlockConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps, norm = not ae),
            BlockConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps, norm = not ae),
            BlockConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps, norm = not ae),
            nn.AdaptiveMaxPool1d(1),
            Flatten(),
            nn.Dropout(dropout_val),
            nn.Linear(channels[-1], 32),
            nn.Dropout(dropout_val),
            nn.ReLU()
        )
        self.classification_layer = nn.Linear(32, out_features)
        self.ae = ae

    def forward(self, x):
        x = feature_extractor(x)
        if not self.ae:
            x = self.classification_layer(x)
        return x

    def embed_size(self):
        return 32


class CNN2d(nn.Module):
    def __init__(self, image_shape, out_features=4, channels=[1, 16, 32, 64], kernels=[9, 7, 5], mom=0.99, eps=0.001):
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
        self.pooling = nn.AdaptiveMaxPool2d(1)
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
