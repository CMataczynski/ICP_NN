import torch
import torch.nn as nn

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ResNet(nn.Module):
    def __init__(self, no_classes, in_channels=1, ae=False, depth=6):
        super().__init__()
        self.downsampling_layers = [
            nn.Conv1d(in_channels, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

        self.feature_layers = [ResBlock(64, 64) for _ in range(depth)]
        self.fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1), Flatten(), nn.Dropout(0.6)]
        self.classification_layer = nn.Linear(64, no_classes)
        self.ae = ae
        self.feature_extractor = nn.Sequential(*self.downsampling_layers, *self.feature_layers, *self.fc_layers)

    def forward(self, X):
        X = self.feature_extractor(X)
        if not self.ae:
            X = self.classification_layer(X)
        return X

