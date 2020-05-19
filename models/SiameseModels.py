from models.models_util import BlockConv, BlockConv2d, conv1x1, conv3x3, norm, ResBlock, ConcatConv2d

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint



class SiameseResNet(nn.Module):
    def __init__(self, no_classes, feature_extraction_layers=6):
        super().__init__()
        self.downsampling_layers = [
            nn.Conv1d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]
        self.feature_layers = [ResBlock(64, 64) for _ in range(feature_extraction_layers)]
        self.fc_layers = [norm(64), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool1d(1), Flatten(),
                    nn.Dropout(0.6), nn.Linear(64,32), nn.ReLU(inplace=True)]
        self.classification_layer = nn.Linear(64, no_classes)
        self.feature_extractor = nn.Sequential(*self.downsampling_layers, *self.feature_layers, *self.fc_layers)

    def forward(self, icp, abp):
        icp = icp.unsqueeze(1)
        abp = abp.unsqueeze(1)
        icp_features = self.feature_extractor(icp)
        abp_features = self.feature_extractor(abp)
        concatenated = torch.cat((icp_features, abp_features), 1)
        return self.classification_layer(concatenated)


class SiameseNeuralODE(nn.Module):
    def __init__(self, no_classes):
        super().__init__()
        self.downsampling_layers = [
            nn.Conv1d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]
        self.feature_layers = [ODEBlock(ODEfunc(64))]
        self.fc_layers = [norm(64), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool1d(1), Flatten(),
                    nn.Dropout(0.6), nn.Linear(64,32), nn.ReLU(inplace=True)]
        self.classification_layer = nn.Linear(64, no_classes)
        self.feature_extractor = nn.Sequential(*self.downsampling_layers, *self.feature_layers, *self.fc_layers)

    def forward(self, icp, abp):
        icp = icp.unsqueeze(1)
        abp = abp.unsqueeze(1)
        icp_features = self.feature_extractor(icp)
        abp_features = self.feature_extractor(abp)
        concatenated = torch.cat((icp_features, abp_features), 1)
        return self.classification_layer(concatenated)


class SiameseShallowCNN(nn.Module):
    def __init__(self, out_features=4, channels=[1, 32, 64, 32], kernels=[9, 5, 3], mom=0.99, eps=0.001, dropout=0.6):
        super().__init__()
        self.feature_layers = [
            BlockConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps),
            BlockConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps),
            BlockConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps),
            nn.AdaptiveMaxPool1d(1),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 32),
            nn.ReLU(inplace=True)
        ]
        self.feature_extractor = nn.Sequential(*self.feature_layers)
        self.classification_layer = nn.Linear(64, out_features)

    def forward(self, icp, abp):
        icp = icp.unsqueeze(1)
        abp = abp.unsqueeze(1)
        icp_features = self.feature_extractor(icp)
        abp_features = self.feature_extractor(abp)
        concatenated = torch.cat((icp_features, abp_features), 1)
        return self.classification_layer(concatenated)
