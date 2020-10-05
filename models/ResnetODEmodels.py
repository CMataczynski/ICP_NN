import os
import argparse
import logging
import time
import numpy as np
import torch
import datetime as dt
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import plot_confusion_matrix
import torchvision.datasets as datasets
from sklearn.metrics import f1_score
import torchvision.transforms as transforms
from utils import Initial_dataset_loader, ShortenOrElongateTransform
from models.models_util import ResBlock, conv1x1, ODEBlock, ODEfunc, norm, Flatten
from torchdiffeq import odeint_adjoint as odeint


class ODE(nn.Module):
    def __init__(self, no_classes, ae=False):
        super().__init__()
        self.downsampling_layers = [
            nn.Conv1d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

        self.feature_layers = [ODEBlock(ODEfunc(64))]
        self.fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1), Flatten(), nn.Dropout(0.6)]
        self.classification_layer = nn.Linear(64, no_classes)
        self.ae = ae
        self.feature_extractor = nn.Sequential(*self.downsampling_layers, *self.feature_layers, *self.fc_layers)

    def forward(self, X):
        X = self.feature_extractor(X)
        if not self.ae:
            X = self.classification_layer(X)
        return X

    def embed_size(self):
        return 64

class ResNet(nn.Module):
    def __init__(self, no_classes):
        super().__init__()
        self.downsampling_layers = [
            nn.Conv1d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

        self.feature_layers = [ResBlock(64, 64) for _ in range(6)]
        self.fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1), Flatten(), nn.Dropout(0.6)]
        self.classification_layer = nn.Linear(64, no_classes)
        self.ae = ae
        self.feature_extractor = nn.Sequential(*self.downsampling_layers, *self.feature_layers, *self.fc_layers)

    def forward(self, X):
        X = self.feature_extractor(X)
        if not self.ae:
            X = self.classification_layer(X)
        return X
        
    def embed_size(self):
        return 64
