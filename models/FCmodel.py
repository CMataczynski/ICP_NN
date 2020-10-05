import torch.nn.functional as F
from torch import nn


class FCmodel(nn.Module):
    def __init__(self, in_features, out_features, hidden1=64, hidden2=32, ae=False):
        drop_val = 0.2
        super(FCmodel, self).__init__()
        self.hidden2 = hidden2
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.Dropout(drop_val),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.Dropout(drop_val),
            nn.ReLU()
        )
        self.ae = ae
        self.classification_layer = nn.Linear(hidden2, out_features)

    def forward(self, X,):
        X = self.feature_extractor(X)
        if not self.ae:
            X = self.classification_layer(X)
        return X

    def embed_size(self):
        return self.hidden2
