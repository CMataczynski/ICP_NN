from torch import nn
import torch.nn.functional as F


class FCmodel(nn.Module):
    def __init__(self, in_features, out_features):
        super(FCmodel, self).__init__()
        self.input_layer = nn.Linear(in_features, 100)
        self.dropout0 = nn.Dropout(0)
        self.hidden_layer_1 = nn.Linear(100, 50)
        self.dropout1 = nn.Dropout(0)
        self.output_layer = nn.Linear(50, out_features)

    def forward(self, X, **kwargs):
        X = F.relu(self.dropout0(self.input_layer(X)))
        X = F.relu(self.dropout1(self.hidden_layer_1(X)))
        X = self.output_layer(X)
        return X