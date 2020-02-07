import torch
import torch.nn.functional as F
from torch import nn

from models.models_util import BlockConv, Flatten, UnFlatten, BlockDeconv


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(180, 32)
        self.fc21 = nn.Linear(32, 8)
        self.fc22 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 32)
        self.fc4 = nn.Linear(32, 180)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu, logvar = self.fc21(h1), self.fc22(h1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z, mu, logvar = self.encode(x.view(-1, 180))
        return self.decode(z), mu, logvar


class CNNVAE(nn.Module):
    def __init__(self, full_features=180, squeeze_size=8, channels=[1, 32, 64, 64], kernels=[4, 4, 4], stride=2):
        super(CNNVAE, self).__init__()

        self.encoder = nn.Sequential(
            BlockConv(channels[0], channels[1], kernels[0], norm=False, stride=stride),
            # size: 89
            BlockConv(channels[1], channels[2], kernels[1], norm=False, stride=stride),
            # size: 43
            BlockConv(channels[2], channels[3], kernels[2], norm=False, stride=stride),
            # size: 20
            Flatten()
        )
        size = int((full_features - (kernels[0] - 1) - 1) / stride) + 1
        size = int((size - (kernels[1] - 1) - 1) / stride) + 1
        size = int((size - (kernels[2] - 1) - 1) / stride) + 1
        # print(size)
        self.fc_squeeze_1 = nn.Linear(size*channels[3], squeeze_size)
        self.fc_squeeze_2 = nn.Linear(size*channels[3], squeeze_size)
        self.fc_expand_1 = nn.Linear(squeeze_size, size*channels[3])

        self.decoder = nn.Sequential(
            UnFlatten(),
            BlockDeconv(channels[3], channels[2], kernels[0] + 1, norm=False, stride=stride),
            BlockDeconv(channels[2], channels[1], kernels[1] + 1, norm=False, stride=stride),
            BlockDeconv(channels[1], channels[0], kernels[2], norm=False, stride=stride, sigmoid=True)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        mu, logvar = self.fc_squeeze_1(h), self.fc_squeeze_2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        x = x.unsqueeze(1)
        return self.bottleneck(self.encoder(x))

    def forward(self, x):
        x = x.unsqueeze(1)
        h = self.encoder(x)
        # print(h.size())
        z, mu, logvar = self.bottleneck(h)
        z = self.fc_expand_1(z)
        z = self.decoder(z)
        return z.view(z.size(0), -1), mu, logvar
