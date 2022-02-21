from torch import nn
from models.models_util import Flatten


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten()
        )

        self.classification_layers = nn.Sequential(
            nn.Linear(8*16*16, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classification_layers(x)
        return x

class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.pooling = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.de_pooling = nn.MaxUnpool2d(2, stride=2)
        self.decoder = nn.ConvTranspose2d(4, 1, kernel_size=5, stride=2, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        representation = self.encoder(x)
        representation, indices = self.pooling(representation)
        unpooled = self.de_pooling(representation, indices)
        y = self.relu(self.decoder(unpooled, output_size=x.size()))
        return y
