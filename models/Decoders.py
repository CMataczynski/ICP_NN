from torch import nn

class FCDecoder(nn.Module):
    def __init__(self, embed_size, output_size):
        super(FCDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, output_size),
            nn.Sigmoid()
            )

    def forward(self, embedding):
        return self.decoder(embedding)
