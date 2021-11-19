import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, input_size, z_size):
        super(FC, self).__init__()
        self.z_size = z_size
        self.flatten = nn.Flatten()
        self.enc1 = nn.Linear(input_size, 256)
        self.enc2 = nn.Linear(256, 512)
        self.enc3 = nn.Linear(512, 784)
        self.enc41 = nn.Linear(784, z_size)
        self.enc42 = nn.Linear(784, z_size)

        self.dec1 = nn.Linear(z_size, 784)
        self.dec2 = nn.Linear(784, 512)
        self.dec3 = nn.Linear(512, 256)
        self.dec4 = nn.Linear(256, input_size)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))

        return self.enc41(x), self.enc42(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = self.dec4(x)

        return x

    def forward(self, x):
        x = self.flatten(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar
