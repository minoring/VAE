import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, img_shape, z_size):
        super(CNN, self).__init__()
        self.img_shape = img_shape
        self.z_size = z_size

        self.enc1 = nn.Conv2d(img_shape[0], 32, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc2 = nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(256)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(512)

        def compute_conv2d_size_out(size, kernel_size=3, stride=2, padding=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        self.enc_h = compute_conv2d_size_out(
            compute_conv2d_size_out(
                (compute_conv2d_size_out(compute_conv2d_size_out(img_shape[1])))))
        self.enc_w = compute_conv2d_size_out(
            compute_conv2d_size_out(
                (compute_conv2d_size_out(compute_conv2d_size_out(img_shape[2])))))
        linear_input_size = self.enc_h * self.enc_w * 512
        self.fc_mu = nn.Linear(linear_input_size, z_size)
        self.fc_logvar = nn.Linear(linear_input_size, z_size)

        self.dec_input = nn.Linear(z_size, self.enc_h * self.enc_w * 512)
        self.dec1 = nn.ConvTranspose2d(512,
                                       256,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.dec2 = nn.ConvTranspose2d(256,
                                       128,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec3 = nn.ConvTranspose2d(128,
                                       32,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1)
        self.dec_bn3 = nn.BatchNorm2d(32)
        self.dec4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn4 = nn.BatchNorm2d(3)
        self.dec5 = nn.Conv2d(3, self.img_shape[0], kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = self.enc_bn1(F.relu(self.enc1(x)))
        x = self.enc_bn2(F.relu(self.enc2(x)))
        x = self.enc_bn3(F.relu(self.enc3(x)))
        x = self.enc_bn4(F.relu(self.enc4(x)))
        x = torch.flatten(x, start_dim=1)

        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        x = F.relu(self.dec_input(z))
        x = x.view(-1, 512, self.enc_h, self.enc_w)
        x = self.dec_bn1(F.relu(self.dec1(x)))
        x = self.dec_bn2(F.relu(self.dec2(x)))
        x = self.dec_bn3(F.relu(self.dec3(x)))
        x = self.dec_bn4(F.relu(self.dec4(x)))
        x = self.dec5(x)

        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar
