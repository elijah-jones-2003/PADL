import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 192x160 → 96x80
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 96x80 → 48x40
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 48x40 → 24x20
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 24x20 → 12x10
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 12 * 10, 32)

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 256 * 12 * 10)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 12x10 → 24x20
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 24x20 → 48x40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 48x40 → 96x80
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # 96x80 → 192x160
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 12, 10)
        return self.deconv(x)

def encode(images):
    device = torch.device("cuda" if images.is_cuda else "cpu")
    encoder = Encoder()
    encoder.load_state_dict(torch.load("models/autoencoder.pth")['encoder_state_dict'])
    encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        latents = encoder(images)
    return latents

def decode(latents):
    device = torch.device("cuda" if latents.is_cuda else "cpu")
    decoder = Decoder()
    decoder.load_state_dict(torch.load("models/autoencoder.pth")['decoder_state_dict'])
    decoder.to(device)
    decoder.eval()
    with torch.no_grad():
        reconstructed = decoder(latents)
    return reconstructed

