"""
RawNet2-style architecture for audio anti-spoofing (ASVspoof).
Operates directly on raw audio waveforms using a SincConv first layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SincConv(nn.Module):
    """
    Learnable band-pass filter bank, based on the sinc function.
    Instead of learning arbitrary filter weights, this layer learns
    the low/high cutoff frequencies of each filter directly.
    """
    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # must be odd for symmetric filters

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialize filter cutoff frequencies (mel-scale spaced, like a filterbank)
        low_freq = 30
        high_freq = sample_rate / 2 - 100
        mel_low = 2595 * np.log10(1 + low_freq / 700)
        mel_high = 2595 * np.log10(1 + high_freq / 700)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        self.low_hz = nn.Parameter(torch.Tensor(hz_points[:-1]).view(-1, 1))
        self.band_hz = nn.Parameter(torch.Tensor(np.diff(hz_points)).view(-1, 1))

        n = (self.kernel_size - 1) / 2
        self.n_ = 2 * np.pi * torch.arange(-n, 0).view(1, -1) / sample_rate
        window = 0.54 - 0.46 * torch.cos(2 * np.pi * torch.arange(0, kernel_size) / kernel_size)
        self.window = window[:int((kernel_size - 1) / 2)]

    def forward(self, x):
        device = x.device
        low = torch.abs(self.low_hz.to(device)) + 1.0
        high = torch.clamp(low + torch.abs(self.band_hz.to(device)), 1.0, self.sample_rate / 2)
        band = (high - low)[:, 0]

        n_ = self.n_.to(device)
        window = self.window.to(device)

        f_times_t_low = torch.matmul(low, n_)
        f_times_t_high = torch.matmul(high, n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n_ / 2)) * window
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(x), 0.3)
        out = self.conv1(out)
        out = F.leaky_relu(self.bn2(out), 0.3)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.pool(out)


class RawNet2(nn.Module):
    def __init__(self, num_classes=2, sample_rate=16000):
        super().__init__()

        self.sinc_conv = SincConv(out_channels=128, kernel_size=251, sample_rate=sample_rate)
        self.bn_sinc = nn.BatchNorm1d(128)
        self.pool_sinc = nn.MaxPool1d(3)

        self.res_blocks = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )

        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x shape: (batch, num_samples) - raw waveform
        """
        x = x.unsqueeze(1)  # (batch, 1, num_samples) - add channel dim for conv1d

        x = torch.abs(self.sinc_conv(x))
        x = self.pool_sinc(F.leaky_relu(self.bn_sinc(x), 0.3))

        x = self.res_blocks(x)  # (batch, 256, reduced_length)

        x = x.permute(0, 2, 1)  # (batch, reduced_length, 256) - for GRU
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]

        logits = self.classifier(last_output)
        return logits


if __name__ == "__main__":
    model = RawNet2()
    dummy_input = torch.randn(2, 64000)  # batch of 2, 4-second audio at 16kHz
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expect (2, 2)