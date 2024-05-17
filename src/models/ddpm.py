"""Module for DDPM model."""

import torch
import torch.nn as nn

from src.models.utils import pos_encoding


class EmbeddingBlock(nn.Module):
    """Embedding block for UNet."""

    def __init__(self, n_steps, d_model):
        super(EmbeddingBlock, self).__init__()
        self.n_steps = n_steps
        self.t_embed = self.init_pos_encoding(d_model)
        # self.l1 = nn.Linear(16, 32)
        # self.l2 = nn.Linear(32, d_model)
        # self.silu = nn.SiLU()

    def init_pos_encoding(self, d_model):
        t_embed = nn.Embedding(self.n_steps, d_model)
        t_embed.weight.data = pos_encoding(self.n_steps, d_model)
        t_embed.requires_grad = False
        return t_embed

    def forward(self, t):
        t = self.t_embed(t)
        # t = self.l1(t)
        # t = self.silu(t)
        # t = self.l2(t)
        return t


class ConvBlock(nn.Module):
    """Convolutional block for UNet."""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same"
        )
        self.relu = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bnorm(x)
        x = self.relu(x)
        return x


class DownsampleBlock(nn.Module):
    """Downsample block block for UNet."""

    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        pool = self.pool(x)
        return x, pool


class UpsampleBlock(nn.Module):
    """Upsample block for UNet."""

    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.conv = ConvBlock(2 * out_channels, out_channels)

    def forward(self, x, down_tensor):
        x = self.upconv(x)
        x = torch.cat((x, down_tensor), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """UNet model for diffusion."""

    def __init__(
        self, batch_size, n_steps, input_size=32, in_channels=4, first_layer_channels=64
    ):
        super(UNet, self).__init__()

        self.batch_size = batch_size

        # input size
        self.s1 = input_size
        self.s2 = self.s1 // 2
        self.s3 = self.s2 // 2

        # number of channels
        self.ch0 = in_channels
        self.ch1 = first_layer_channels
        self.ch2 = self.ch1 * 2
        self.ch3 = self.ch2 * 2

        # embedding blocks
        self.em1 = EmbeddingBlock(n_steps, in_channels * self.s1 * self.s1)
        self.em2 = EmbeddingBlock(n_steps, self.ch1 * self.s2 * self.s2)
        self.em3 = EmbeddingBlock(n_steps, self.ch2 * self.s3 * self.s3)
        self.em4 = EmbeddingBlock(n_steps, self.ch3 * self.s3 * self.s3)
        self.em5 = EmbeddingBlock(n_steps, self.ch2 * self.s2 * self.s2)

        # downsample blocks
        self.e1 = DownsampleBlock(self.ch0, self.ch1)
        self.e2 = DownsampleBlock(self.ch1, self.ch2)

        # upsample blocks
        self.d1 = UpsampleBlock(self.ch3, self.ch2)
        self.d2 = UpsampleBlock(self.ch2, self.ch1)

        # middle conv block
        self.middle = ConvBlock(self.ch2, self.ch3)

        # output layer
        self.out = nn.Conv2d(self.ch1, self.ch0, kernel_size=1, padding="same")

    def forward(self, x, t):
        t1 = self.em1(t).view(-1, self.ch0, self.s1, self.s1)
        t2 = self.em2(t).view(-1, self.ch1, self.s2, self.s2)
        t3 = self.em3(t).view(-1, self.ch2, self.s3, self.s3)
        t4 = self.em4(t).view(-1, self.ch3, self.s3, self.s3)
        t5 = self.em5(t).view(-1, self.ch2, self.s2, self.s2)

        x1, pool1 = self.e1(x + t1)
        x2, pool2 = self.e2(pool1 + t2)
        x = self.middle(pool2 + t3)
        x = self.d1(x + t4, x2)
        x = self.d2(x + t5, x1)
        x = self.out(x)
        return x


class DDPM(nn.Module):
    """DDPM model for diffusion."""

    def __init__(self, unet, device, min_beta=1e-4, max_beta=0.02):
        super(DDPM, self).__init__()
        self.unet = unet.to(device)
        self.device = device
        self.n_steps = unet.em1.n_steps
        self.betas = torch.linspace(min_beta, max_beta, self.n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor(
            [torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]
        ).to(device)

    def forward(self, x, t, eps=None):
        a = self.alpha_bars[t]
        if eps is None:
            eps = torch.randn(x.shape).to(self.device)
        x_with_noise = (
            a.sqrt().reshape(-1, 1, 1, 1) * x
            + (1 - a).sqrt().reshape(-1, 1, 1, 1) * eps
        )
        return x_with_noise

    def backward(self, x, t):
        return self.unet(x, t)
