"""Module for the VAE model."""

import torch
from diffusers import AutoencoderKL


class VAE:

    vae_url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"

    def __init__(self, device):
        self.model = AutoencoderKL.from_single_file(self.vae_url).to(device)
        self.device = device

    def to_latent(self, input):
        with torch.no_grad():
            latent = self.model.encode(input.to(self.device))
        return latent.latent_dist.sample()

    def to_image(self, encoded):
        with torch.no_grad():
            output_img = self.model.decode(encoded)
        return output_img.sample
