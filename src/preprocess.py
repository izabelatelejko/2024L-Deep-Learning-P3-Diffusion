"""Module for preprocess utils."""

import cv2
import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

GLOBAL_MEAN = np.load("data/global_mean.npy")
GLOBAL_STD = np.load("data/global_std.npy")


def calculate_global_mean_and_std(data_path="data/data0/lsun/bedroom"):
    """Calculate and save global mean and std for each color channel across all images."""
    subdirs = [str(i) for i in range(10)] + ["a", "b", "c", "d", "e", "f"]
    means = np.array([0, 0, 0])
    stds = np.array([0, 0, 0])
    img_count = 0
    for subdir1 in tqdm(subdirs, "Processing..."):
        for subdir2 in subdirs:
            for subdir3 in subdirs:
                img_subdir = os.path.join(data_path, subdir1, subdir2, subdir3)
                for filename in os.listdir(img_subdir):
                    img = cv2.imread(os.path.join(img_subdir, filename))

                    means = means + np.mean(img, axis=(0, 1))
                    stds = stds + np.std(img, axis=(0, 1))
                    img_count += 1

    np.save("data/global_std.npy", stds / img_count)
    np.save("data/global_mean.npy", means / img_count)


def get_data_loader(data_path, batch_size, vae=None, n=1000):
    """Returns a dataloader for the given data_path."""
    if vae is not None:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=GLOBAL_MEAN, std=GLOBAL_STD),
                transforms.Lambda(lambda x: vae.to_latent(x.unsqueeze(0)).squeeze(0)),
            ]
        )
    else:
        transform = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )

    image_dataset = ImageFolder(root=data_path, transform=transform)
    image_dataset = Subset(image_dataset, torch.randperm(len(image_dataset))[:n])
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    return dataloader
