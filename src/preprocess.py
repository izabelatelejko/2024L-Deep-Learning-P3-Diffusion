"""Module for preprocess utils."""

import cv2
import numpy as np
import os
from tqdm import tqdm


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
