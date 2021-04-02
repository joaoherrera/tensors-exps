import os
import glob
import torch
import numpy as np
import cv2


class Dataset(torch.utils.data.Dataset):
    """ A default class to define a dataset in PyTorch.
    """

    def __init__(self, dataset_path, in_channels=3, out_channels=1):
        self.dataset_path = dataset_path
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.sample_paths = []
        self.label_paths = []

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.sample_paths[idx], cv2.IMREAD_ANYDEPTH)
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_ANYDEPTH)

        return image, label

    def load(self, samples_dataset_path, labels_dataset_path, image_ext=['png', 'jpg']):
        """ Gather sample paths and their correspondent labels.

        Args:
            samples_dataset_path (str): Path to the samples directory.
            labels_dataset_path (str): Path to the labels directory.
            image_ext (list, optional): Valid extensions. Defaults to ['png', 'jpg'].
        """

        sample_paths = []
        label_paths = []

        for ext in image_ext:
            sample_paths.extend(glob.glob(f"{samples_dataset_path}{os.sep}*.{ext}"))
            label_paths.extend(glob.glob(f"{labels_dataset_path}{os.sep}*.{ext}"))

        # Check whether each sample has its correspondent label.
        sample_names = [os.path.basename(sample) for sample in sample_paths]
        label_names = [os.path.basename(sample) for label in label_paths]

        commom = list(set(sample_names).intersection(label_names))

        self.sample_paths = [os.path.join(samples_dataset_path, imname) for imname in commom]
        self.label_paths = [os.path.join(labels_dataset_path, imname) for imname in commom]

    def save(self, images, image_paths):
        raise NotImplementedError

    def init(self, batch_size, image_ext=['png', 'jpg']):
        samples_dataset = os.path.join(self.dataset_path, "samples")
        labels_dataset = os.path.join(self.dataset_path, "labels")

        self.load(samples_dataset, labels_dataset, image_ext)

        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, drop_last=False, num_workers=0)
