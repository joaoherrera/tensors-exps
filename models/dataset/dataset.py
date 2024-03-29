import math
import os
from dataclasses import dataclass
from typing import Callable, Tuple, Any

import cv2
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
from dataset.annotations import CocoAnnotations


class DatasetUtils:
    """Base operations for datasets."""

    @staticmethod
    def read_image(image_path: str, read_mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
        if not os.path.exists(image_path):
            raise ValueError("Path does not exist.")

        return cv2.imread(image_path, read_mode).transpose(2, 0, 1)

    @staticmethod
    def read_paths(directory_path: str) -> list[str]:
        if not os.path.isdir(directory_path):
            raise ValueError(f"Path {directory_path} is not a directory.")
        
        return os.listdir(directory_path)


@dataclass
class CocoDataset(Dataset):
    data_directory_path: str
    data_annotation_path: str
    augmentations: Callable = None
    preprocessing: Callable = None
    seed: Any = 2022
    
    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        super().__init__()

        self.tree = CocoAnnotations(self.data_annotation_path)
        self.images = CocoAnnotations.to_dict(self.tree.data["images"], "id")
        self.categories = CocoAnnotations.to_dict(self.tree.data["categories"], "id")
        self.annotations = self.tree.data.get("annotations")
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        annotation = self.annotations[idx]
        image_data = self.images[annotation["image_id"]]
        image_path = os.path.join(self.data_directory_path, image_data["file_name"])
        image = DatasetUtils.read_image(image_path)
        
        # apply preprocessing
        if self.preprocessing is not None:
            image, annotation = self.preprocessing(image, annotation)
        
        # apply augmentations
        if self.augmentations is not None:
            image, annotation = self.augmentations(image, annotation)
        
        image = torch.tensor(image, dtype=torch.float32)
        category = torch.tensor(self.categories[annotation["category_id"]]["id"], dtype=torch.float32)

        return image, category
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    @classmethod
    def dataloader(cls, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(cls, batch_size=batch_size, shuffle=shuffle)