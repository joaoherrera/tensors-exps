import math
import os
from dataclasses import dataclass
from typing import Callable, Tuple, Any

import cv2
import numpy as np
import torch
from dataset.annotations import CocoAnnotations


class DatasetUtils:
    """Base operations for datasets."""

    @staticmethod
    def read_image(image_path: str, read_mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
        if not os.path.exists(image_path):
            raise ValueError("Path does not exist.")

        return cv2.imread(image_path, read_mode)

    @staticmethod
    def read_paths(directory_path: str) -> list[str]:
        if not os.path.isdir(directory_path):
            raise ValueError(f"Path {directory_path} is not a directory.")
        
        return os.listdir(directory_path)


@dataclass
class CocoDataset(torch.utils.data.Dataset):
    data_directory_path: str
    data_annotation_path: str
    augmentations: Callable = None
    preprocessing: Callable = None
    seed: Any = math.pi
    
    def __init__(self) -> None:
        super().__init__()
        self.tree = CocoAnnotations(self.data_annotation_path)
        self.images = CocoAnnotations.to_dict(self.tree.data["images"], "id")
        self.categories = CocoAnnotations.to_dict(self.tree.data["categories"], "id")
        self.annotations = self.tree.data.get("annotations")
    
    def __getitem__(self, idx) -> Tuple[np.ndarray, int]:
        # apply preprocessing
        # apply augmentations
        pass
    
    def __len__(self) -> int:
        return len(self.annotations)