from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset

from configs import (
    TRAIN_DIR,
    TEST_DIR,
)

class MeshDataset(Dataset):
    """
    Custom PyTorch Dataset for
    3D Mesh Quality Control.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_dir: Path,
        image_transform=None,
        mode: str = "multimodal",
    ):
        """
        Args:
            dataframe: train.csv or test.csv
            dataset_dir: dataset/train or dataset/test
            image_transform: torchvision transforms
            mode:
                "image"
                "mesh"
                "multimodal"
        """

        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)

        self.image_transform = image_transform

        self.mode = mode

    def __len__(self):
        """
        Returns
        -------
        Total number of samples.
        """

        return len(self.dataframe)

    # Image Path

    def get_image_path(self, item_id: str) -> Path:

        return self.dataset_dir / f"{item_id}.png"

    # Mesh Path
    def get_mesh_path(self, item_id: str) -> Path:

        return self.dataset_dir / f"{item_id}.npz"
    
    def _load_image(self, image_path: Path):
        """
        Load a PNG image.

        Args:
            image_path: Path to image.

        Returns:
            PIL Image
        """

        image = Image.open(image_path).convert("RGB")

        if self.image_transform is not None:
            image = self.image_transform(image)

        return image
    def _load_mesh(self, mesh_path: Path):
        """
        Load mesh (.npz).

        Returns:
            vertices
            faces
        """

        mesh = np.load(mesh_path)

        vertices = mesh["vertices"]

        faces = mesh["faces"]

        return vertices, faces
    def _load_labels(self, row: pd.Series):

        if "quality" not in row.index:

            return None, None

        labels = row.iloc[1:-1].to_numpy(dtype=np.float32)

        quality = np.float32(row["quality"])

        return labels, quality
    def __getitem__(self, index: int):

        row = self.dataframe.iloc[index]

        item_id = str(row["item_id"])

        image_path = self.get_image_path(item_id)

        mesh_path = self.get_mesh_path(item_id)

        image = None
        vertices = None
        faces = None

        if self.mode in ["image", "multimodal"]:
            image = self._load_image(image_path)

        if self.mode in ["mesh", "multimodal"]:
            vertices, faces = self._load_mesh(mesh_path)

        labels, quality = self._load_labels(row)

        labels = torch.tensor(labels, dtype=torch.float32)
        quality = torch.tensor(quality, dtype=torch.float32)

        sample = {
            "item_id": item_id,
            "labels": labels,
            "quality": quality,
        }

        if self.mode in ["image", "multimodal"]:
            sample["image"] = image

        if self.mode in ["mesh", "multimodal"]:
            sample["vertices"] = vertices
            sample["faces"] = faces

        return sample