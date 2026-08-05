from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

import pandas as pd

from configs import (
    TRAIN_CSV,
    TRAIN_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)

from src.datasets.mesh_dataset import MeshDataset
from src.datasets.transforms import (
    get_train_transforms,
    get_val_transforms,
)
def get_dataloaders():
    train_df = pd.read_csv(TRAIN_CSV)

    train_dataframe, val_dataframe = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    train_dataset = MeshDataset(
        dataframe=train_dataframe,
        dataset_dir=TRAIN_DIR,
        image_transform=get_train_transforms(),
        mode="image",
    )

    val_dataset = MeshDataset(
        dataframe=val_dataframe,
        dataset_dir=TRAIN_DIR,
        image_transform=get_val_transforms(),
        mode="image",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


    return train_loader, val_loader