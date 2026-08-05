from configs import TRAIN_CSV, TRAIN_DIR

import pandas as pd

from src.datasets.mesh_dataset import MeshDataset


def main():

    train_df = pd.read_csv(TRAIN_CSV)

    dataset = MeshDataset(
        dataframe=train_df,
        dataset_dir=TRAIN_DIR,
        mode="multimodal",
    )

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Dataset Size : {len(dataset)}")

    sample = dataset[0]

    print("\nSample Keys")

    print(sample.keys())

    print("\nItem ID")

    print(sample["item_id"])

    print("\nImage")

    print(sample["image"])

    print("\nVertices Shape")

    print(sample["vertices"].shape)

    print("\nFaces Shape")

    print(sample["faces"].shape)

    print("\nLabels")

    print(sample["labels"])

    print("\nQuality")

    print(sample["quality"])


if __name__ == "__main__":
    main()