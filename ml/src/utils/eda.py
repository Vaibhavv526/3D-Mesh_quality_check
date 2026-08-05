"""
===========================================
3D Mesh Quality Control - EDA Module
===========================================

Author: K Vaibhav
Project: 3D Mesh Quality Control
Purpose:
    Perform Exploratory Data Analysis (EDA)
    on the competition dataset.

This module currently analyzes:
    - train.csv
    - test.csv
    - Label statistics
    - Missing values
    - Duplicate item_ids

Future extensions:
    - Image analysis
    - Mesh analysis
    - Dataset integrity checks
    - Visualization reports
"""
from configs import (
    TRAIN_CSV,
    TEST_CSV,
    TRAIN_DIR,
    TEST_DIR,

)

from pathlib import Path
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file.

    Args:
        file_path: Path to CSV.

    Returns:
        pandas DataFrame
    """

    logger.info(f"Loading {file_path.name}")

    return pd.read_csv(file_path)

def dataset_overview(train_df: pd.DataFrame,
                     test_df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Training Samples : {len(train_df)}")
    print(f"Testing Samples  : {len(test_df)}")

    print(f"\nTraining Shape : {train_df.shape}")
    print(f"Testing Shape  : {test_df.shape}")

def check_missing_values(train_df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = train_df.isnull().sum()

    print(missing)

def check_duplicates(train_df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("DUPLICATE ITEM IDs")
    print("=" * 60)

    duplicates = train_df["item_id"].duplicated().sum()

    print(f"Duplicate item_ids : {duplicates}")

def label_distribution(train_df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)

    labels = train_df.columns[1:]

    for label in labels:

        positives = train_df[label].sum()
        negatives = len(train_df) - positives

        print(
            f"{label:<15} Positive: {positives:<5} Negative: {negatives}"
        )

def verify_training_files(train_df):

    print("\n" + "=" * 60)
    print("TRAIN DATASET VALIDATION")
    print("=" * 60)

    missing_png = 0
    missing_npz = 0

    for item_id in train_df["item_id"]:

        png_path = TRAIN_DIR / f"{item_id}.png"
        npz_path = TRAIN_DIR / f"{item_id}.npz"

        if not png_path.exists():
            missing_png += 1

        if not npz_path.exists():
            missing_npz += 1

    print(f"Missing PNG Files : {missing_png}")
    print(f"Missing NPZ Files : {missing_npz}")


def verify_test_files(test_df):

    print("\n" + "=" * 60)
    print("TEST DATASET VALIDATION")
    print("=" * 60)

    missing_png = 0
    missing_npz = 0

    for item_id in test_df["item_id"]:

        png_path = TEST_DIR / f"{item_id}.png"
        npz_path = TEST_DIR / f"{item_id}.npz"

        if not png_path.exists():
            missing_png += 1

        if not npz_path.exists():
            missing_npz += 1

    print(f"Missing PNG Files : {missing_png}")
    print(f"Missing NPZ Files : {missing_npz}")

def main():
    train_df = load_csv(TRAIN_CSV)
    test_df = load_csv(TEST_CSV)

    dataset_overview(train_df, test_df)

    check_missing_values(train_df)

    check_duplicates(train_df)

    label_distribution(train_df)
    verify_training_files(train_df)
    
    verify_test_files(test_df)

if __name__ == "__main__":
    main()

