"""
===========================================
3D Mesh Quality Control - Training Script
===========================================

Author: K Vaibhav

Purpose:
    Main training pipeline for the
    3D Mesh Quality Control model.
"""

from src.utils.class_weights import compute_pos_weight
from src.callbacks.early_stopping import EarlyStopping
import random
import numpy as np
import torch

from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
)

from src.datasets.dataloader import get_dataloaders

from src.models.convnext import ConvNeXtModel

from src.losses.loss import MultiTaskLoss

from src.engine.trainer import Trainer
from src.engine.validator import Validator

def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

def print_device():

    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)

    print(f"Device : {DEVICE}")

    if torch.cuda.is_available():

        print(f"GPU : {torch.cuda.get_device_name(0)}")

def main():

    set_seed(RANDOM_SEED)

    print_device()
    print("\nLoading Dataloaders...")

    train_loader, val_loader = get_dataloaders()
    train_dataframe = train_loader.dataset.dataframe

    pos_weight = compute_pos_weight(
        train_dataframe,
    ).to(DEVICE)

    print("\nPositive Class Weights")
    print(pos_weight)
    print(f"Training Batches   : {len(train_loader)}")
    print(f"Validation Batches : {len(val_loader)}")
    print("\nBuilding Model...")

    model = ConvNeXtModel()

    print(model.__class__.__name__)
    criterion = MultiTaskLoss(
        pos_weight=pos_weight,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
    )

    validator = Validator(
        model=model,
        criterion=criterion,
        device=DEVICE,
    )

    early_stopping = EarlyStopping(
        patience=5,
    )

    print("\n✓ Model Initialized Successfully")

    print(f"Epochs        : {EPOCHS}")
    print(f"Learning Rate : {LEARNING_RATE}")
    print(f"Device        : {DEVICE}")
    checkpoint_dir = Path("checkpoints")

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):

        print("\n" + "=" * 60)

        print(f"Epoch {epoch + 1}/{EPOCHS}")

        print("=" * 60)

        train_loss = trainer.train_one_epoch(
            train_loader,
        )
        val_loss, metrics = validator.validate(
            val_loader,
        )

        early_stopping.step(
            metrics["f1"],
        )
        
        
        print(f"\nTrain Loss      : {train_loss:.4f}")

        print(f"Validation Loss : {val_loss:.4f}")

        print(f"F1 Score        : {metrics['f1']:.4f}")

        print(f"Accuracy        : {metrics['accuracy']:.4f}")

        print(f"Precision       : {metrics['precision']:.4f}")

        print(f"Recall          : {metrics['recall']:.4f}")

        print(
            f"Learning Rate   : "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

        if metrics["f1"] > best_val_f1:

            best_val_f1 = metrics["f1"]

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                },
                checkpoint_dir / "best_model.pth",
            )

            print("\n✓ Best model saved.")

        if early_stopping.should_stop:

            print(
                f"\nEarly stopping triggered after "
                f"{epoch + 1} epochs."
            )

            break
        scheduler.step()
if __name__ == "__main__":
    main()