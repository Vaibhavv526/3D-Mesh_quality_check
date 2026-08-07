"""
===========================================
3D Mesh Quality Control - Inference
===========================================

Author: K Vaibhav

Purpose:
    Load the trained model and perform
    inference on new samples.
"""

from src.evaluation.evaluator import Evaluator

from src.datasets.dataloader import get_dataloaders

import torch

from configs import DEVICE

from src.models.convnext import ConvNeXtModel

def load_model(
    checkpoint_path,
):

    model = ConvNeXtModel()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
    )

    model.to(DEVICE)

    model.eval()

    return model


from pathlib import Path


def main():

    checkpoint_path = Path(
        "checkpoints/best_model.pth"
    )

    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = load_model(
        checkpoint_path,
    )

    print("\n✓ Model Loaded Successfully!")

    print(model.__class__.__name__)

    print("\nLoading Validation Dataset...")

    _, val_loader = get_dataloaders()

    evaluator = Evaluator(
        model=model,
        device=DEVICE,
    )

    print("\n" + "=" * 60)
    print("FULL VALIDATION EVALUATION")
    print("=" * 60)

    metrics = evaluator.evaluate(
        val_loader,
    )

    print(f"\nF1 Score   : {metrics['f1']:.4f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")

    print(
        f"Validation Samples : {len(val_loader.dataset)}"
    )

    sample = next(
        iter(val_loader)
    )

    print("\nFirst Batch Loaded Successfully!")

    print(
        f"Image Shape : {sample['image'].shape}"
    )

    print(
        f"Labels Shape : {sample['labels'].shape}"
    )

    print(
        f"Quality Shape : {sample['quality'].shape}"
    )

    images = sample["image"].to(DEVICE)

    with torch.no_grad():

        defect_logits, quality_logits = model(images)

    print("\nForward Pass Successful!")

    print(
        f"Defect Logits Shape : {defect_logits.shape}"
    )

    print(
        f"Quality Logits Shape : {quality_logits.shape}"
    )

    defect_probs = torch.sigmoid(
        defect_logits,
    )

    quality_probs = torch.sigmoid(
        quality_logits,
    )

    print("\nFirst Sample Predictions")

    print(
        defect_probs[0]
    )

    print(
        quality_probs[0]
    )

    print("\nGround Truth Defects")

    print(
        sample["labels"][0]
    )

    print("\nGround Truth Quality")

    print(
        sample["quality"][0]
    )

    predicted_defects = (
        defect_probs > 0.5
    ).int()

    predicted_quality = (
        quality_probs > 0.5
    ).int()

    print("\nPredicted Defect Labels")

    print(
        predicted_defects[0]
    )

    print("\nPredicted Quality")

    print(
        predicted_quality[0]
    )


if __name__ == "__main__":
    main()