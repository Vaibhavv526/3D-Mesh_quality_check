import torch

from src.metrics.threshold_optimizer import ThresholdOptimizer


def main():

    predictions = torch.load(
        "validation_predictions.pt"
    )

    targets = torch.load(
        "validation_targets.pt"
    )

    thresholds = ThresholdOptimizer.find_best_thresholds(
        predictions,
        targets,
    )

    print("\nBest Thresholds")
    print("=" * 40)

    for index, threshold in enumerate(
        thresholds
    ):

        print(
            f"Class {index}: {threshold:.2f}"
        )

    optimized_f1 = (
        ThresholdOptimizer.evaluate_thresholds(
            predictions,
            targets,
            thresholds,
        )
    )

    print("\n" + "=" * 40)
    print(
        f"Original Macro F1 : 0.4560"
    )

    print(
        f"Optimized Macro F1: {optimized_f1:.4f}"
    )

    print(
        f"Improvement        : "
        f"{optimized_f1 - 0.4560:+.4f}"
    )


if __name__ == "__main__":
    main()