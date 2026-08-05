import torch

from src.losses.loss import MultiLabelLoss


def main():

    criterion = MultiLabelLoss()

    predictions = torch.randn(16, 10)

    targets = torch.randint(
        0,
        2,
        (16, 10),
    ).float()

    loss = criterion(
        predictions,
        targets,
    )

    print("=" * 60)
    print("LOSS FUNCTION TEST")
    print("=" * 60)

    print(f"Predictions Shape : {predictions.shape}")
    print(f"Targets Shape     : {targets.shape}")
    print(f"Loss Value        : {loss.item():.6f}")


if __name__ == "__main__":
    main()