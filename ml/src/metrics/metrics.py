import torch

from sklearn.metrics import (
    f1_score,
    accuracy_score,
)


class Metrics:

    @staticmethod
    def compute_metrics(
        predictions,
        targets,
    ):

        predictions = torch.sigmoid(predictions)

        predictions = (predictions > 0.5).float()

        predictions = predictions.cpu().numpy()

        targets = targets.cpu().numpy()

        f1 = f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )

        accuracy = accuracy_score(
            targets,
            predictions,
        )

        return {
            "f1": f1,
            "accuracy": accuracy,
        }