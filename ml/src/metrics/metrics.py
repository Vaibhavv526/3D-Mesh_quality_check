import torch

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
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

        precision = precision_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )

        return {
            "f1": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }