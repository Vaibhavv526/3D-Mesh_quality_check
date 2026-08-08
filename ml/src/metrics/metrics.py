import torch

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)


class Metrics:

    @staticmethod
    def compute_metrics(
        predictions,
        targets,
        threshold=0.5,
    ):

        predictions = torch.sigmoid(
            predictions,
        )

        if isinstance(threshold, list):

            threshold = torch.tensor(
                threshold,
                device=predictions.device,
            )

        predictions = (
            predictions > threshold
        ).float()

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

    @staticmethod
    def compute_class_report(
        predictions,
        targets,
        threshold=0.5,
    ):
        predictions = torch.sigmoid(
            predictions,
        )

        if isinstance(threshold, list):

            threshold = torch.tensor(
                threshold,
                device=predictions.device,
            )

        predictions = (
            predictions > threshold
        ).float()

        predictions = predictions.cpu().numpy()

        targets = targets.cpu().numpy()

        report = classification_report(
            targets,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        return report