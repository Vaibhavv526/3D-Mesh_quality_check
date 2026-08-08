import torch

from sklearn.metrics import f1_score


class ThresholdOptimizer:

    @staticmethod
    def find_best_thresholds(
        predictions,
        targets,
    ):

        predictions = torch.sigmoid(
            predictions,
        ).cpu().numpy()

        targets = targets.cpu().numpy()

        best_thresholds = []

        for class_index in range(
            predictions.shape[1]
        ):

            best_f1 = 0.0
            best_threshold = 0.5

            for threshold in torch.arange(
                0.10,
                0.91,
                0.05,
            ):

                threshold = float(
                    threshold
                )

                class_predictions = (
                    predictions[:, class_index]
                    > threshold
                ).astype(int)

                class_targets = (
                    targets[:, class_index]
                ).astype(int)

                f1 = f1_score(
                    class_targets,
                    class_predictions,
                    zero_division=0,
                )

                if f1 > best_f1:

                    best_f1 = f1
                    best_threshold = threshold

            best_thresholds.append(
                best_threshold
            )

        return best_thresholds

    @staticmethod
    def evaluate_thresholds(
        predictions,
        targets,
        thresholds,
    ):

        predictions = torch.sigmoid(
            predictions,
        ).cpu().numpy()

        targets = targets.cpu().numpy()

        predicted_labels = (
            predictions
            > thresholds
        ).astype(int)

        macro_f1 = f1_score(
            targets,
            predicted_labels,
            average="macro",
            zero_division=0,
        )

        return macro_f1