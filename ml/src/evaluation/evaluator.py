import torch

from tqdm import tqdm

from src.metrics.metrics import Metrics


class Evaluator:

    def __init__(
        self,
        model,
        device,
    ):

        self.model = model.to(device)

        self.device = device

    def evaluate(
        self,
        dataloader,
    ):

        self.model.eval()

        all_predictions = []

        all_targets = []

        progress_bar = tqdm(
            dataloader,
            desc="Evaluating",
        )

        with torch.no_grad():

            for batch in progress_bar:

                images = batch["image"].to(
                    self.device,
                )

                labels = batch["labels"].to(
                    self.device,
                )

                defect_logits, _ = self.model(
                    images,
                )

                all_predictions.append(
                    defect_logits.cpu(),
                )

                all_targets.append(
                    labels.cpu(),
                )
            all_predictions = torch.cat(
                all_predictions,
                dim=0,
            )

            all_targets = torch.cat(
                all_targets,
                dim=0,
            )

            metrics = Metrics.compute_metrics(
                all_predictions,
                all_targets,
            )

            return metrics