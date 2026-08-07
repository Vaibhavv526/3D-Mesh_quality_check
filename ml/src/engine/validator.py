from tqdm import tqdm
from src.metrics.metrics import Metrics

import torch

class Validator:

    def __init__(
        self,
        model,
        criterion,
        device,
    ):

        self.model = model.to(device)

        self.criterion = criterion

        self.device = device

    def validate(
        self,
        dataloader,
    ):

        self.model.eval()

        running_loss = 0.0

        all_predictions = []
        all_targets = []

        progress_bar = tqdm(
            dataloader,
            desc="Validation",
        )

        with torch.no_grad():

            for batch in progress_bar:

                images = batch["image"].to(self.device)

                labels = batch["labels"].to(self.device)

                defect_logits, quality_logits = self.model(images)

                quality = batch["quality"].to(self.device)

                loss = self.criterion(
                    defect_logits,
                    labels,
                    quality_logits,
                    quality,
                )

                all_predictions.append(
                    defect_logits.cpu()
                )

                all_targets.append(
                    labels.cpu()
                )

                running_loss += loss.item()

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
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
        return (
            running_loss / len(dataloader),
            metrics,
        )