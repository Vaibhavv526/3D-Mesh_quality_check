from tqdm import tqdm

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

                running_loss += loss.item()

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        return running_loss / len(dataloader)