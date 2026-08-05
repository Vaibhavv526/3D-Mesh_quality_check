from tqdm import tqdm

import torch


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
    ):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(
        self,
        dataloader,
    ):

        self.model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc="Training",
        )

        for batch in progress_bar:

            images = batch["image"].to(self.device)

            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            defect_logits, quality_logits = self.model(images)

            quality = batch["quality"].to(self.device)

            loss = self.criterion(
                defect_logits,
                labels,
                quality_logits,
                quality,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return running_loss / len(dataloader)