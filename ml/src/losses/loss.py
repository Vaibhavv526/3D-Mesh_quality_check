import torch.nn as nn


class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        pos_weight=None,
    ):

        super().__init__()

        self.defect_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
        )

        self.quality_loss = nn.BCEWithLogitsLoss()
    def forward(

        self,

        defect_logits,
        defect_targets,

        quality_logits,
        quality_targets,
    ):

        defect_loss = self.defect_loss(
            defect_logits,
            defect_targets,
        )

        quality_loss = self.quality_loss(
            quality_logits,
            quality_targets.unsqueeze(1),
        )

        total_loss = defect_loss + 0.5 * quality_loss

        return total_loss