import torch
import torch.nn as nn

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)

from configs import NUM_LABELS


class ConvNeXtModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT
        )

        in_features = self.backbone.classifier[2].in_features

        self.backbone.classifier[2] = nn.Identity()
        self.defect_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(
                in_features,
                NUM_LABELS,
            ),
        )

        self.quality_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(
                in_features,
                1,
            ),
        )

    def forward(self, x):

        features = self.backbone(x)

        defect_logits = self.defect_head(features)

        quality_logit = self.quality_head(features)

        return defect_logits, quality_logit