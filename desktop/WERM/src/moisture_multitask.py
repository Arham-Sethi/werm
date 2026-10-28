import torch
import torch.nn as nn
from torchvision.models import resnet18


class MoistureMultiTask(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = resnet18(weights=None)

        # We'll replace the default fc with identity and use our own heads
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc_cls = nn.Linear(in_features, num_classes)  # 3-class
        self.fc_reg = nn.Linear(in_features, 1)            # regression

    def forward(self, x):
        feats = self.backbone(x)         # [B, 512]
        logits = self.fc_cls(feats)      # [B, 3]
        y = self.fc_reg(feats)           # [B, 1]
        return logits, y
