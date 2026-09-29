import torch
import torch.nn as nn

class ClassificationAgent(nn.Module):
    """
    Final Classification Network based on fused multimodal representations.
    """
    def __init__(self, in_features, num_classes=2):
        super(ClassificationAgent, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits
