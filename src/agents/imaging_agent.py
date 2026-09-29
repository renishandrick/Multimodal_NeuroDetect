import torch
import torch.nn as nn

class ImagingAgent(nn.Module):
    """
    2D Convolutional Neural Network for MRI Imaging data.
    Input shape: [batch_size, channels, height, width] -> [batch, 3, 224, 224]
    Output shape: [batch_size, embedding_dim] -> [batch, 64]
    """
    def __init__(self, in_channels=3, embedding_dim=64):
        super(ImagingAgent, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projection = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        embedding = self.projection(x)
        return embedding
