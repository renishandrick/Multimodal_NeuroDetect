import torch
import torch.nn as nn

class SensorAgent(nn.Module):
    """
    1D CNN for Gait Sensor Time-Series data.
    Input shape: [batch, channels, seq_len] -> [batch, 16, seq_len] (16 sensors)
    Output shape: [batch_size, embedding_dim] -> [batch, 64]
    """
    def __init__(self, in_channels=16, embedding_dim=64):
        super(SensorAgent, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        features = self.conv_block(x)
        features = features.squeeze(-1)
        embedding = self.projection(features)
        return embedding
