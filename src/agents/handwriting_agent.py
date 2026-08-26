import torch
import torch.nn as nn

class HandwritingAgent(nn.Module):
    """
    1D Convolutional Neural Network for Handwriting time-series data.
    Input shape: [batch_size, num_channels, sequence_length] -> [batch, 5, 2000]
    Output shape: [batch_size, embedding_dim] -> [batch, 64]
    """
    def __init__(self, in_channels=5, embedding_dim=64):
        super(HandwritingAgent, self).__init__()
        
        # 1D CNN Feature Extractor
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Global Average Pooling to flatten the sequence length
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Projection to desired embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x shape: [batch, 2000, 5] from dataloader, PyTorch Conv1d expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1) # -> [batch, 5, 2000]
        
        features = self.conv_block(x) # -> [batch, 64, 1]
        features = features.squeeze(-1) # -> [batch, 64]
        
        embedding = self.projection(features) # -> [batch, embedding_dim]
        return embedding
