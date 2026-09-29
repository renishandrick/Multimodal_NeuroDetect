import torch
import torch.nn as nn

class SpeechAgent(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Speech (Acoustic Tabular) data from Telemonitoring dataset.
    Input shape: [batch_size, num_features] -> [batch, 21] 
    Output shape: [batch_size, embedding_dim] -> [batch, 64]
    """
    def __init__(self, in_features=21, embedding_dim=64):
        super(SpeechAgent, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        embedding = self.mlp(x)
        return embedding
