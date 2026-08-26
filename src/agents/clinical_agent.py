import torch
import torch.nn as nn

class ClinicalAgent(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Clinical (Acoustic Tabular) data.
    Input shape: [batch_size, num_features] -> [batch, 22]
    Output shape: [batch_size, embedding_dim] -> [batch, 64]
    """
    def __init__(self, in_features=22, embedding_dim=64):
        super(ClinicalAgent, self).__init__()
        
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
        # x shape: [batch, 22]
        embedding = self.mlp(x) # -> [batch, embedding_dim]
        return embedding
