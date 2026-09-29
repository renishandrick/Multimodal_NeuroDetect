import torch
import torch.nn as nn

class GeneticAgent(nn.Module):
    """
    MLP for Gene Expression Data (GSE6613 series matrix).
    Input shape: [batch_size, num_genes]
    Output shape: [batch_size, embedding_dim]
    """
    def __init__(self, in_features=1000, embedding_dim=64): # using top 1000 highly variant genes
        super(GeneticAgent, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        embedding = self.mlp(x)
        return embedding
