import torch
import torch.nn as nn

class FusionAgent(nn.Module):
    """
    Self-Attention Fusion Network to merge N multimodal embeddings.
    """
    def __init__(self, embedding_dim=64, num_heads=4):
        super(FusionAgent, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, embeddings_list):
        """
        embeddings_list: List of tensors, each of shape [batch, embedding_dim]
        Returns:
            fused_representation: [batch, embedding_dim * num_modalities]
        """
        # Stack embeddings along the sequence dimension
        # Resulting shape: [batch, num_modalities, embedding_dim]
        stacked_seq = torch.stack(embeddings_list, dim=1)
        
        # Self-attention across modalities
        attn_output, _ = self.self_attention(query=stacked_seq, key=stacked_seq, value=stacked_seq)
        
        # Flatten the sequence to create a single long fused feature vector
        batch_size = attn_output.size(0)
        fused = attn_output.reshape(batch_size, -1)
        
        return fused
