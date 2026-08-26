import torch
import torch.nn as nn

class FusionAgent(nn.Module):
    """
    Cross-Attention Fusion Network to merge multimodal embeddings.
    """
    def __init__(self, embedding_dim=64, num_heads=4, num_classes=2):
        super(FusionAgent, self).__init__()
        
        # PyTorch MultiheadAttention expects sequence dimension first (seq_len, batch, embed_dim)
        # Since we are passing single embeddings, seq_len = 1
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        
        # Final classification head after concatenating the two embeddings (64 + 64 = 128)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, hw_embed, clin_embed):
        # hw_embed: [batch, 64] -> [batch, 1, 64] for attention
        # clin_embed: [batch, 64] -> [batch, 1, 64]
        
        hw_seq = hw_embed.unsqueeze(1)
        clin_seq = clin_embed.unsqueeze(1)
        
        # Let Handwriting (Query) attend to Clinical (Key, Value)
        attn_output, _ = self.cross_attention(query=hw_seq, key=clin_seq, value=clin_seq)
        
        # Squeeze back to [batch, 64]
        attn_output = attn_output.squeeze(1)
        
        # Concatenate original clinical with the attention-enriched handwriting representation
        fused = torch.cat((attn_output, clin_embed), dim=1) # -> [batch, 128]
        
        # Classify
        logits = self.classifier(fused) # -> [batch, num_classes]
        return logits
