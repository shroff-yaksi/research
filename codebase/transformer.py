import torch
import torch.nn as nn
import math

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) for conditioning.
    Scales and shifts the normalized input based on the conditioning vector (time + class).
    """
    def __init__(self, d_model, d_cond):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Linear layer to predict scale (gamma) and shift (beta) from condition
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 2 * d_model)
        )
        
        # Initialize to zero so it starts as standard LayerNorm
        nn.init.zeros_(self.cond_proj[1].weight)
        nn.init.zeros_(self.cond_proj[1].bias)

    def forward(self, x, cond):
        # x: [batch, seq_len, d_model]
        # cond: [batch, d_cond]
        
        # Project condition to [batch, 2 * d_model]
        scale_shift = self.cond_proj(cond) # [batch, 2*d_model]
        
        # Split into gamma and beta
        gamma, beta = scale_shift.chunk(2, dim=-1) # [batch, d_model] each
        
        # Unsqueeze for broadcasting over sequence length
        gamma = gamma.unsqueeze(1) # [batch, 1, d_model]
        beta = beta.unsqueeze(1)   # [batch, 1, d_model]
        
        # Apply AdaLN: gamma * norm(x) + beta
        # We add 1 to gamma so that initial scale is 1 (identity)
        return (1 + gamma) * self.norm(x) + beta

class TabularTransformer(nn.Module):
    """
    Transformer-based backbone for Tabular Diffusion (DiT style).
    """
    def __init__(self, d_in, d_model=256, nhead=4, num_layers=4, d_cond=64, dropout=0.1):
        super().__init__()
        
        # 1. Embeddings (Time and Class)
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        
        # Class embedding for Guidance (0=Majority, 1=Minority, 2=Null/Unconditional)
        self.class_emb = nn.Embedding(3, d_cond)
        
        # 2. Input Projection
        self.input_proj = nn.Linear(d_in, d_model)
        
        # 3. Transformer Encoder Layers with AdaLN
        self.layers = nn.ModuleList([
            DiTBlock(d_model, nhead, d_cond, dropout) for _ in range(num_layers)
        ])
        
        # 4. Final Output Projection
        self.final_norm = AdaLN(d_model, d_cond)
        self.output_proj = nn.Linear(d_model, d_in)
        
    def forward(self, x, t, y=None):
        """
        x: [batch, d_in] (Latent vector)
        t: [batch, 1] (Time step)
        y: [batch] (Class labels)
        """
        # Compute embeddings
        t_emb = self.time_mlp(t) # [batch, d_cond]
        
        if y is not None:
            y_emb = self.class_emb(y) # [batch, d_cond]
        else:
            # If y is None, use the Null token (index 2)
            # Create a tensor of 2s
            device = x.device
            y_null = torch.full((x.shape[0],), 2, device=device, dtype=torch.long)
            y_emb = self.class_emb(y_null)
            
        # Combine conditioning
        cond = t_emb + y_emb # [batch, d_cond]
        
        # Project input to model dimension
        x = self.input_proj(x) # [batch, d_model]
        x = x.unsqueeze(1)     # [batch, 1, d_model] (Treat as seq_len=1)
        
        # Apply Transformer Blocks
        for layer in self.layers:
            x = layer(x, cond)
            
        # Final Norm and Projection
        x = self.final_norm(x, cond)
        x = self.output_proj(x) # [batch, 1, d_in]
        
        return x.squeeze(1) # [batch, d_in]

class DiTBlock(nn.Module):
    """
    A single Transformer block with AdaLN conditioning.
    Structure:
    x = x + Attention(AdaLN(x, c))
    x = x + MLP(AdaLN(x, c))
    """
    def __init__(self, d_model, nhead, d_cond, dropout=0.1):
        super().__init__()
        
        # Self-Attention
        self.norm1 = AdaLN(d_model, d_cond)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-Forward
        self.norm2 = AdaLN(d_model, d_cond)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, cond):
        # x: [batch, seq_len, d_model]
        # cond: [batch, d_cond]
        
        # 1. Self-Attention Block
        x_norm = self.norm1(x, cond)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # 2. MLP Block
        x_norm = self.norm2(x, cond)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x
