import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

class TextProjector(nn.Module):
    """Projects text embeddings to shared space."""
    
    def __init__(
        self,
        input_dim: int = 3072,
        output_dim: int = 1024,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_layers: int = 4,
    ):
        """
        Initialize text projector.
        
        Args:
            input_dim: Dimension of text embeddings (3072)
            output_dim: Dimension of shared embedding space
            hidden_dim: Hidden layer dimension (default: 1024)
            dropout: Dropout rate
            num_layers: Number of residual layers (default: 2)
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = 1024
        
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project text embeddings to shared space.
        
        Args:
            x: Text embeddings (batch_size, input_dim)
            
        Returns:
            Projected embeddings (batch_size, output_dim)
        """
        h = self.fc_in(x)
        for blk in self.blocks:       # residual MLP stack
            h = h + blk(h)
        h = self.fc_out(h)
        return F.normalize(h, dim=-1, eps=1e-3)


class CrossAttentionImageProjector(nn.Module):
    """Simplified projector with self-attention + cross-attention."""
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 1024,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_layers: int = 2,  # Kept for compatibility, not used
        num_heads: int = 4,  # Reduced from 8
    ):
        """
        Initialize simplified cross-attention image projector.
        
        Args:
            input_dim: Dimension of AION embeddings (768)
            output_dim: Dimension of shared embedding space (default: 1024)
            hidden_dim: Hidden dimension for attention (default: output_dim)
            dropout: Dropout rate
            num_layers: Kept for compatibility but not used
            num_heads: Number of attention heads (reduced to 4)
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = output_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Project input to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Token pooling to reduce sequence length
        # 576 tokens -> 64 tokens (9x reduction)
        self.token_pool = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, stride=9, padding=0)
        
        # Single self-attention layer
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP after self-attention
        self.mlp1_norm = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),  # Reduced from 4x
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Learned query vector
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Single cross-attention layer
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final MLP
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),  # Reduced from 4x
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize query vector
        nn.init.normal_(self.query, std=0.02)
        
        # Initialize other weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project image embeddings to shared space using self-attention + cross-attention.
        
        Args:
            x: Image embeddings (batch_size, n_tokens, input_dim)
            
        Returns:
            Projected embeddings (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        x = F.normalize(x, dim=-1, eps=1e-6)  # Normalize AION embeddings input (handles [B, N, D])
        
        # Project input
        x = self.input_proj(x)  # (B, N, hidden_dim)
        
        # Pool tokens to reduce sequence length
        x = x.transpose(1, 2)  # (B, hidden_dim, N)
        x = self.token_pool(x)  # (B, hidden_dim, N//9)
        x = x.transpose(1, 2)  # (B, N//9, hidden_dim)
        
        # Self-attention with residual on pooled tokens
        x_norm = self.self_attn_norm(x)
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + x_attn
        
        # MLP with residual
        x = x + self.mlp1(self.mlp1_norm(x))
        
        # Cross-attention with learned query
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, hidden_dim)
        q_norm = self.cross_attn_norm(query)
        attended, _ = self.cross_attn(q_norm, x, x, need_weights=False)
        query = query + attended
        
        # Final processing
        output = self.final_norm(query).squeeze(1)  # (B, hidden_dim)
        output = self.final_mlp(output)  # (B, output_dim)
        
        return F.normalize(output, dim=-1, eps=1e-3)


class SimpleImageProjector(nn.Module):
    """Simple projector for mean AION embeddings."""
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 1024,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_layers: int = 4,
    ):
        """
        Initialize simple image projector.
        
        Args:
            input_dim: Dimension of AION embeddings (768)
            output_dim: Dimension of shared embedding space
            hidden_dim: Hidden layer dimension (default: 1024)
            dropout: Dropout rate
            num_layers: Number of residual layers (default: 4)
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = 1024
        
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project image embeddings to shared space.
        
        Args:
            x: Image embeddings (batch_size, input_dim)
            
        Returns:
            Projected embeddings (batch_size, output_dim)
        """
        x = F.normalize(x, dim=-1, eps=1e-6) # Normalize AION embeddings input
        h = self.fc_in(x)
        for blk in self.blocks:       # residual MLP stack
            h = h + blk(h)
        h = self.fc_out(h)
        return F.normalize(h, dim=-1, eps=1e-3)