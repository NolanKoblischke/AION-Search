import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .projections import TextProjector, CrossAttentionImageProjector, SimpleImageProjector

class GalaxyClipModel(nn.Module):
    """CLIP model for aligning galaxy images and text descriptions."""
    
    def __init__(
        self,
        image_input_dim: int = 768,
        text_input_dim: int = 3072,
        embedding_dim: int = 1024,
        image_hidden_dim: int = 768,
        text_hidden_dim: int = 1024,
        dropout: float = 0.1,
        use_mean_embeddings: bool = True
    ):
        """
        Initialize CLIP model.
        
        Args:
            image_input_dim: AION embedding dimension
            text_input_dim: Text embedding dimension  
            embedding_dim: Shared embedding space dimension
            image_hidden_dim: Hidden dimension for image projector
            text_hidden_dim: Hidden dimension for text projector
            dropout: Dropout rate
            use_mean_embeddings: Whether using mean embeddings (True) or full embeddings (False)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_mean_embeddings = use_mean_embeddings
        
        # Choose appropriate image projector based on embedding type
        if use_mean_embeddings:
            # Simple projector for mean embeddings (1D vectors)
            self.image_projector = SimpleImageProjector(
                input_dim=image_input_dim,
                output_dim=embedding_dim,
                hidden_dim=image_hidden_dim,
                dropout=dropout
            )
        else:
            # Cross-attention projector for full embeddings (2D sequences)
            self.image_projector = CrossAttentionImageProjector(
                input_dim=image_input_dim,
                output_dim=embedding_dim,
                hidden_dim=image_hidden_dim,
                dropout=dropout
            )
        
        self.text_projector = TextProjector(
            input_dim=text_input_dim,
            output_dim=embedding_dim,
            hidden_dim=text_hidden_dim,
            dropout=dropout
        )
        
        # Learnable logit scale parameter initialized to standard CLIP temperature 1/0.07
        # Using log parameterization for numerical stability
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07, dtype=torch.float32)))
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CLIP training.
        
        Args:
            batch: Dictionary containing 'image_embedding' and 'text_embedding'
            
        Returns:
            Dictionary with projected embeddings and logits
        """
        image_features = batch['image_embedding']
        text_features = batch['text_embedding']

        # Project to shared space and normalize
        image_features = self.image_projector(image_features)
        text_features = self.text_projector(text_features)

        # Compute similarity matrix with learnable logit scale
        # Clamp after exp to preserve gradients
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        return {
            'image_features': image_features,
            'text_features': text_features,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'logit_scale': logit_scale
        }
    
    def compute_contrastive_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE).
        
        Args:
            outputs: Model outputs from forward pass
            
        Returns:
            Contrastive loss
        """
        logits_per_image = outputs['logits_per_image']
        logits_per_text = outputs['logits_per_text']
        
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # Cross-entropy loss for both directions
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2