"""CLIP model architecture for galaxy embeddings."""

from .clip_model import GalaxyClipModel
from .projections import CrossAttentionImageProjector, TextProjector

__all__ = ["GalaxyClipModel", "CrossAttentionImageProjector", "TextProjector"]