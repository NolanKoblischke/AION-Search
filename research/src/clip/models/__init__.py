"""CLIP model architecture for galaxy embeddings."""

from .clip_model import AIONSearchClipModel
from .projections import CrossAttentionImageProjector, TextProjector

__all__ = ["AIONSearchClipModel", "CrossAttentionImageProjector", "TextProjector"]