"""Model loading helpers for the controlled HSC lens experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.clip.models.clip_model import AIONSearchClipModel
from src.experiments.hsc_lens_rerank.constants import HF_MODEL_REPO, HF_MODEL_REVISION


def _build_model(config: dict[str, Any], device: str) -> AIONSearchClipModel:
    model = AIONSearchClipModel(
        image_input_dim=config["image_input_dim"],
        text_input_dim=config["text_input_dim"],
        embedding_dim=config["embedding_dim"],
        image_hidden_dim=config.get("image_hidden_dim", 768),
        text_hidden_dim=config.get("text_hidden_dim", 1024),
        dropout=config.get("dropout", 0.1),
        use_mean_embeddings=config.get("use_mean_embeddings", True),
    ).to(device)
    return model


def load_aion_search_model(model_source: str = HF_MODEL_REPO, device: str = "cpu") -> tuple[AIONSearchClipModel, dict[str, Any]]:
    """Load AION-Search from Hugging Face or a local PyTorch checkpoint.

    `model_source` defaults to the public Hugging Face model repository. Passing
    a local path keeps the original checkpoint format supported for internal
    provenance checks.
    """
    source_path = Path(model_source)
    if source_path.exists():
        checkpoint = torch.load(source_path, map_location=device, weights_only=False)
        config = checkpoint["model_config"]
        model = _build_model(config, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, config

    revision = HF_MODEL_REVISION if model_source == HF_MODEL_REPO else None
    config_path = hf_hub_download(model_source, "config.json", revision=revision)
    weights_path = hf_hub_download(model_source, "model.safetensors", revision=revision)

    with open(config_path) as f:
        config = json.load(f)

    model = _build_model(config, device)
    state_dict = load_file(weights_path, device=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config
