"""AION-Search model loading helpers for the GZ10 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.clip.models.clip_model import AIONSearchClipModel
from src.experiments.gz10.constants import HF_MODEL_REPO, HF_MODEL_REVISION


def _build_clip_model(config: dict, device: str) -> AIONSearchClipModel:
    return AIONSearchClipModel(
        image_input_dim=config["image_input_dim"],
        text_input_dim=config["text_input_dim"],
        embedding_dim=config["embedding_dim"],
        image_hidden_dim=config.get("image_hidden_dim", 768),
        text_hidden_dim=config.get("text_hidden_dim", 1024),
        dropout=config.get("dropout", 0.1),
        use_mean_embeddings=config.get("use_mean_embeddings", True),
    ).to(device)


def load_clip_model(model_path: str, device: str, hf_model_repo: str = HF_MODEL_REPO):
    """Load AION-Search from a local checkpoint or the public HF model repo.

    This follows the same acquisition pattern as the retrieval experiments:
    if `model_path` exists, load the legacy PyTorch checkpoint; otherwise
    download `config.json` and `model.safetensors` from `hf_model_repo`.
    """
    local_path = Path(model_path)
    if local_path.exists():
        checkpoint = torch.load(local_path, map_location=device, weights_only=False)
        config = checkpoint["model_config"]
        model = _build_clip_model(config, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, config

    revision = HF_MODEL_REVISION if hf_model_repo == HF_MODEL_REPO else None
    config_path = hf_hub_download(hf_model_repo, "config.json", revision=revision)
    weights_path = hf_hub_download(hf_model_repo, "model.safetensors", revision=revision)

    with open(config_path) as f:
        config = json.load(f)

    model = _build_clip_model(config, device)
    state_dict = load_file(weights_path, device=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config
