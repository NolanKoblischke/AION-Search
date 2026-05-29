# AION-Search: Semantic search for galaxy images using AI-generated captions

[![arXiv](https://img.shields.io/badge/arXiv-2512.11982-b31b1b.svg)](https://arxiv.org/abs/2512.11982)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://aion-search.github.io)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HuggingFace-yellow.svg)](https://huggingface.co/collections/astronolan/aion-search)
[![Demo](https://img.shields.io/badge/%F0%9F%9A%80%20Demo-HuggingFace-blue.svg)](https://astronolan-aion-search.hf.space/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Citation](https://img.shields.io/badge/Citation-BibTeX-orange.svg)](#citation)

AION-Search is a text-based search engine for galaxy images trained from GPT-4.1-mini descriptions and summaries of HSC and Legacy Survey galaxies. The released model uses the 255,948-galaxy training subset described in the paper after benchmark-overlap filtering.

🔭 **Use AION-Search now!**  
Try the live web demo: [AION-Search App](https://astronolan-aion-search.hf.space/)

📚 **Checkout our results**  
More details at: [Project Page](https://aion-search.github.io/)

📦 **Explore Our Datasets**  
Access all data products (embeddings and captions): [HuggingFace Datasets](https://huggingface.co/collections/astronolan/aion-search)

## Quick Start

### Installation

```bash
git clone https://github.com/NolanKoblischke/AION-Search.git
cd AION-Search
pip install -e . # or uv pip install -e .
```

### Requirements

This package requires an **OpenAI API key** for generating text embeddings.

1. Get your API key at [platform.openai.com](https://platform.openai.com)
2. Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=<your-api-key>
   ```

### Usage

```python
from aionsearch import AIONSearchClipModel

# Load pretrained model from HuggingFace
model = AIONSearchClipModel.from_pretrained()

# Project AION image embeddings into shared space
aion_embedding = # Embedding of an image using github.com/PolymathicAI/AION
projected_image = model.image_projector(aion_embedding)  # (batch, 768) -> (batch, 1024)

# Project OpenAI text embeddings into shared space  
text_embedding = # Embedding of text using text-embedding-3-large
projected_text = model.text_projector(text_embedding)    # (batch, 3072) -> (batch, 1024)

# Compute similarity for semantic search
similarity = projected_image @ projected_text.T
```

See [`examples/quick_start.ipynb`](examples/quick_start.ipynb) for a complete walkthrough that downloads a galaxy image, generates embeddings with AION, and performs text-to-image similarity search.

---

## Research Code

The [`research/`](research/) directory contains the paper-facing experiment code and small frozen artifacts needed to reproduce the main analyses where redistribution is practical. It includes:

- retrieval experiments for spirals, mergers, and gravitational lenses in [`research/src/experiments/retrieval/`](research/src/experiments/retrieval/)
- the controlled HSC lens re-ranking experiment in [`research/src/experiments/hsc_lens_rerank/`](research/src/experiments/hsc_lens_rerank/)
- Galaxy10 zero-shot and MLP-probe experiments in [`research/src/experiments/gz10/`](research/src/experiments/gz10/)
- stream-candidate catalog comparison in [`research/src/experiments/stream_finding/`](research/src/experiments/stream_finding/)
- selected cached tables, labels, catalogs, and README assets under [`research/data/`](research/data/) and [`research/assets/`](research/assets/)

The research environment can be installed from [`research/pyproject.toml`](research/pyproject.toml):

```bash
cd research
uv sync
```

Some large inputs and external services are intentionally not vendored into the Git repository. The experiment READMEs document which public Hugging Face artifacts, raw survey products, or API credentials are required for each workflow.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{koblischke2025semantic,
      title={Semantic Search for 100M+ Galaxy Images Using AI-Generated Captions}, 
      author={Nolan Koblischke and Liam Parker and Francois Lanusse and Jo Bovy and Irina Espejo and Shirley Ho},
      year={2025},
      eprint={2512.11982},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2512.11982}, 
}
```
