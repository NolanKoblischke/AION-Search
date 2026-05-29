# Unified Retrieval Evaluation

Evaluates AION and AION-Search retrieval performance on GZ-DECaLS (spirals/mergers) and lens datasets using nDCG@10.

Run all commands from the `research/` directory.

## Tasks

| Task | Embedding | Query Type | Dataset | Metrics |
|---|---|---|---|---|
| `aion_gz5` | AION (768-dim) | Image-to-image | GZ-DECaLS | Spiral, Merger |
| `aion_lens` | AION (768-dim) | Image-to-image | Lens parent sample | Lens |
| `aion_search_gz5` | AION-Search (1024-dim) | Text-to-image | GZ-DECaLS | Spiral, Merger |
| `aion_search_lens` | AION-Search (1024-dim) | Text-to-image | Lens parent sample | Lens |

## Quick Start (HuggingFace)

The easiest way to run evaluation is with `--use-hf`, which downloads pre-computed embeddings, relevance labels, and k-fold assignments directly from HuggingFace:

- GZ5: [astronolan/gz-decals-embeddings](https://huggingface.co/datasets/astronolan/gz-decals-embeddings)
- Lens: [astronolan/lens-retrieval-ls-embeddings](https://huggingface.co/datasets/astronolan/lens-retrieval-ls-embeddings)

For search tasks, the CLIP model weights are also downloaded automatically from [astronolan/aion-search](https://huggingface.co/astronolan/aion-search) (unless `--model-path` is provided).

### Full dataset evaluation (default)

```bash
# GZ5
uv run python -m src.experiments.retrieval.eval_retrieval --task aion_gz5 --use-hf
uv run python -m src.experiments.retrieval.eval_retrieval --task aion_search_gz5 --use-hf

# Lens
uv run python -m src.experiments.retrieval.eval_retrieval --task aion_lens --use-hf
uv run python -m src.experiments.retrieval.eval_retrieval --task aion_search_lens --use-hf
```

### 10-fold cross-validation

Add `--kfold` to split evaluation across 10 folds (using the `kfold` column in the HF datasets):

```bash
# GZ5
uv run python -m src.experiments.retrieval.eval_retrieval --task aion_gz5 --use-hf --kfold
uv run python -m src.experiments.retrieval.eval_retrieval --task aion_search_gz5 --use-hf --kfold

# Lens
uv run python -m src.experiments.retrieval.eval_retrieval --task aion_lens --use-hf --kfold
uv run python -m src.experiments.retrieval.eval_retrieval --task aion_search_lens --use-hf --kfold
```

## Output

Each run produces:
- `*_results.json` -- nDCG@k summary and per-fold breakdown (if `--kfold`)
- `*_objects.csv` -- per-object rankings with ra, dec, similarity, and relevance scores

K-fold CSVs can be generated with:

```bash
uv run python -m src.experiments.retrieval.make_kfolds --n-folds 10
```

## Reranking with GPT-4.1

Requires `OPENAI_API_KEY` in `.env`. Run on full-dataset results only.

### Download images

```bash
uv run python -m src.experiments.retrieval.download_images \
    --objects-csv <output-dir>/default/aion_search_gz5_kfold_spiral_objects.csv \
    --folds 0 \
    --output-dir data/retrieval_results/full/rerank/default/spiral/images
```

### Rerank

```bash
uv run python -m src.experiments.retrieval.rerank \
    --images-dir data/retrieval_results/full/rerank/default/spiral/images \
    --dataset spiral \
    --model gpt-4.1
```
