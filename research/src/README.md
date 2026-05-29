# AION-Search Training Provenance

This directory contains code for the numbered AION-Search training pipeline.

The paper training set was built from two collection passes:

1. an initial pass with 20,000 HSC galaxies and 80,000 Legacy Survey galaxies
2. a second pass with 100,000 HSC galaxies and 100,000 Legacy Survey galaxies

Those passes were combined before captioning and training, giving the 300,000 sampled images: 120,000 HSC and 180,000 Legacy. After caption generation and benchmark-overlap filtering, 255,948 galaxies were used for model training.

## External Inputs

Full raw regeneration requires:

- Multimodal Universe HSC PDR3 Wide and Legacy Survey DR10 South data in the expected HDF5 layout. Start from the Multimodal Universe Hugging Face organization: <https://huggingface.co/MultimodalUniverse>
- AION-1-Base image encoder access for generating image embeddings.
- OpenAI API access for GPT-4.1-mini caption generation, GPT-4.1-nano summary generation, and `text-embedding-3-large` text embeddings.
- GPU compute for AION embedding generation and CLIP training.

The example paths in `01_collect_galaxies.py`, such as `hsc/pdr3_wide_21` and `MMU/legacysurvey/dr10_south_21`, represent the expected MMU/AstroPile filesystem layout. Replace them with local paths to the corresponding MMU datasets.

## Numbered Pipeline

Run commands from `research/`.

### 1. Collect raw galaxies

```bash
uv run python -m src.01_collect_galaxies \
  --output data/processed/galaxy_data.hdf5
```

### 2. Submit caption-generation batch jobs

```bash
uv run python -m src.02a_generate_descriptions_batch \
  --input data/processed/galaxy_data.hdf5 \
  --prompt src/prompts/general_promptv4.txt \
  --output-dir data/processed
```

### 3. Download and materialize caption results

After the batch jobs complete, use `check_batch.py` on the generated batch-info JSON file:

```bash
uv run python -m src.check_batch path/to/batch_info.json
```

This downloads OpenAI batch results and writes a generated `galaxy_descriptions_*.hdf5` file under `data/processed/`.

### 4. Submit summary-generation batch jobs

```bash
uv run python -m src.03a_generate_augmented_descriptions_batch \
  --input data/processed/galaxy_descriptions.hdf5 \
  --output-dir data/processed
```

Then materialize the summary results with:

```bash
uv run python -m src.check_batch path/to/summary_batch_info.json \
  --base-hdf5 data/processed/galaxy_descriptions.hdf5
```

This writes a generated `galaxy_descriptions_augmented_*.hdf5` file.

### 5. Generate text embeddings

```bash
uv run python -m src.04_generate_text_embeddings \
  --input data/processed/galaxy_descriptions_augmented.hdf5 \
  --galaxy-data data/processed/galaxy_data.hdf5 \
  --output-dir data/processed
```

This writes `data/processed/galaxy_text_embeddings.hdf5`.

### 6. Generate unified image/text embeddings

```bash
uv run python -m src.05_generate_unified_embeddings \
  --galaxy-data data/processed/galaxy_data.hdf5 \
  --text-embeddings data/processed/galaxy_text_embeddings.hdf5 \
  --output data/processed/galaxy_embeddings_unified.parquet
```

This generates AION image embeddings and writes the unified parquet expected by training.

### 7. Train the contrastive model

```bash
uv run python -m src.06_train_clip \
  --config configs/baseline_original.yaml
```

## Frozen Public Artifacts

The preferred reproducibility route is to use the frozen public artifacts rather than rerunning the full raw MMU and OpenAI pipeline:

| Paper object | Public artifact | Notes |
| --- | --- | --- |
| Released AION-Search model | `astronolan/aion-search` | Model weights used by retrieval, GZ10, and HSC-lens code. |
| Generated captions, summaries, text embeddings, AION mean embeddings | `astronolan/galaxy-descriptions` | Contains all galaxies that successfully received descriptions. This is larger than the final paper-training subset. |
| Retrieval benchmark datasets | AION-Search Hugging Face collection | Public retrieval artifacts used for GZ-DECaLS and lens evaluations. |

The `astronolan/galaxy-descriptions` dataset contains roughly 276,000 rows because it includes all galaxies that successfully received descriptions. The manuscript's 255,948 training galaxies are the subset used after filtering out galaxies overlapping the retrieval benchmarks. Those benchmark artifacts are public in the AION-Search Hugging Face collection: <https://huggingface.co/collections/astronolan/aion-search>