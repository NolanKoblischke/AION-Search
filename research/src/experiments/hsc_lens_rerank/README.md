# Controlled HSC lens re-ranking experiment

This directory contains the code for the re-ranking experiment on gravitational lens retrieval.

## Files

| File | Purpose |
| --- | --- |
| `prepare_sample.py` | Builds `data/evals/lens/hsc_lens_rerank/hsc_lens_rerank_eval_objects.csv` from the HSC lens FITS table and lens catalogs. |
| `build_aion_embeddings.py` | Generates `data/evals/lens/lens_aion_embeddings_hsc.hdf5`. |
| `evaluate_baseline.py` | Loads AION-Search from Hugging Face, ranks the controlled HSC set, counts lenses at k, and can save the top-k image cache. |
| `vlm_rerank.py` | Scores candidate HSC images with the lens prompt and re-ranks them. |
| `run_rerank_grid.py` | Runs the paper grid over GPT-4.1-nano, GPT-4.1-mini, GPT-4.1, and best-of-m values 1 and 5. |
| `plot_results.py` | Recreates the re-ranking cost/performance plot from saved experiment outputs. |

## Required data

These scripts are self-contained in code, but the full experiment starts from a large external FITS table containing the lens image catalog, the creation of which is described in the paper appendix and in the original AION-1 paper. Lens labels are assigned by cross-matching the FITS coordinates to the published lens catalogs included under `data/evals/lens/catalogs/`.

## Included controlled-sample record

The exact controlled sample used in the paper is included:

- `data/evals/lens/hsc_lens_rerank/hsc_lens_rerank_eval_objects.csv`
- `data/evals/lens/hsc_lens_rerank/hsc_lens_rerank_eval_summary.txt`

`hsc_lens_rerank_eval_objects.csv` contains the  `object_id`, `ra`, `dec`, and `lensgrade` values for the 200 confirmed lenses and 20,000 sampled non-lenses. `hsc_lens_rerank_eval_summary.txt` records the sampling seed, target counts, and grade distribution

## Reproduce from the controlled sample

Build or restore the controlled HSC sample:

```bash
uv run python -m src.experiments.hsc_lens_rerank.prepare_sample \
  --fits-file /path/to/lens_image_catalog_part_000.fits \
  --output-dir data/evals/lens/hsc_lens_rerank
```

Generate HSC AION embeddings for the controlled sample:

```bash
uv run python -m src.experiments.hsc_lens_rerank.build_aion_embeddings \
  --input /path/to/lens_image_catalog_part_000.fits \
  --lens-csv data/evals/lens/hsc_lens_rerank/hsc_lens_rerank_eval_objects.csv \
  --output-hsc data/evals/lens/lens_aion_embeddings_hsc.hdf5
```

Evaluate AION-Search and save the top-1000 HSC image cache for VLM re-ranking:

```bash
uv run python -m src.experiments.hsc_lens_rerank.evaluate_baseline \
  --model-source astronolan/aion-search \
  --eval-name lens_hsc \
  --k-values 10 20 30 50 100 500 1000 \
  --save-k 1000 \
  --output-file data/experiments/rerank/best_lens_hsc_model_top1000_lens_hsc.hdf5 \
  --lens-csv data/evals/lens/hsc_lens_rerank/hsc_lens_rerank_eval_objects.csv \
  --fits-path /path/to/lens_image_catalog_part_000.fits
```

Run the paper re-ranking grid:

```bash
uv run python -m src.experiments.hsc_lens_rerank.run_rerank_grid \
  --model-source astronolan/aion-search \
  --top-k-values 1000 \
  --n-runs 3 \
  --best-of-m-values 1 5 \
  --cache-hdf5 data/experiments/rerank/best_lens_hsc_model_top1000_lens_hsc.hdf5
```

The default model grid is `gpt-4.1-nano`, `gpt-4.1-mini`, and `gpt-4.1`.
Model names and API prices are read from the shared `src/utils/models.jsonl` file.

Create the plot from saved outputs:

```bash
uv run python -m src.experiments.hsc_lens_rerank.plot_results \
  --experiment-dir data/experiments/rerank/multi_model_optimized_20250812_232602 \
  --eval-summary data/eval_results/lens_hsc_aion_baseline_results_20250804_114443.txt
```