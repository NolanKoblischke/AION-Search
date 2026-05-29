# GalaxyBench Galaxy Zoo Sample

This directory documents the input data used for the Figure 1 Galaxy Zoo description benchmark.

## Artifacts

- `gz5_legacysurvey_images.hdf5`: raw parent artifact used only when regenerating the 64-galaxy benchmark selection from scratch. This file is not included in this repository.
- `gz5_selected_galaxies.hdf5`: legacy local output consumed by `galaxybench/eval/combined/` when running new VLM evaluations.
- `selected_galaxies_summary.json`: small provenance summary for the 64 selected galaxies. It records 64 galaxies across 21 confident Galaxy Zoo decision-tree paths.

## Public Frozen Input

The exact 64-galaxy benchmark input is published on Hugging Face:

- Repository: `astronolan/galaxy-description-benchmark`
- Revision: `ebb13986d04b6b5e47529fb1fc68761839bffd75`
- File: `data/data.parquet`

This dataset contains the selected images, object IDs, RA/Dec, Galaxy Zoo vote columns, and `decision_tree` labels. It is the recommended public input for reproducing Figure 1 without access to the raw parent HDF5.

To materialize the legacy HDF5 expected by the current GalaxyBench loader:

```bash
cd research
uv run python -m galaxybench.dataprep.galaxyzoo.download_from_hf
```

This creates `galaxybench/dataprep/galaxyzoo/gz5_selected_galaxies.hdf5`.

## From-Scratch Selection

If `data/gz5_legacysurvey_images.hdf5` is available, the selected sample can be regenerated with:

```bash
cd research
uv run python -m galaxybench.dataprep.galaxyzoo.prep
```

The selection keeps galaxies with complete confident decision-tree paths using:

- maximum 5 galaxies per unique decision-tree path
- at least 10 votes at each selected node
- at least 0.7 agreement at each selected node

## Parent Artifact Provenance

`gz5_legacysurvey_images.hdf5` is a local parent artifact derived from the AION-1 Galaxy Zoo retrieval benchmark described in the AION-1 paper (arXiv:2510.17960). That benchmark starts from the Galaxy Zoo-DECaLS catalog (Walmsley et al. 2022), which provides citizen-science morphology votes for Legacy Survey galaxies (Dey et al. 2019). The AION-1 construction removes objects with fewer than three volunteer votes and cross-matches Galaxy Zoo-DECaLS with the Legacy Survey Southern Galactic Cap, yielding roughly 171,000 galaxies.

For the AION-1 spiral/merger retrieval protocol, high-confidence query galaxies are those where the relevant volunteer vote fraction exceeds 90%, giving roughly 25,000 spiral queries and 700 merger queries. Retrieved candidates are scored with soft relevance labels equal to the Galaxy Zoo vote fraction for the queried class, so a galaxy with 70% spiral votes receives relevance 0.7 for a spiral query.

This repository does not include the original parent-HDF5 build script or the full `gz5_legacysurvey_images.hdf5` artifact. The public frozen 64-galaxy derivative does exist above. The from-scratch selector in `prep.py` documents how the 64-image GalaxyBench sample is chosen once the parent HDF5 is available.