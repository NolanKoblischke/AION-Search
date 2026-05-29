# HSC lens evaluation catalogs

These small catalog files are inputs to `src/experiments/hsc_lens_rerank/prepare_sample.py`.

Usage in the controlled HSC lens experiment:

- `masterlens.csv` and `hsc_lenses.csv` define the confirmed-lens positive set.
- `masterlens.csv`, `hsc_lenses.csv`, `lenscat.csv`, `stein_training_lenses_legacy.tsv`, and `stein_new_lenses_legacy.tsv` are all used to filter sampled non-lenses, excluding objects within 5 arcsec of known lenses or lens candidates.

The HSC image FITS table itself is not included in this Git repository. See the paper appendix for the provenance of the HSC lens parent sample and use the paths documented in `src/experiments/hsc_lens_rerank/README.md` when regenerating the controlled sample.

