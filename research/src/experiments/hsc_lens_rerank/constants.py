"""Shared constants for the controlled HSC lens re-ranking experiment."""

DEFAULT_FITS_FILE = "lens_image_catalog_part_000.fits"
DEFAULT_LENS_CSV = "data/evals/lens/hsc_lens_rerank/hsc_lens_rerank_eval_objects.csv"
DEFAULT_MODELS_FILE = "src/utils/models.jsonl"
HF_MODEL_REPO = "astronolan/aion-search"
HF_MODEL_REVISION = "e6d56ee28b6768f4e3e4494b2c0b32a00abb2594"

LENS_QUESTION = (
    "Does this galaxy image display signs of gravitational lensing? Rank 1-10 "
    "where 10 means you are entirely sure there are signs of gravitational "
    "lensing and 1 being you are entirely sure there are no signs of "
    "gravitational lensing."
)
