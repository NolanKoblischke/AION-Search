"""
Constants for GZ10 pipeline experiments.
"""

CLASS_LABELS = {
    0: "Disturbed Galaxies",
    1: "Merging Galaxies",
    2: "Round Smooth Galaxies",
    3: "In-between Round Smooth Galaxies",
    4: "Cigar Shaped Smooth Galaxies",
    5: "Barred Spiral Galaxies",
    6: "Unbarred Tight Spiral Galaxies",
    7: "Unbarred Loose Spiral Galaxies",
    8: "Edge-on Galaxies without Bulge",
    9: "Edge-on Galaxies with Bulge",
}

DATA_DIR = "data/gz10"
HF_DATASET_REPO = "astronolan/galaxy10-aion"
HF_DATASET_REVISION = "ffe5949ad4acfe29a996ad8592501630051fe68d"
HF_MODEL_REPO = "astronolan/aion-search"
HF_MODEL_REVISION = "e6d56ee28b6768f4e3e4494b2c0b32a00abb2594"
CLIP_CHECKPOINT = f"{DATA_DIR}/aionsearchmodel.pt"

# Output files
IMAGE_EMBEDDINGS_PARQUET = f"{DATA_DIR}/gz10_image_embeddings.parquet"
LABEL_EMBEDDINGS_PARQUET = f"{DATA_DIR}/gz10_label_embeddings.parquet"
AION_CLASSIFICATIONS_PARQUET = f"{DATA_DIR}/gz10_aion_classifications.parquet"
