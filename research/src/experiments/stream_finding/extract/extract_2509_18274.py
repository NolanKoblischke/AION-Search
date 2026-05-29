"""Extract tidal feature candidates from Desmons et al. HSC-SSP classifications (2509.18274)

FITS catalog with ML scores and visual classifications for 34,331 galaxies.
We keep galaxies that are either visually confirmed to have tidal features
(tidal_feature_vis == 1) or have a high ML score (P_merge > 0.5).
"""

from pathlib import Path

import pandas as pd
from astropy.io import fits

from .base import StreamExtractor


class Extract2509_18274(StreamExtractor):
    arxiv_id = "2509_18274"
    needs_resolution = False
    name_column = None

    def extract(self, source_dir: Path) -> pd.DataFrame:
        fits_file = source_dir / "Desmons_HSC_SSP_TF_Classifications.fits"

        from astropy.table import Table

        table = Table.read(fits_file)
        df = table.to_pandas()

        mask = (df["tidal_feature_vis"] == 1) | (df["P_merge"] > 0.5)
        df = df[mask].reset_index(drop=True)
        return df
