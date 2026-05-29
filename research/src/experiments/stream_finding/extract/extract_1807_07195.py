"""Extract dwarf galaxy tidal feature catalog from Hood et al. 2018 (1807.07195)

Downloaded from VizieR catalog J/ApJS/237/36.
"""

from pathlib import Path

import pandas as pd

from .base import StreamExtractor


class Extract1807_07195(StreamExtractor):
    arxiv_id = "1807_07195"
    needs_resolution = False
    name_column = None

    def extract(self, source_dir: Path) -> pd.DataFrame:
        from astroquery.vizier import Vizier

        catalog = Vizier(row_limit=-1).get_catalogs("J/ApJS/237/36")
        table = catalog[0]
        df = table.to_pandas()
        df = df.rename(columns={"RAJ2000": "ra", "DEJ2000": "dec"})
        return df
