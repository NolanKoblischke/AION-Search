"""Extract tidal feature catalog from Atkinson et al. 2013 (1301.4275)

CFHTLS tidal feature survey. Downloaded from VizieR catalog J/ApJ/765/28.
"""

from pathlib import Path

import pandas as pd

from .base import StreamExtractor


class Extract1301_4275(StreamExtractor):
    arxiv_id = "1301_4275"
    needs_resolution = False
    name_column = None

    def extract(self, source_dir: Path) -> pd.DataFrame:
        from astroquery.vizier import Vizier

        catalog = Vizier(row_limit=-1).get_catalogs("J/ApJ/765/28")
        table = catalog[0]
        df = table.to_pandas()
        df = df.rename(columns={"RAJ2000": "ra", "DEJ2000": "dec"})
        return df
