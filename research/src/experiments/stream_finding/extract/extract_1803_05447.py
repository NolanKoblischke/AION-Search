"""Extract RESOLVE tidal feature catalog from Hood et al. 2018 (1803.05447).

Downloaded from VizieR catalog J/ApJ/857/144/table2.
"""

from pathlib import Path

import pandas as pd

from .base import StreamExtractor


class Extract1803_05447(StreamExtractor):
    arxiv_id = "1803_05447"
    needs_resolution = False
    name_column = None

    def extract(self, source_dir: Path) -> pd.DataFrame:
        from astroquery.vizier import Vizier

        catalog = Vizier(row_limit=-1).get_catalogs("J/ApJ/857/144/table2")
        table = catalog[0]
        df = table.to_pandas()
        df = df.rename(columns={"_RA": "ra", "_DE": "dec"})
        return df
