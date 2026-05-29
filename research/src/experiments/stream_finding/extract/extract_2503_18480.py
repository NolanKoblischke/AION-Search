"""Extract Table C2 from Table_C2_features_streams_tails.txt for 2503.18480

Fixed-width format file containing annotated tidal tails and streams
from CFHT deep imaging (MATLAS, UNIONS/CFIS, VESTIGE, NGVS).
"""

from pathlib import Path

import pandas as pd

from .base import StreamExtractor


class Extract2503_18480(StreamExtractor):
    arxiv_id = "2503_18480"
    needs_resolution = True
    name_column = "Galaxy"

    def extract(self, source_dir: Path) -> pd.DataFrame:
        data_file = source_dir / "Table_C2_features_streams_tails.txt"

        rows = []
        with open(data_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                # Fixed-width format based on ReadMe:
                # Bytes 1-3: Feature#, 5-10: Type, 12-19: Galaxy, 21-25: Length,
                # 27-32: Area, 34-40: Width, 42-45: SB, 47-52: Flux, 54-72: Total_galaxy_mag
                feature_num = line[0:3].strip()
                feature_type = line[4:10].strip()
                galaxy = line[11:19].strip()
                length_kpc = line[20:25].strip()
                area_kpc2 = line[26:32].strip()
                width_kpc = line[33:40].strip()
                sb = line[41:45].strip()
                flux_pct = line[46:52].strip()
                total_mag = line[53:72].strip()

                rows.append({
                    "Feature_ID": feature_num,
                    "Type": feature_type,
                    "Galaxy": galaxy,
                    "Length_kpc": length_kpc,
                    "Area_kpc2": area_kpc2,
                    "Width_kpc": width_kpc,
                    "SB_mag_arcsec2": sb,
                    "Flux_pct": flux_pct,
                    "Total_galaxy_mag": total_mag,
                })

        return pd.DataFrame(rows)
