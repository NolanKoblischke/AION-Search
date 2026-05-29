"""Extract STRRINGS data from Table1.csv and resolve coords via SGA-2020.fits (2508.02154)"""

from pathlib import Path

import pandas as pd
from astropy.io import fits

from .base import StreamExtractor


class Extract2508_02154(StreamExtractor):
    arxiv_id = "2508_02154"
    needs_resolution = False  # Uses FITS lookup, not SIMBAD
    name_column = None

    def extract(self, source_dir: Path) -> pd.DataFrame:
        input_csv = source_dir / "Table1.csv"
        fits_file = source_dir / "SGA-2020.fits"

        df = pd.read_csv(input_csv)

        with fits.open(fits_file) as hdul:
            ellipse = hdul["ELLIPSE"].data

            galaxy_coords = {
                row["GALAXY"]: (row["RA"], row["DEC"])
                for row in ellipse
            }
            group_coords = {
                row["GROUP_NAME"]: (row["RA"], row["DEC"])
                for row in ellipse
                if row["GROUP_PRIMARY"]
            }

        ra_list = []
        dec_list = []
        missing = []

        for name in df["Name"]:
            if name.endswith("_GROUP"):
                if name in group_coords:
                    ra, dec = group_coords[name]
                else:
                    ra, dec = None, None
                    missing.append(name)
            else:
                if name in galaxy_coords:
                    ra, dec = galaxy_coords[name]
                else:
                    ra, dec = None, None
                    missing.append(name)

            ra_list.append(ra)
            dec_list.append(dec)

        df["RA"] = ra_list
        df["DEC"] = dec_list

        if missing:
            print(f"  Warning: {len(missing)} names not found in SGA-2020.fits")

        return df

    def run(self, data_root: Path) -> None:
        """Custom run that produces both streams.csv and all_radec.csv."""
        source = self.source_dir(data_root)
        output = self.output_dir(data_root)
        output.mkdir(parents=True, exist_ok=True)

        df = self.extract(source)

        # Save all_radec.csv (all features)
        all_radec_path = output / "all_radec.csv"
        df.to_csv(all_radec_path, index=False)
        print(f"  Wrote {len(df)} rows to {all_radec_path}")

        # Save streams.csv (streams only)
        streams_df = df[df["Majority feature"] == "Streams"].copy()
        streams_path = output / "streams.csv"
        streams_df.to_csv(streams_path, index=False)
        print(f"  Wrote {len(streams_df)} rows to {streams_path}")
