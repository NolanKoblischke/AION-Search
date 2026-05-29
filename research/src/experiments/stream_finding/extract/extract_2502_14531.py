"""Extract Tables 3, 4 & 5 from paper.tex for 2502.14531

Table 3 contains the 15 primary analysis targets (streams, tails, 1 spiral arm)
with HMS coordinates. Tables 4 and 5 contain the supplementary catalog of tidal
features from WWFI Abell cluster sample (decimal degree coordinates).
"""

import re
from pathlib import Path

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from .base import StreamExtractor


def hms_to_deg(ra_hms: str, dec_hms: str) -> tuple[float, float]:
    """Convert HMS/DMS coordinate strings to decimal degrees."""
    coord = SkyCoord(ra_hms, dec_hms, unit=(u.hourangle, u.deg))
    return coord.ra.deg, coord.dec.deg


class Extract2502_14531(StreamExtractor):
    arxiv_id = "2502_14531"
    needs_resolution = False
    name_column = None

    def _extract_table3(self, content: str) -> list[dict]:
        """Extract Table 3 (primary features) with HMS coordinates."""
        match = re.search(
            r"\\begin\{tabular\}\{lccccc\}(.*?)\\end\{tabular\}",
            content,
            re.DOTALL,
        )
        if not match:
            return []

        rows = []
        lines = match.group(1).split("\\\\")

        for line in lines:
            line = re.sub(r"\\hline\s*", "", line).strip()
            if not line or line.startswith("%"):
                continue
            if "alpha" in line or "delta" in line:
                continue
            if "J2000" in line:
                continue
            if "(1)" in line and "(2)" in line and "(3)" in line:
                continue

            # Strip parentheses (the spiral arm row is wrapped in parens)
            line = line.strip("() ")

            parts = [p.strip() for p in line.split("&")]
            if len(parts) < 6:
                continue

            ra_hms = parts[0].strip()
            dec_hms = parts[1].strip()

            # Validate: must start with a digit (coordinate)
            if not ra_hms or not ra_hms[0].isdigit():
                continue

            # Strip citation commands from redshift
            z_raw = parts[2].strip()
            z_val = re.sub(r"\\cite[pt]?\{[^}]*\}", "", z_raw).strip()

            candidate = parts[3].strip()
            # Clean LaTeX from candidate (e.g. $z_\text{A2558}$)
            candidate = re.sub(r"\$[^$]*\$", candidate, candidate)

            angular_scale = parts[4].strip()
            feature = parts[5].strip()

            try:
                ra_deg, dec_deg = hms_to_deg(ra_hms, dec_hms)
            except Exception:
                continue

            rows.append({
                "RA_deg": f"{ra_deg:.6f}",
                "Dec_deg": f"{dec_deg:.6f}",
                "z": z_val,
                "Candidate": candidate,
                "Angular_Scale_kpc_arcsec": angular_scale,
                "Feature": feature,
            })

        return rows

    def _extract_tables45(self, content: str) -> list[dict]:
        """Extract Tables 4 & 5 (supplementary features) with decimal degree coords."""
        table_matches = re.findall(
            r"\\begin\{tabular\}\{c c c c c c\}(.*?)\\end\{tabular\}",
            content,
            re.DOTALL,
        )

        rows = []
        seen_coords = set()

        for table_content in table_matches:
            lines = table_content.split("\\\\")

            for line in lines:
                line = re.sub(r"\\hline\s*", "", line)
                line = line.replace("\\&", "<<AMP>>")
                stripped = line.strip()

                if not stripped or stripped.startswith("%"):
                    continue
                if "alpha" in stripped or "delta" in stripped:
                    continue
                if "J2000" in stripped:
                    continue
                if "(1)" in stripped and "(2)" in stripped and "(3)" in stripped:
                    continue

                parts = [p.strip() for p in stripped.split("&")]
                if len(parts) < 6:
                    continue

                ra = parts[0].strip()
                try:
                    float(ra)
                except ValueError:
                    continue

                dec = parts[1].strip()
                z = parts[2].strip()
                candidate = parts[3].strip()
                angular_scale = parts[4].strip()
                feature = parts[5].strip()
                feature = feature.replace("<<AMP>>", "&")

                coord_key = (ra, dec)
                if coord_key in seen_coords:
                    continue
                seen_coords.add(coord_key)

                rows.append({
                    "RA_deg": ra,
                    "Dec_deg": dec,
                    "z": z,
                    "Candidate": candidate,
                    "Angular_Scale_kpc_arcsec": angular_scale,
                    "Feature": feature,
                })

        return rows

    def extract(self, source_dir: Path) -> pd.DataFrame:
        tex_file = source_dir / "arxiv_source" / "paper.tex"

        with open(tex_file, "r") as f:
            content = f.read()

        table3_rows = self._extract_table3(content)
        tables45_rows = self._extract_tables45(content)

        all_rows = table3_rows + tables45_rows
        return pd.DataFrame(all_rows)
