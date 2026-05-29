"""Extract Table 1 from 45003corr.tex for 2407.20483

Handles entries with inherited coordinates (IC 1657-2, IC 1657-3, PGC 597851-2).
"""

import re
from pathlib import Path

import pandas as pd

from .base import StreamExtractor


class Extract2407_20483(StreamExtractor):
    arxiv_id = "2407_20483"
    needs_resolution = False
    name_column = None

    def extract(self, source_dir: Path) -> pd.DataFrame:
        tex_file = source_dir / "arxiv_source" / "45003corr.tex"

        with open(tex_file, "r") as f:
            content = f.read()

        table_match = re.search(
            r"\\begin\{tabular\}\{lccccccccccc\}(.*?)\\end\{tabular\}",
            content,
            re.DOTALL,
        )
        if not table_match:
            raise ValueError("Could not find table")

        table_content = table_match.group(1)
        lines = table_content.split("\\\\")

        rows = []
        last_ra = None
        last_dec = None
        last_d = None
        last_type = None

        for line in lines:
            line = line.strip()

            if not line or line.startswith("\\hline") or line.startswith("%"):
                continue
            if "Host" in line and "RA" in line:
                continue
            if line.strip() == "& deg & deg":
                continue
            if "kpc" in line and "Mpc" in line and "deg" in line:
                continue

            parts = [p.strip() for p in line.split("&")]
            if len(parts) < 11:
                continue

            host = parts[0].strip()
            if not host or host.startswith("\\"):
                continue

            ra = parts[1].strip()
            dec = parts[2].strip()
            d_mpc = parts[3].strip()
            morph_type = parts[4].strip()

            if ra and dec:
                last_ra = ra
                last_dec = dec
                last_d = d_mpc
                last_type = morph_type
            else:
                ra = last_ra
                dec = last_dec
                d_mpc = last_d if not d_mpc else d_mpc
                morph_type = last_type if not morph_type else morph_type

            d_kpc = parts[5].strip()
            w_kpc = parts[6].strip()
            morphology = parts[7].strip()
            dsi_r = parts[8].strip()
            dsi_g = parts[9].strip()
            dsi_z = parts[10].strip()
            reported = parts[11].strip() if len(parts) > 11 else ""

            rows.append({
                "Host": host,
                "RA_deg": ra,
                "Dec_deg": dec,
                "D_Mpc": d_mpc,
                "Type": morph_type,
                "d_kpc": d_kpc,
                "w_kpc": w_kpc,
                "Morphology": morphology,
                "DSI_r": dsi_r,
                "DSI_g": dsi_g,
                "DSI_z": dsi_z,
                "Reported": reported,
            })

        return pd.DataFrame(rows)
