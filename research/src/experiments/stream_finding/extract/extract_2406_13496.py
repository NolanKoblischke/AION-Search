"""Extract Tables 1 & 2 from table1.tex and table2.tex for 2406.13496

Table 1: Tidal structures (streams, shells, tails, etc.)
Table 2: Structural features (lopsided disks, warps, polar rings)
"""

import re
from pathlib import Path

import pandas as pd

from .base import StreamExtractor


class Extract2406_13496(StreamExtractor):
    arxiv_id = "2406_13496"
    needs_resolution = False
    name_column = None

    def extract(self, source_dir: Path) -> pd.DataFrame:
        base_dir = source_dir / "arxiv_source"

        table1_rows = self._parse_table_file(base_dir / "table1.tex")
        table2_rows = self._parse_table_file(base_dir / "table2.tex")

        return pd.DataFrame(table1_rows + table2_rows)

    def _parse_table_file(self, tex_file: Path) -> list[dict]:
        with open(tex_file, "r") as f:
            content = f.read()

        table_match = re.search(
            r"\\begin\{tabular\}\s*\{[^}]+\}(.*?)\\end\{tabular\}",
            content,
            re.DOTALL,
        )
        if not table_match:
            raise ValueError(f"Could not find table in {tex_file}")

        table_content = table_match.group(1)
        lines = table_content.split("\\\\")

        rows = []
        for line in lines:
            line = re.sub(r"\\hline\s*", "", line).strip()

            if not line or line.startswith("%"):
                continue
            if "Name" in line and "R.A." in line:
                continue
            if line.startswith("&") and "degree" in line:
                continue
            if "\\parbox" in line or "\\textbf" in line:
                continue

            parts = [p.strip() for p in line.split("&")]
            if len(parts) < 7:
                continue

            name = parts[0].strip()
            if not name:
                continue
            if name.startswith("\\") and "_" not in name:
                continue

            name_clean = name.replace("$^\\star$", "^").replace("$", "").strip()
            name_clean = name_clean.replace("\\_", "_")

            ra = parts[1].strip()
            dec = parts[2].strip()
            pa = parts[3].strip()
            structure = parts[4].strip()
            q = parts[5].strip()
            c0 = parts[6].strip()

            rows.append({
                "Name": name_clean,
                "RA_deg": ra,
                "Dec_deg": dec,
                "PA_deg": pa,
                "Structure": structure,
                "q": q,
                "C0": c0,
            })

        return rows
