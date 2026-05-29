"""Extract Table 1 from martinezdelgado_aanda.tex for 2104.06071"""

import re
import csv
from pathlib import Path

import pandas as pd

from .base import StreamExtractor


class Extract2104_06071(StreamExtractor):
    arxiv_id = "2104_06071"
    needs_resolution = True
    name_column = "Name"

    def extract(self, source_dir: Path) -> pd.DataFrame:
        tex_file = source_dir / "arxiv_source" / "martinezdelgado_aanda.tex"

        with open(tex_file, "r") as f:
            content = f.read()

        table_match = re.search(
            r"\\begin\{tabular\}\{llrrrr@\{.*?\}lrr@\{.*?\}l\}(.*?)\\end\{tabular\}",
            content,
            re.DOTALL,
        )
        if not table_match:
            raise ValueError("Could not find table")

        table_content = table_match.group(1)
        lines = table_content.split("\\\\")

        rows = []
        for line in lines:
            line = re.sub(r"\\hline\s*", "", line).strip()

            if not line or line.startswith("%"):
                continue
            if "Name" in line and "Type" in line:
                continue
            if "\\multicolumn" in line:
                continue
            if "mag" in line and "km" in line:
                continue

            parts = [p.strip() for p in line.split("&")]
            if len(parts) < 9:
                continue

            name = parts[0].strip()
            if not name:
                continue
            if name.startswith("\\") and not any(c.isalnum() for c in name[1:5]):
                continue

            name = name.replace("$-$", "-").replace("$", "").strip()
            morph_type = parts[1].strip()
            t_val = parts[2].replace("$", "").replace("\\pm", "+/-").strip()
            b_t = parts[3].replace("$", "").replace("\\pm", "+/-").strip()
            k_t = parts[4].replace("$", "").replace("\\pm", "+/-").strip()
            v_lg = parts[5].strip()
            v_lg_err = parts[6].strip()
            d_mpc = parts[7].strip()
            fov1 = parts[8].strip()
            fov2 = parts[9].strip() if len(parts) > 9 else fov1
            fov = f"{fov1}x{fov2}"

            rows.append({
                "Name": name,
                "Type": morph_type,
                "T": t_val,
                "B_T": b_t,
                "K_T": k_t,
                "V_LG": v_lg,
                "V_LG_err": v_lg_err,
                "D_Mpc": d_mpc,
                "FOV_arcmin": fov,
            })

        return pd.DataFrame(rows)
