"""Extract Table 1 from aatemplate.tex for 2209.08636"""

import re
from pathlib import Path

import pandas as pd

from .base import StreamExtractor


class Extract2209_08636(StreamExtractor):
    arxiv_id = "2209_08636"
    needs_resolution = True
    name_column = "Host"

    def extract(self, source_dir: Path) -> pd.DataFrame:
        tex_file = source_dir / "arxiv_source" / "aatemplate.tex"

        with open(tex_file, "r") as f:
            content = f.read()

        table_match = re.search(
            r"\\begin\{tabular\}\{lcccccccc\}(.*?)\\end\{tabular\}",
            content,
            re.DOTALL,
        )
        if not table_match:
            raise ValueError("Could not find table")

        table_content = table_match.group(1)
        lines = table_content.split("\\\\")

        rows = []
        for line in lines:
            if "%" in line:
                line = line.split("%")[0]
            line = re.sub(r"\\hline\s*", "", line)

            if "\n" in line:
                for subline in line.split("\n"):
                    if "&" in subline:
                        line = subline
                        break

            stripped = line.strip()
            if not stripped:
                continue
            if "Host" in stripped and "Reference" in stripped:
                continue
            if "Mpc" in stripped and "mag" in stripped:
                continue
            if "maximum" in stripped and "average" in stripped:
                continue

            parts = [p.strip() for p in stripped.split("&")]
            if len(parts) < 8:
                continue

            host = parts[0].strip()
            if not host or host.startswith("\\"):
                continue

            d_mpc = parts[1].strip()
            mu_r_limit = parts[2].strip()
            dsi_max = parts[3].strip()
            dsi_avg = parts[4].strip()
            mu_g = parts[5].replace("$", "").replace("\\pm", "+/-").strip()
            mu_r = parts[6].replace("$", "").replace("\\pm", "+/-").strip()
            g_r = parts[7].replace("$", "").replace("\\pm", "+/-").strip()
            ref = parts[8].strip() if len(parts) > 8 else ""
            ref = ref.replace("$\\ast$", "*").replace("\\ast", "*")
            ref = re.sub(r"\s+", "", ref)

            rows.append({
                "Host": host,
                "D_Mpc": d_mpc,
                "mu_r_limit": mu_r_limit,
                "DSI_max": dsi_max,
                "DSI_avg": dsi_avg,
                "mu_g_stream": mu_g,
                "mu_r_stream": mu_r,
                "g_r_0_stream": g_r,
                "Reference": ref,
            })

        return pd.DataFrame(rows)
