"""Base class for stream extractors."""

from pathlib import Path

import pandas as pd


class StreamExtractor:
    arxiv_id: str
    needs_resolution: bool  # Whether SIMBAD resolution is needed
    name_column: str | None = None  # Column to resolve (e.g. "Name", "Host", "Galaxy")

    def extract(self, source_dir: Path) -> pd.DataFrame:
        """Parse source files -> DataFrame. Subclasses implement this."""
        raise NotImplementedError

    def source_dir(self, data_root: Path) -> Path:
        return data_root / "streams" / "sources" / self.arxiv_id

    def output_dir(self, data_root: Path) -> Path:
        return data_root / "streams" / self.arxiv_id
