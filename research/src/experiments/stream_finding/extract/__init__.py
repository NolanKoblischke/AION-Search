"""Registry of all stream extractors."""

from .base import StreamExtractor
from .extract_1301_4275 import Extract1301_4275
from .extract_1803_05447 import Extract1803_05447
from .extract_1807_07195 import Extract1807_07195
from .extract_2104_06071 import Extract2104_06071
from .extract_2209_08636 import Extract2209_08636
from .extract_2406_13496 import Extract2406_13496
from .extract_2407_20483 import Extract2407_20483
from .extract_2502_14531 import Extract2502_14531
from .extract_2503_18480 import Extract2503_18480
from .extract_2508_02154 import Extract2508_02154
from .extract_2509_18274 import Extract2509_18274

EXTRACTORS: list[StreamExtractor] = [
    Extract1301_4275(),
    Extract1803_05447(),
    Extract1807_07195(),
    Extract2104_06071(),
    Extract2209_08636(),
    Extract2406_13496(),
    Extract2407_20483(),
    Extract2502_14531(),
    Extract2503_18480(),
    Extract2508_02154(),
    Extract2509_18274(),
]

_EXTRACTOR_MAP = {e.arxiv_id: e for e in EXTRACTORS}


def get_extractor(arxiv_id: str) -> StreamExtractor:
    if arxiv_id not in _EXTRACTOR_MAP:
        raise KeyError(f"No extractor for '{arxiv_id}'. Available: {list(_EXTRACTOR_MAP)}")
    return _EXTRACTOR_MAP[arxiv_id]


def get_all_extractors() -> list[StreamExtractor]:
    return EXTRACTORS
