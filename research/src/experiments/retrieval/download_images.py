"""
Download top-K images from eval_retrieval ranking CSVs.

Reads per-object ranking CSV files produced by eval_retrieval.py and downloads
cutout images from the DESI Legacy Survey via HiPS2FITS.

CLI:
    uv run python src/experiments/retrieval/download_images.py \
        --objects-csv <output-dir>/default/aion_search_gz5_kfold_spiral_objects.csv

    uv run python src/experiments/retrieval/download_images.py \
        --objects-csv <output-dir>/default/aion_search_lens_kfold_objects.csv \
        --top-k 500 --folds 0 1 2
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from PIL import Image

from src.plotting_scripts.default_plot import process_galaxy_image

HIPS_BASE_URLS = [
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
    "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits",
]
FOV = 0.01863  # 256 pixels * 0.262 "/pixel = 67.07" = 0.01863 deg
WIDTH = 256
HEIGHT = 256
BANDS = ["g", "r", "i", "z"]

_thread_local = threading.local()


def _get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess


def download_band(ra: float, dec: float, band: str, base_url: str,
                  timeout: int = 30) -> np.ndarray | None:
    url = (
        f"{base_url}?"
        f"hips=CDS/P/DESI-Legacy-Surveys/DR10/{band}&"
        f"ra={ra}&dec={dec}&"
        f"fov={FOV}&width={WIDTH}&height={HEIGHT}&format=fits"
    )
    try:
        resp = _get_session().get(url, timeout=timeout)
        if resp.status_code == 200:
            with fits.open(BytesIO(resp.content), memmap=False) as hdu:
                data = hdu[0].data
            if data is not None and data.shape == (HEIGHT, WIDTH):
                return data.astype(np.float32)
    except Exception:
        pass
    return None


def download_cutout(ra: float, dec: float, base_url: str,
                    retry: int = 10, delay: float = 1.0) -> np.ndarray | None:
    band_data = []
    for band in BANDS:
        data = None
        for attempt in range(retry):
            data = download_band(ra, dec, band, base_url=base_url)
            if data is not None:
                break
            time.sleep(min(delay * (2 ** attempt), 60) + random.random() * 0.5)
        if data is None:
            return None
        band_data.append(data)
    return np.stack(band_data, axis=0)


def is_valid_cutout(cutout: np.ndarray) -> bool:
    if cutout is None or np.any(np.isnan(cutout)):
        return False
    return all(not np.all(cutout[b] == 0) for b in range(cutout.shape[0]))


def download_fold_images(fold_df: pd.DataFrame, fold_dir: Path,
                         workers: int) -> dict:
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(fold_dir / "metadata.csv", index=False)

    successful = []
    failed = []
    lock = threading.Lock()

    def download_and_save(row):
        rank = int(row['rank'])
        global_idx = int(row['global_index'])
        base_url = HIPS_BASE_URLS[rank % len(HIPS_BASE_URLS)]
        cutout = download_cutout(row['ra'], row['dec'], base_url=base_url)
        if is_valid_cutout(cutout):
            try:
                img = process_galaxy_image(cutout)
                img_uint8 = (img * 255).astype(np.uint8)
                Image.fromarray(img_uint8).save(
                    fold_dir / f"rank_{rank:04d}_idx_{global_idx}.png",
                    format='PNG',
                )
                return (rank, global_idx, True)
            except Exception:
                return (rank, global_idx, False)
        return (rank, global_idx, False)

    rows = [row for _, row in fold_df.iterrows()]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_and_save, r) for r in rows]
        for future in as_completed(futures):
            rank, global_idx, ok = future.result()
            with lock:
                (successful if ok else failed).append((rank, global_idx))

    if failed:
        pd.DataFrame(failed, columns=['rank', 'global_index']).to_csv(
            fold_dir / "failed.csv", index=False)

    return {
        'fold': int(fold_df['fold_index'].iloc[0]),
        'successful': len(successful),
        'failed': len(failed),
        'total': len(rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download images from eval_retrieval ranking CSV")
    parser.add_argument("--objects-csv", type=str, required=True,
                        help="Path to *_objects.csv from eval_retrieval.py")
    parser.add_argument("--top-k", type=int, default=1000,
                        help="Number of top results per fold to download")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/kfolds/<stem>_images_<timestamp>)")
    parser.add_argument("--folds", type=int, nargs='+', default=None,
                        help="Specific folds to process (default: all)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of download workers")

    args = parser.parse_args()

    df = pd.read_csv(args.objects_csv)
    print(f"Loaded {len(df)} rows from {args.objects_csv}")

    all_folds = sorted(df['fold_index'].unique())
    folds = args.folds if args.folds is not None else all_folds
    folds = [f for f in folds if f in all_folds]
    print(f"Processing folds: {folds}")

    if args.output_dir is None:
        stem = Path(args.objects_csv).stem.replace('_objects', '')
        output_dir = Path(f"data/retrieval_results/full/rerank/{stem}/images")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    config = {
        'objects_csv': args.objects_csv,
        'top_k': args.top_k,
        'folds': [int(f) for f in folds],
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    all_results = []
    start = time.time()

    for fi, fold in enumerate(folds):
        fold_start = time.time()
        fold_df = df[df['fold_index'] == fold].nsmallest(args.top_k, 'rank')

        print(f"\n{'=' * 60}")
        print(f"Fold {fold} ({fi + 1}/{len(folds)}): downloading {len(fold_df)} images")
        print(f"{'=' * 60}")

        result = download_fold_images(
            fold_df, output_dir / f"fold_{fold}", args.workers)
        all_results.append(result)

        elapsed = time.time() - fold_start
        total_elapsed = time.time() - start
        print(f"Fold {fold}: {result['successful']}/{result['total']} successful, "
              f"{elapsed:.1f}s, total {total_elapsed / 60:.1f}min")

        if fi + 1 < len(folds):
            eta = (total_elapsed / (fi + 1)) * (len(folds) - fi - 1) / 60
            print(f"Estimated remaining: {eta:.1f}min")

    summary = {
        'total_folds': len(folds),
        'total_successful': sum(r['successful'] for r in all_results),
        'total_failed': sum(r['failed'] for r in all_results),
        'total_time_minutes': (time.time() - start) / 60,
        'folds': all_results,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE: {summary['total_successful']} downloaded, "
          f"{summary['total_failed']} failed, "
          f"{summary['total_time_minutes']:.1f}min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
