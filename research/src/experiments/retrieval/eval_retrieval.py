"""
Unified k-fold evaluation script for AION and AION-search on GZ5 and lens datasets.

Tasks:
  aion_gz5         - AION raw embeddings on GZ5 (image-to-image, spiral/merger)
  aion_lens        - AION raw embeddings on lenses (image-to-image)
  aion_search_gz5  - AION-search CLIP embeddings on GZ5 (text query, spiral/merger)
  aion_search_lens - AION-search CLIP embeddings on lenses (text query)
"""

import argparse
import json
import sys
from pathlib import Path
import logging

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from astropy.coordinates import SkyCoord
from astropy import units as u
from huggingface_hub import hf_hub_download, HfApi

from src.evals.eval_utils import generate_query_embedding, setup_logging
from src.clip.models.clip_model import AIONSearchClipModel


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def dcg(relevances: np.ndarray) -> float:
    return np.sum((2 ** relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))


def ndcg_at_k(relevances_ranked: np.ndarray, k: int, all_relevances: np.ndarray) -> float:
    actual_dcg = dcg(relevances_ranked[:k])
    ideal_dcg = dcg(np.sort(all_relevances)[::-1][:k])
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def load_clip_model(model_path: str, device: str) -> AIONSearchClipModel:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['model_config']
    model = AIONSearchClipModel(
        image_input_dim=cfg['image_input_dim'],
        text_input_dim=cfg['text_input_dim'],
        embedding_dim=cfg['embedding_dim'],
        use_mean_embeddings=cfg['use_mean_embeddings'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


HF_MODEL_REPO = "astronolan/aion-search"
HF_MODEL_REVISION = "e6d56ee28b6768f4e3e4494b2c0b32a00abb2594"
HF_DATASET_REVISIONS = {
    "astronolan/gz-decals-embeddings": "c11f7a02aa1ed00b85f3dd43c222271046445a2e",
    "astronolan/lens-retrieval-ls-embeddings": "f5507c433552084e2b3d195a27dae5110037d64d",
}


def load_clip_model_from_hf(device: str) -> AIONSearchClipModel:
    """Download and load CLIP model from HuggingFace (safetensors + config.json)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json

    config_path = hf_hub_download(HF_MODEL_REPO, "config.json", revision=HF_MODEL_REVISION)
    weights_path = hf_hub_download(HF_MODEL_REPO, "model.safetensors", revision=HF_MODEL_REVISION)

    with open(config_path) as f:
        cfg = json.load(f)

    model = AIONSearchClipModel(
        image_input_dim=cfg['image_input_dim'],
        text_input_dim=cfg['text_input_dim'],
        embedding_dim=cfg['embedding_dim'],
        image_hidden_dim=cfg.get('image_hidden_dim', 768),
        text_hidden_dim=cfg.get('text_hidden_dim', 1024),
        dropout=cfg.get('dropout', 0.1),
        use_mean_embeddings=cfg.get('use_mean_embeddings', True),
    ).to(device)

    state_dict = load_file(weights_path, device=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_kfold_assignments(kfolds_path: str, n_rows: int) -> np.ndarray:
    df = pd.read_csv(kfolds_path)
    fold_labels = np.full(n_rows, -1, dtype=int)
    row_indices = df['row_index'].to_numpy()
    fold_labels[row_indices] = df['kfold'].to_numpy()
    if np.any(fold_labels == -1):
        raise ValueError(f"K-fold assignments missing for {int(np.sum(fold_labels == -1))} rows")
    return fold_labels


def load_hf_parquet(repo_id: str, columns: list[str], logger, revision: str | None = None) -> dict[str, np.ndarray]:
    """Download all parquet shards from a HuggingFace dataset and return requested columns."""
    api = HfApi()
    all_files = api.list_repo_files(repo_id, repo_type='dataset', revision=revision)
    shard_files = sorted(f for f in all_files if f.endswith('.parquet'))
    logger.info(f"Downloading {len(shard_files)} parquet shards from https://huggingface.co/datasets/{repo_id}")
    if revision:
        logger.info(f"Using Hugging Face revision: {revision}")
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub' / f'datasets--{repo_id.replace("/", "--")}'
    logger.info(f"HF cache directory: {cache_dir}")

    tables = []
    for sf in shard_files:
        path = hf_hub_download(repo_id, sf, repo_type='dataset', revision=revision)
        tables.append(pq.read_table(path, columns=columns))

    combined = pa.concat_tables(tables)
    data = {}
    for col in columns:
        arr = combined.column(col)
        if pa.types.is_list(arr.type):
            data[col] = np.array([row.as_py() for row in arr], dtype=np.float32)
        elif pa.types.is_floating(arr.type):
            data[col] = arr.to_numpy().astype(np.float64 if arr.type == pa.float64() else np.float32)
        elif pa.types.is_integer(arr.type):
            data[col] = arr.to_numpy()
        elif pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            data[col] = arr.to_pandas().values
        elif pa.types.is_boolean(arr.type):
            data[col] = arr.to_numpy()
        else:
            data[col] = arr.to_pandas().values
    return data


def load_lens_data(
    parent_ra: np.ndarray,
    parent_dec: np.ndarray,
    hsc_path: str,
    masterlens_path: str,
    match_radius_arcsec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load lens catalogs and return (relevance, query_indices)."""
    hsc = pd.read_csv(hsc_path)
    masterlens = pd.read_csv(masterlens_path)
    parent_coords = SkyCoord(ra=parent_ra * u.degree, dec=parent_dec * u.degree)

    # Relevance: match parent to combined catalog
    combined = pd.concat([hsc[['ra', 'dec']], masterlens[['ra', 'dec']]], ignore_index=True)
    lens_coords = SkyCoord(ra=combined['ra'].values * u.degree, dec=combined['dec'].values * u.degree)
    _, sep, _ = parent_coords.match_to_catalog_sky(lens_coords)
    relevance = (sep.arcsec <= match_radius_arcsec).astype(np.float32)

    # Query indices: de-duplicate masterlens against HSC, then match to parent
    hsc_coords = SkyCoord(ra=hsc['ra'].values * u.degree, dec=hsc['dec'].values * u.degree)
    master_coords = SkyCoord(ra=masterlens['ra'].values * u.degree, dec=masterlens['dec'].values * u.degree)
    _, sep_dedup, _ = master_coords.match_to_catalog_sky(hsc_coords)
    combined_q = pd.concat([masterlens[sep_dedup.arcsec > 1.0], hsc], ignore_index=True)
    q_coords = SkyCoord(ra=combined_q['ra'].values * u.degree, dec=combined_q['dec'].values * u.degree)
    idx, sep_match, _ = q_coords.match_to_catalog_sky(parent_coords)
    query_indices = np.unique(idx[sep_match.arcsec <= match_radius_arcsec])

    return relevance, query_indices


def project_through_clip(embeddings: np.ndarray, model: AIONSearchClipModel, device: str) -> np.ndarray:
    t = torch.tensor(embeddings, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model.image_projector(t)
    return out.cpu().numpy()


def get_text_query_features(text: str, model: AIONSearchClipModel, device: str) -> np.ndarray:
    emb = generate_query_embedding(text)
    t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.text_projector(t)
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# NDCG evaluation
# ---------------------------------------------------------------------------

def eval_text_query_ndcg(similarities, relevances, fold_labels, k):
    """Per-fold NDCG for text-query (search) tasks."""
    result = []
    for fold in sorted(np.unique(fold_labels)):
        m = fold_labels == fold
        order = np.argsort(similarities[m])[::-1]
        result.append(ndcg_at_k(relevances[m][order], k, relevances[m]))
    return result


def eval_image_query_ndcg(embeddings, query_indices, relevances, fold_labels, k,
                          ra=None, dec=None, save_top_n=100):
    """Per-fold NDCG for image-query (raw embedding) tasks.

    Returns (fold_ndcgs, query_details, result_rows):
      - fold_ndcgs: list of per-fold mean NDCG
      - query_details: list of dicts with per-query info (for manifest CSV)
      - result_rows: list of tuples with per-query top-N neighbors (for objects CSV)

    If ra/dec are None, query_details and result_rows are empty lists.
    """
    fold_ndcgs = []
    query_details = []
    result_rows = []
    query_counter = 0

    for fold in sorted(np.unique(fold_labels)):
        fold_mask = fold_labels == fold
        fold_idx = np.where(fold_mask)[0]
        fold_emb = embeddings[fold_idx]
        fold_rel = relevances[fold_idx]
        fold_qi = query_indices[fold_mask[query_indices]]

        if len(fold_qi) == 0:
            fold_ndcgs.append(0.0)
            continue

        idx_to_pos = {idx: pos for pos, idx in enumerate(fold_idx)}
        max_k = min(k, len(fold_idx) - 1)
        save_n = min(save_top_n, len(fold_idx) - 1)
        scores = []
        if max_k > 0:
            sims_all = fold_emb @ embeddings[fold_qi].T
            for i, qi in enumerate(fold_qi):
                pos = idx_to_pos[qi]
                sims = sims_all[:, i].copy()
                sims[pos] = -np.inf

                # Get top-N for saving; use top max_k for NDCG
                top_n = np.argpartition(sims, -save_n)[-save_n:]
                top_n = top_n[np.argsort(sims[top_n])[::-1]]
                top = top_n[:max_k]

                all_rels = fold_rel.copy()
                all_rels[pos] = 0.0
                q_ndcg = ndcg_at_k(fold_rel[top], max_k, all_rels)
                scores.append(q_ndcg)

                if ra is not None:
                    query_details.append({
                        'query_index': query_counter,
                        'fold_index': int(fold),
                        'global_index': int(qi),
                        'ra': float(ra[qi]),
                        'dec': float(dec[qi]),
                        'relevance': float(relevances[qi]),
                        'ndcg@10': float(q_ndcg),
                    })
                    for rank_i, ni in enumerate(top_n):
                        gi = fold_idx[ni]
                        result_rows.append((
                            query_counter, int(fold), rank_i + 1, int(gi),
                            float(ra[gi]), float(dec[gi]),
                            float(sims[ni]), float(relevances[gi]),
                        ))
                    query_counter += 1

        fold_ndcgs.append(float(np.mean(scores)) if scores else 0.0)
    return fold_ndcgs, query_details, result_rows


# ---------------------------------------------------------------------------
# Summarize and save
# ---------------------------------------------------------------------------

def summarize_and_save(metric_ndcgs, fold_labels, k, output_dir, filename, logger,
                       extra_summary=None):
    unique_folds = sorted(np.unique(fold_labels))

    fold_results = []
    for fi, fold in enumerate(unique_folds):
        d = {'fold': int(fold)}
        for prefix, vals in metric_ndcgs.items():
            key = f'{prefix}_ndcg@{k}' if prefix else f'ndcg@{k}'
            d[key] = vals[fi]
        d['n_samples'] = int(np.sum(fold_labels == fold))
        fold_results.append(d)

    summary = dict(extra_summary) if extra_summary else {}
    for prefix, vals in metric_ndcgs.items():
        mk = f'{prefix}_mean_ndcg@{k}' if prefix else f'mean_ndcg@{k}'
        sk = f'{prefix}_std_ndcg@{k}' if prefix else f'std_ndcg@{k}'
        summary[mk] = float(np.mean(vals))
        summary[sk] = float(np.std(vals))
    summary['n_folds'] = len(unique_folds)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / filename, 'w') as f:
        json.dump({'summary': summary, 'folds': fold_results}, f, indent=2)
    logger.info(f"Results saved to: {output_dir / filename}")

    for prefix in metric_ndcgs:
        mk = f'{prefix}_mean_ndcg@{k}' if prefix else f'mean_ndcg@{k}'
        sk = f'{prefix}_std_ndcg@{k}' if prefix else f'std_ndcg@{k}'
        label = f"{prefix.capitalize()} " if prefix else "Mean "
        logger.info(f"{label}NDCG@{k}: {summary[mk]:.4f} +/- {summary[sk]:.4f}")


# ---------------------------------------------------------------------------
# Dataset loaders -- each returns (embeddings, fold_labels, metrics, extra_summary)
#
# metrics: dict mapping prefix -> (relevances, query_indices)
#   - prefix is '' for single-metric tasks, 'spiral'/'merger' for GZ5
#   - query_indices used by raw tasks, ignored by search tasks
# ---------------------------------------------------------------------------

def load_gz5_dataset(args, logger):
    if args.use_hf:
        return _load_gz5_from_hf(args, logger)

    kfolds_df = pd.read_csv(args.kfolds_path)
    row_indices = kfolds_df['row_index'].to_numpy()
    if args.kfold:
        fold_labels = kfolds_df['kfold'].to_numpy()
    else:
        fold_labels = np.zeros(len(row_indices), dtype=int)

    logger.info(f"Loading GZ5 data for {len(row_indices):,} rows")
    with h5py.File(args.hdf5_path, 'r') as f:
        data_table = f['__astropy_table__']
        if args.embedding_key not in data_table.dtype.names:
            raise KeyError(f"Embedding key '{args.embedding_key}' not found in HDF5")
        data = data_table[row_indices]
        embeddings = np.array(data[args.embedding_key], dtype=np.float32)
        spiral = np.array(data['has-spiral-arms_yes_fraction'], dtype=np.float32)
        merger = np.array(data['merging_merger_fraction'], dtype=np.float32)
        votes = np.array(data['smooth-or-featured_total-votes'], dtype=np.float32)

    spiral_qi = np.where((spiral > 0.9) & (votes >= 3))[0]
    merger_qi = np.where((merger > 0.9) & (votes >= 3))[0]
    ra = np.array(data['ra'], dtype=np.float64)
    dec = np.array(data['dec'], dtype=np.float64)
    logger.info(f"Spiral queries: {len(spiral_qi)}, Merger queries: {len(merger_qi)}")

    metrics = {
        'spiral': (spiral, spiral_qi),
        'merger': (merger, merger_qi),
    }
    extra = {
        'total_samples': int(len(row_indices)),
        'spiral_queries': int(len(spiral_qi)),
        'merger_queries': int(len(merger_qi)),
    }
    return embeddings, fold_labels, metrics, extra, ra, dec


def _load_gz5_from_hf(args, logger):
    is_search = 'search' in args.task
    emb_col = 'aion_search_embedding' if is_search else 'aion_mean_embedding'
    columns = [
        emb_col, 'ra', 'dec',
        'has-spiral-arms_yes_fraction', 'merging_merger_fraction',
        'smooth-or-featured_total-votes',
    ]
    if args.kfold:
        columns.append('kfold')
    logger.info(f"Loading GZ5 from HuggingFace ({args.hf_repo_id}), embedding: {emb_col}")
    data = load_hf_parquet(args.hf_repo_id, columns, logger, revision=args.hf_revision)

    embeddings = data[emb_col]
    fold_labels = data['kfold'] if args.kfold else np.zeros(len(embeddings), dtype=int)
    ra = data['ra'].astype(np.float64)
    dec = data['dec'].astype(np.float64)
    spiral = data['has-spiral-arms_yes_fraction'].astype(np.float32)
    merger = data['merging_merger_fraction'].astype(np.float32)
    votes = data['smooth-or-featured_total-votes'].astype(np.float32)

    spiral_qi = np.where((spiral > 0.9) & (votes >= 3))[0]
    merger_qi = np.where((merger > 0.9) & (votes >= 3))[0]
    logger.info(f"Loaded {len(embeddings):,} rows. Spiral queries: {len(spiral_qi)}, Merger queries: {len(merger_qi)}")

    metrics = {
        'spiral': (spiral, spiral_qi),
        'merger': (merger, merger_qi),
    }
    extra = {'total_samples': int(len(embeddings))}
    return embeddings, fold_labels, metrics, extra, ra, dec


def load_lens_dataset(args, logger):
    if args.use_hf:
        return _load_lens_from_hf(args, logger)

    with h5py.File(args.hdf5_path, 'r') as f:
        data_table = f['__astropy_table__']
        n_rows = len(data_table)
        if args.embedding_key not in data_table.dtype.names:
            raise KeyError(f"Embedding key '{args.embedding_key}' not found in HDF5")
        logger.info(f"Loading embeddings ({args.embedding_key}) for {n_rows:,} rows")
        data = data_table[:]
        embeddings = np.array(data[args.embedding_key], dtype=np.float32)
        ra = np.array(data['ra'], dtype=np.float64)
        dec = np.array(data['dec'], dtype=np.float64)

    relevance, query_indices = load_lens_data(
        ra, dec, args.hsc_path, args.masterlens_path, args.match_arcsec,
    )
    logger.info(f"Relevant objects: {int(np.sum(relevance))}, Query lenses: {len(query_indices)}")

    if args.kfold:
        fold_labels = load_kfold_assignments(args.kfolds_path, n_rows)
    else:
        fold_labels = np.zeros(n_rows, dtype=int)
    metrics = {'': (relevance, query_indices)}
    extra = {
        'total_samples': int(n_rows),
        'total_relevant': int(np.sum(relevance)),
        'total_queries': int(len(query_indices)),
    }
    return embeddings, fold_labels, metrics, extra, ra, dec


def _load_lens_from_hf(args, logger):
    is_search = 'search' in args.task
    emb_col = 'aion_search_embedding' if is_search else 'aion_mean_embedding'
    columns = [emb_col, 'ra', 'dec', 'is_lens']
    if args.kfold:
        columns.append('kfold')
    logger.info(f"Loading lens data from HuggingFace ({args.hf_repo_id}), embedding: {emb_col}")
    data = load_hf_parquet(args.hf_repo_id, columns, logger, revision=args.hf_revision)

    embeddings = data[emb_col]
    fold_labels = data['kfold'] if args.kfold else np.zeros(len(embeddings), dtype=int)
    ra = data['ra'].astype(np.float64)
    dec = data['dec'].astype(np.float64)
    relevance = data['is_lens'].astype(np.float32)
    query_indices = np.where(relevance > 0)[0]

    logger.info(f"Loaded {len(embeddings):,} rows. Relevant: {int(np.sum(relevance))}, Queries: {len(query_indices)}")

    metrics = {'': (relevance, query_indices)}
    extra = {
        'total_samples': int(len(embeddings)),
        'total_relevant': int(np.sum(relevance)),
        'total_queries': int(len(query_indices)),
    }
    return embeddings, fold_labels, metrics, extra, ra, dec


# ---------------------------------------------------------------------------
# Search query loading
# ---------------------------------------------------------------------------

QUERY_DEFAULTS = {
    'spiral': 'visible spiral arms',
    'merger': 'merging',
    '': 'gravitational lens',
}


def load_search_queries(args, metric_names, logger):
    """Return list of {metric_name: query_text} dicts."""
    defaults = {n: QUERY_DEFAULTS[n] for n in metric_names}
    if not args.queries:
        return [defaults]

    with open(args.queries, 'r') as f:
        data = json.load(f)
    per_metric = {n: data.get(n or 'lens', [defaults[n]]) for n in metric_names}
    n_sets = max(len(v) for v in per_metric.values())
    logger.info(f"Loaded {n_sets} query sets from {args.queries}")
    return [{n: per_metric[n][i % len(per_metric[n])] for n in metric_names} for i in range(n_sets)]


# ---------------------------------------------------------------------------
# Per-object CSV output
# ---------------------------------------------------------------------------

def save_objects_csv(similarities_by_name, metrics, fold_labels, ra, dec,
                     output_dir, task_prefix, logger):
    """Save one CSV per metric: rank, fold_index, global_index, ra, dec, similarity, relevance."""
    output_dir = Path(output_dir)
    for name, sims in similarities_by_name.items():
        relevances = metrics[name][0]
        rows = []
        for fold in sorted(np.unique(fold_labels)):
            fold_idx = np.where(fold_labels == fold)[0]
            fold_sims = sims[fold_idx]
            order = np.argsort(fold_sims)[::-1]
            for rank_i, oi in enumerate(order):
                gi = fold_idx[oi]
                rows.append((rank_i + 1, int(fold), int(gi), ra[gi], dec[gi],
                             float(fold_sims[oi]), float(relevances[gi])))

        suffix = f"_{name}" if name else ""
        csv_path = output_dir / f"{task_prefix}{suffix}_objects.csv"
        pd.DataFrame(rows, columns=[
            'rank', 'fold_index', 'global_index', 'ra', 'dec', 'similarity', 'relevance',
        ]).to_csv(csv_path, index=False)
        logger.info(f"Object CSV saved to: {csv_path}")


def save_query_manifest(query_details, output_dir, task_prefix, metric_name, logger):
    """Save per-query manifest CSV with NDCG scores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{metric_name}" if metric_name else ""
    csv_path = output_dir / f"{task_prefix}{suffix}_queries.csv"
    pd.DataFrame(query_details).to_csv(csv_path, index=False)
    logger.info(f"Query manifest saved to: {csv_path}")


def save_query_objects_csv(result_rows, output_dir, task_prefix, metric_name, logger):
    """Save per-query top-N objects CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{metric_name}" if metric_name else ""
    csv_path = output_dir / f"{task_prefix}{suffix}_objects.csv"
    pd.DataFrame(result_rows, columns=[
        'query_index', 'fold_index', 'rank', 'global_index',
        'ra', 'dec', 'similarity', 'relevance',
    ]).to_csv(csv_path, index=False)
    logger.info(f"Query objects CSV saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Unified evaluation
# ---------------------------------------------------------------------------

def eval_task(args, logger):
    is_lens = 'lens' in args.task
    is_search = 'search' in args.task
    task_label = 'lens' if is_lens else 'gz5'

    # Load dataset
    if is_lens:
        embeddings, fold_labels, metrics, extra_summary, ra, dec = load_lens_dataset(args, logger)
    else:
        embeddings, fold_labels, metrics, extra_summary, ra, dec = load_gz5_dataset(args, logger)

    if is_search:
        if args.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            args.device = 'cpu'

        if args.use_hf and not args._model_path_explicit:
            model = load_clip_model_from_hf(args.device)
            logger.info(f"Loaded CLIP model from HuggingFace ({HF_MODEL_REPO})")
        else:
            model = load_clip_model(args.model_path, args.device)
            logger.info(f"Loaded CLIP model from {args.model_path}")

        if args.use_hf:
            logger.info("Using pre-projected CLIP embeddings from HuggingFace")
            clip_features = embeddings
        else:
            logger.info("Projecting embeddings through CLIP model")
            clip_features = project_through_clip(embeddings, model, args.device)

        query_sets = load_search_queries(args, list(metrics.keys()), logger)

        for qi, query_texts in enumerate(query_sets):
            logger.info(f"Query set {qi}: " +
                        ", ".join(f"{n or 'lens'}='{t}'" for n, t in query_texts.items()))

            ndcgs = {}
            sims_by_name = {}
            for name, (relevances, _) in metrics.items():
                qf = get_text_query_features(query_texts[name], model, args.device)
                sims = (clip_features @ qf.T).reshape(-1)
                sims_by_name[name] = sims
                ndcgs[name] = eval_text_query_ndcg(sims, relevances, fold_labels, args.k)

            query_extra = {(f'{n}_query' if n else 'query'): t for n, t in query_texts.items()}
            if not args.queries:
                subdir = "default"
            else:
                subdir = f"query_{qi}"
            query_output_dir = str(Path(args.output_dir) / subdir)

            prefix = f"aion_search_{task_label}_kfold"
            fname = f"{prefix}_results.json"

            summarize_and_save(ndcgs, fold_labels, args.k, query_output_dir, fname, logger,
                               extra_summary={**query_extra, **extra_summary})
            save_objects_csv(sims_by_name, metrics, fold_labels, ra, dec,
                             query_output_dir, prefix, logger)
    else:
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms

        ndcgs = {}
        prefix = f"aion_{task_label}_kfold"
        for name, (relevances, query_indices) in metrics.items():
            ndcgs[name], query_details, result_rows = eval_image_query_ndcg(
                embeddings, query_indices, relevances, fold_labels, args.k,
                ra=ra, dec=dec, save_top_n=100,
            )
            save_query_manifest(query_details, args.output_dir, prefix, name, logger)
            save_query_objects_csv(result_rows, args.output_dir, prefix, name, logger)

        summarize_and_save(ndcgs, fold_labels, args.k, args.output_dir,
                           f"{prefix}_results.json", logger,
                           extra_summary=extra_summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_DEFAULTS = {
    'aion_gz5': {
        'hdf5_path': 'data/gz5_base_embedded.hdf5',
        'kfolds_path': 'data/retrieval_results/kfolds/gz5.csv',
        'embedding_key': 'embeddings',
        'output_dir': 'data/retrieval_results/aion_gz5',
        'hf_repo_id': 'astronolan/gz-decals-embeddings',
        'hf_revision': HF_DATASET_REVISIONS['astronolan/gz-decals-embeddings'],
    },
    'aion_lens': {
        'hdf5_path': 'data/lens/lens_parent_sample_v1_embedded_oct24_base.hdf5',
        'kfolds_path': 'data/retrieval_results/kfolds/lens.csv',
        'embedding_key': 'embeddings_ls',
        'output_dir': 'data/retrieval_results/aion_lens',
        'hf_repo_id': 'astronolan/lens-retrieval-ls-embeddings',
        'hf_revision': HF_DATASET_REVISIONS['astronolan/lens-retrieval-ls-embeddings'],
    },
    'aion_search_gz5': {
        'hdf5_path': 'data/gz5_base_embedded.hdf5',
        'kfolds_path': 'data/retrieval_results/kfolds/gz5.csv',
        'embedding_key': 'embeddings',
        'output_dir': 'data/retrieval_results/aion_search_gz5',
        'hf_repo_id': 'astronolan/gz-decals-embeddings',
        'hf_revision': HF_DATASET_REVISIONS['astronolan/gz-decals-embeddings'],
    },
    'aion_search_lens': {
        'hdf5_path': 'data/lens/lens_parent_sample_v1_embedded_oct24_base.hdf5',
        'kfolds_path': 'data/retrieval_results/kfolds/lens.csv',
        'embedding_key': 'embeddings_ls',
        'output_dir': 'data/retrieval_results/aion_search_lens',
        'hf_repo_id': 'astronolan/lens-retrieval-ls-embeddings',
        'hf_revision': HF_DATASET_REVISIONS['astronolan/lens-retrieval-ls-embeddings'],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified k-fold evaluation for AION / AION-search")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_DEFAULTS.keys()))
    parser.add_argument("--use-hf", action="store_true",
                        help="Load dataset from HuggingFace instead of local HDF5")
    parser.add_argument("--kfold", action="store_true",
                        help="Use k-fold cross-validation (default: evaluate on full dataset)")
    parser.add_argument("--hdf5-path", type=str, default=None)
    parser.add_argument("--kfolds-path", type=str, default=None)
    parser.add_argument("--embedding-key", type=str, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--hsc-path", type=str, default="data/lens/hsc_lenses.csv")
    parser.add_argument("--masterlens-path", type=str, default="data/lens/masterlens.csv")
    parser.add_argument("--match-arcsec", type=float, default=1.0)
    parser.add_argument("--model-path", type=str, default="data/aionsearchmodel.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--hf-revision", type=str, default=None,
                        help="Hugging Face dataset revision. Defaults to the pinned paper revision for the selected task.")

    args = parser.parse_args()
    # Track whether --model-path was explicitly provided by the user
    args._model_path_explicit = '--model-path' in sys.argv
    for key, value in TASK_DEFAULTS[args.task].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    eval_task(args, logger)


if __name__ == "__main__":
    main()
