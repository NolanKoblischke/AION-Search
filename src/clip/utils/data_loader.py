"""
Data loader for multi-text training using unified parquet file with nested text embeddings.
This loader handles the new unified format from 05_generate_unified_embeddings.py.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import random
from astropy.coordinates import SkyCoord
from astropy import units as u

logger = logging.getLogger(__name__)


class UnifiedMultiTextDataset(Dataset):
    """Dataset for unified parquet file with multiple text embeddings per galaxy."""
    
    def __init__(self, parquet_path, split="train", train_ratio=0.8, 
                 text_sampling_strategy="random", epoch=0, max_train_samples=None,
                 num_embedding=None, exclude_crossmatch_path=None):
        self.parquet_path = Path(parquet_path)
        self.split = split
        self.train_ratio = train_ratio
        self.text_sampling_strategy = text_sampling_strategy
        self.epoch = epoch
        self.max_train_samples = max_train_samples
        self.num_embedding = num_embedding
        self.exclude_crossmatch_path = exclude_crossmatch_path
        
        # Load the parquet file
        logger.info(f"Loading unified embeddings from {self.parquet_path}")
        self.df = pd.read_parquet(self.parquet_path)
        initial_count = len(self.df)
        
        # Apply coordinate-based exclusion if crossmatch file provided
        if exclude_crossmatch_path is not None:
            self._apply_coordinate_exclusion()
            logger.info(f"Excluded {initial_count - len(self.df):,} galaxies from crossmatch file")
            logger.info(f"Remaining galaxies after exclusion: {len(self.df):,}")
        
        # Create train/val split based on galaxy_index
        n_samples = len(self.df)
        indices = np.arange(n_samples)
        self.seed = 42
        
        # Deterministic split based on galaxy_index
        split_mask = []
        for idx in range(n_samples):
            galaxy_idx = self.df.iloc[idx]['galaxy_index']
            # Hash the galaxy index for deterministic assignment
            sample_hash = hash((galaxy_idx, self.seed)) % 10000 / 10000.0
            is_train = sample_hash < self.train_ratio
            split_mask.append(is_train)
        
        split_mask = np.array(split_mask)
        
        if split == "train":
            self.indices = indices[split_mask]
            # Limit training samples if specified
            if self.max_train_samples is not None and len(self.indices) > self.max_train_samples:
                rng = np.random.RandomState(self.seed)
                selected_indices = rng.choice(self.indices, size=self.max_train_samples, replace=False)
                self.indices = np.sort(selected_indices)  # Sort for reproducibility
                logger.info(f"Limited training set to {self.max_train_samples} samples")
        else:
            self.indices = indices[~split_mask]
        
        # Log final dataset sizes
        total_excluded = initial_count - len(self.df) if exclude_crossmatch_path else 0
        logger.info(f"Dataset initialized for {split} split:")
        logger.info(f"  Total galaxies loaded: {initial_count:,}")
        if exclude_crossmatch_path:
            logger.info(f"  Galaxies excluded (benchmark overlap): {total_excluded:,}")
            logger.info(f"  Galaxies after exclusion: {len(self.df):,}")
        logger.info(f"  {split.capitalize()} split size: {len(self.indices):,} samples")
        logger.info(f"  Text sampling strategy: {text_sampling_strategy}")
        
        # Validate num_embedding parameter for specific_summary strategy
        if text_sampling_strategy == "specific_summary" and num_embedding is None:
            raise ValueError("num_embedding parameter is required when using 'specific_summary' strategy")
        
        # Check data structure
        sample_row = self.df.iloc[0]
        n_augmented = len(sample_row['augmented_embeddings'])
        logger.info(f"Each galaxy has 1 original + {n_augmented} augmented embeddings = {1 + n_augmented} total")
        
        # Validate num_embedding is within valid range
        if text_sampling_strategy == "specific_summary":
            total_embeddings = 1 + n_augmented
            if num_embedding < 0 or num_embedding >= total_embeddings:
                raise ValueError(f"num_embedding must be between 0 and {total_embeddings-1}, got {num_embedding}")
            logger.info(f"Using specific embedding at index {num_embedding}")
    
    def _apply_coordinate_exclusion(self):
        """Apply coordinate-based exclusion using crossmatch results."""
        logger.info(f"Loading crossmatch results from {self.exclude_crossmatch_path}")
        
        # Load crossmatch results
        crossmatch_df = pd.read_csv(self.exclude_crossmatch_path)
        logger.info(f"Loaded {len(crossmatch_df):,} galaxies to exclude")
        
        # Check if we have ra/dec columns in our data
        if 'ra' not in self.df.columns or 'dec' not in self.df.columns:
            # Try to load from the descriptions file to get coordinates
            descriptions_path = self.parquet_path.parent / "galaxy_descriptions_merged.hdf5"
            if descriptions_path.exists():
                logger.info("Loading coordinates from descriptions file")
                import h5py
                with h5py.File(descriptions_path, 'r') as f:
                    if '__astropy_table__' in f:
                        table = f['__astropy_table__']
                        object_ids = table['object_id'][:]
                        ras = table['ra'][:]
                        decs = table['dec'][:]
                    else:
                        object_ids = f['object_id'][:]
                        ras = f['ra'][:]
                        decs = f['dec'][:]
                
                # Convert bytes to strings if needed
                if isinstance(object_ids[0], bytes):
                    object_ids = [oid.decode('utf-8') if isinstance(oid, bytes) else str(oid) for oid in object_ids]
                else:
                    object_ids = [str(oid) for oid in object_ids]
                
                # Create lookup dict
                coord_lookup = {oid: (ra, dec) for oid, ra, dec in zip(object_ids, ras, decs)}
                
                # Add ra/dec to our dataframe
                self.df['ra'] = self.df['object_id'].map(lambda x: coord_lookup.get(str(x), (np.nan, np.nan))[0])
                self.df['dec'] = self.df['object_id'].map(lambda x: coord_lookup.get(str(x), (np.nan, np.nan))[1])
            else:
                logger.error("No coordinates found in data and descriptions file not available")
                return
        
        # Create SkyCoord objects for our data
        data_coords = SkyCoord(ra=self.df['ra'].values*u.degree, 
                              dec=self.df['dec'].values*u.degree)
        
        # Create SkyCoord objects for crossmatch data
        crossmatch_coords = SkyCoord(ra=crossmatch_df['ra'].values*u.degree,
                                    dec=crossmatch_df['dec'].values*u.degree)
        
        # Find matches within 1 arcsec
        logger.info("Performing coordinate matching (1 arcsec tolerance)...")
        idx, d2d, d3d = data_coords.match_to_catalog_sky(crossmatch_coords)
        matches = d2d < 1.0*u.arcsec
        
        # Keep only non-matching galaxies
        self.df = self.df[~matches].reset_index(drop=True)
        
        logger.info(f"Found {matches.sum():,} matches to exclude")
    
    def __len__(self):
        return len(self.indices)
    
    def set_epoch(self, epoch):
        """Set current epoch for round-robin sampling."""
        self.epoch = epoch
    
    def _get_all_embeddings_and_sources(self, row):
        """Combine original and augmented embeddings into single lists."""
        # Start with original embedding
        all_embeddings = [np.array(row['text_embedding'], dtype=np.float32)]
        all_sources = [row['description_sources'][0]]  # 'original'
        
        # Add augmented embeddings
        for aug_emb, aug_source in zip(row['augmented_embeddings'], row['description_sources'][1:]):
            all_embeddings.append(np.array(aug_emb, dtype=np.float32))
            all_sources.append(aug_source)
        
        return all_embeddings, all_sources
    
    def _sample_text_embedding(self, text_embeddings, text_sources, galaxy_idx):
        """Sample one text embedding from multiple options."""
        n_texts = len(text_embeddings)
        
        if self.text_sampling_strategy == "original":
            # Always use original text (index 0)
            idx = 0
        elif self.text_sampling_strategy == "summaries-only":
            # Only use summaries (exclude original at index 0)
            if n_texts > 1:
                rng = random.Random(galaxy_idx + self.epoch * 1000000)
                idx = rng.randint(1, n_texts - 1)  # Start from 1 to exclude original
            else:
                # Fallback to original if no summaries available
                idx = 0
        elif self.text_sampling_strategy == "specific_summary":
            # Use the specific embedding index provided
            if self.num_embedding < n_texts:
                idx = self.num_embedding
            else:
                # Fallback to original if index out of range
                logger.warning(f"Requested embedding index {self.num_embedding} out of range for {n_texts} embeddings, using original")
                idx = 0
        elif self.text_sampling_strategy == "random":
            # Random sampling with seed based on galaxy_idx and epoch
            rng = random.Random(galaxy_idx + self.epoch * 1000000)
            idx = rng.randint(0, n_texts - 1)
        elif self.text_sampling_strategy == "round-robin":
            # Cycle through texts based on epoch
            idx = (self.epoch + galaxy_idx) % n_texts
        elif self.text_sampling_strategy == "weighted":
            # Weight towards original (50%) and summaries (50% / n_summaries each)
            rng = random.Random(galaxy_idx + self.epoch * 1000000)
            n_summaries = n_texts - 1
            if n_summaries > 0:
                summary_weight = 0.5 / n_summaries
                weights = [0.5] + [summary_weight] * n_summaries
            else:
                weights = [1.0]
            idx = rng.choices(range(n_texts), weights=weights)[0]
        else:
            idx = 0  # Default to original
        
        return text_embeddings[idx], text_sources[idx], idx
    
    def __getitem__(self, idx):
        """Get a single sample with randomly selected text embedding."""
        actual_idx = self.indices[idx]
        row = self.df.iloc[actual_idx]
        
        # Get AION embedding
        aion_embedding = np.array(row['aion_embedding'], dtype=np.float32)
        
        # Get all text embeddings and sources
        text_embeddings, text_sources = self._get_all_embeddings_and_sources(row)
        
        # Sample one text embedding
        galaxy_idx = row['galaxy_index']
        selected_text, selected_source, text_idx = self._sample_text_embedding(
            text_embeddings, text_sources, galaxy_idx
        )
        
        # Log selection details periodically (every 100th sample)
        if idx % 100 == 0:
            logger.debug(f"Galaxy {galaxy_idx}: Selected {selected_source} (index {text_idx}) from {len(text_sources)} options")
        
        return {
            'aion_embedding': torch.from_numpy(aion_embedding),
            'text_embedding': torch.from_numpy(selected_text),
            'galaxy_index': galaxy_idx,
            'text_source': selected_source,
            'text_index': text_idx,
            'object_id': row['object_id']
        }


def create_unified_multi_text_loaders(
    unified_embeddings_path,
    batch_size=64,
    train_ratio=0.8,
    pin_memory=True,
    text_sampling_strategy="random",
    num_workers=4,
    max_train_samples=None,
    num_embedding=None,
    exclude_crossmatch_path=None,
    **kwargs
):
    """
    Create train and validation data loaders for multi-text training from unified parquet.
    
    Args:
        unified_embeddings_path: Path to unified parquet file
        batch_size: Batch size for training
        train_ratio: Fraction of samples for training
        pin_memory: Whether to pin memory for GPU transfer
        text_sampling_strategy: How to sample text embeddings ("original", "summaries-only", "specific_summary", "random", "round-robin", "weighted")
        num_workers: Number of data loading workers
        max_train_samples: Maximum number of training samples (for data scaling experiments)
        num_embedding: When using "specific_summary" strategy, the index of the embedding to use
        exclude_crossmatch_path: Path to crossmatch CSV file to exclude benchmark galaxies
        **kwargs: Additional arguments
    """
    
    # Convert to Path
    parquet_path = Path(unified_embeddings_path)
    
    if not parquet_path.exists():
        raise ValueError(f"Unified embeddings file not found: {parquet_path}")
    
    logger.info(f"Creating unified multi-text data loaders from {parquet_path}")
    logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")
    logger.info(f"Text sampling strategy: {text_sampling_strategy}")
    
    # Create datasets
    train_dataset = UnifiedMultiTextDataset(
        parquet_path=parquet_path,
        split="train",
        train_ratio=train_ratio,
        text_sampling_strategy=text_sampling_strategy,
        max_train_samples=max_train_samples,
        num_embedding=num_embedding,
        exclude_crossmatch_path=exclude_crossmatch_path
    )
    
    val_dataset = UnifiedMultiTextDataset(
        parquet_path=parquet_path,
        split="val",
        train_ratio=train_ratio,
        text_sampling_strategy=text_sampling_strategy,
        num_embedding=num_embedding,
        exclude_crossmatch_path=exclude_crossmatch_path
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle within the train split
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader