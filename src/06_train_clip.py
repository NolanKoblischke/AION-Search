"""
Train CLIP model for galaxy alignment using pre-computed AION embeddings.
"""

import argparse
from pathlib import Path
from datetime import datetime
import logging
import torch
import torch.optim as optim
import yaml
from dotenv import load_dotenv
load_dotenv()
import wandb

from src.clip.utils.logging_utils import setup_logging
from src.clip.utils.data_loader import create_unified_multi_text_loaders
from src.clip.models.clip_model import GalaxyClipModel


def train_clip_model(
    aion_embeddings_path: str,
    output_dir: str = "runs/clip_model",
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    embedding_dim: int = 1024,
    train_ratio: float = 0.8,
    device: str = None,
    scheduler_t0: int = 10,
    scheduler_t_mult: int = 2,
    scheduler_eta_min: float = 5e-7,
    gradient_clip_max_norm: float = 1.0,
    use_mean_embeddings: bool = True,
    warmup_epochs: int = 10,  # Number of epochs for linear warmup
    save_checkpoint_frequency: int = 10,
    # Multi-text parameters
    use_multi_text: bool = False,
    text_sampling_strategy: str = "random",
    num_embedding: int = None,
    # Data scaling parameters
    max_train_samples: int = None,
    # Benchmark exclusion
    exclude_crossmatch_path: str = None
):
    """Train CLIP model with pre-computed AION embeddings."""
    
    logger = logging.getLogger(__name__)
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    if max_train_samples is not None:
        logger.info(f"Limiting training set to {max_train_samples} samples")
        # Adjust batch size if it's larger than the training set
        if batch_size > max_train_samples:
            original_batch_size = batch_size
            # Use quarter of training samples, but at least 1 and at most the original batch size
            batch_size = max(1, min(original_batch_size, max_train_samples // 4))
            logger.info(f"Adjusted batch size from {original_batch_size} to {batch_size} (1/4 of {max_train_samples} training samples)")
    
    if use_multi_text:
        logger.info(f"Using multi-text data loader with strategy: {text_sampling_strategy}")
        if text_sampling_strategy == "specific_summary":
            logger.info(f"Using specific embedding at index: {num_embedding}")
        train_loader, val_loader = create_unified_multi_text_loaders(
            unified_embeddings_path=aion_embeddings_path,
            batch_size=batch_size,
            train_ratio=train_ratio,
            text_sampling_strategy=text_sampling_strategy,
            max_train_samples=max_train_samples,
            num_embedding=num_embedding,
            exclude_crossmatch_path=exclude_crossmatch_path,
        )
    else:
        # Single text mode - still use unified loader but with "original" text only
        logger.info("Using single text data loader (original descriptions only)")
        train_loader, val_loader = create_unified_multi_text_loaders(
            unified_embeddings_path=aion_embeddings_path,
            batch_size=batch_size,
            train_ratio=train_ratio,
            text_sampling_strategy="original",  # Always use original text
            max_train_samples=max_train_samples,
            exclude_crossmatch_path=exclude_crossmatch_path,
        )
    
    # Get sample batch to determine dimensions
    sample_batch = next(iter(train_loader))
    aion_embedding_shape = sample_batch['aion_embedding'].shape[1:]  # Remove batch dimension
    text_dim = sample_batch['text_embedding'].shape[-1]
    
    # Handle both mean embeddings (1D) and full embeddings (2D)
    if use_mean_embeddings:
        aion_dim = aion_embedding_shape[0]
        logger.info(f"AION embedding dimension: {aion_dim}")
    else:
        # Full embeddings have shape (n_tokens, aion_dim)
        logger.info(f"AION embedding dimension: {aion_embedding_shape[1]} x {aion_embedding_shape[0]} (channels x tokens)")
        aion_dim = aion_embedding_shape[1]  # This is the input_dim for the model
    
    logger.info(f"Text embedding dimension: {text_dim}")
    
    # Create model with appropriate configuration
    model_kwargs = {
        'image_input_dim': aion_dim,
        'text_input_dim': text_dim,
        'embedding_dim': embedding_dim,
        'use_mean_embeddings': use_mean_embeddings
    }
    
    model = GalaxyClipModel(**model_kwargs).to(device)
    
    # Setup training with AdamW optimizer (following AstroCLIP)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Setup schedulers with linear warmup
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, SequentialLR
    
    # Linear warmup scheduler
    def warmup_lambda(epoch):
        """Linear warmup for first warmup_epochs epochs"""
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Cosine annealing scheduler (will be used after warmup)
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=scheduler_t0,  # Number of epochs for the first restart
        T_mult=scheduler_t_mult,  # Factor to increase T_0 after a restart
        eta_min=scheduler_eta_min  # Minimum learning rate
    )
    
    # Combine warmup and cosine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    # Training state
    best_val_loss = float('inf')
    history = {
        'train_losses': [],
        'val_losses': []
    }
    
    # Create checkpoints directory
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    # Log text source distribution for first batch
    if use_multi_text:
        first_batch = next(iter(train_loader))
        text_sources = first_batch.get('text_source', [])
        if text_sources is not None and len(text_sources) > 0:
            from collections import Counter
            source_counts = Counter(text_sources)
            logger.info(f"First batch text source distribution: {dict(source_counts)}")
            logger.info(f"Text sampling strategy: {text_sampling_strategy}")
    
    for epoch in range(num_epochs):
        # Update epoch for multi-text loader if using it
        if use_multi_text and hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
            val_loader.dataset.set_epoch(epoch)
        
        # Training phase
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Log text source distribution every 10 epochs
        if use_multi_text and epoch % 10 == 0:
            sample_batch = next(iter(train_loader))
            text_sources = sample_batch.get('text_source', [])
            if text_sources is not None and len(text_sources) > 0:
                from collections import Counter
                source_counts = Counter(text_sources)
                logger.info(f"Epoch {epoch+1} text source distribution: {dict(source_counts)}")
        
        for batch_idx, batch in enumerate(train_loader):
            # Prepare batch for model
            model_input = {
                'image_embedding': batch['aion_embedding'].to(device),
                'text_embedding': batch['text_embedding'].to(device)
            }
            
            # Forward pass
            outputs = model(model_input)
            
            # Compute CLIP loss
            loss = model.compute_contrastive_loss(outputs)
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            total_train_loss += loss.item()
            num_train_batches += 1
            
            # Log progress every batch
            logger.info(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            # Gradient clipping for stability
            if gradient_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / num_train_batches
        history['train_losses'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                model_input = {
                    'image_embedding': batch['aion_embedding'].to(device),
                    'text_embedding': batch['text_embedding'].to(device)
                }
                
                outputs = model(model_input)
                loss = model.compute_contrastive_loss(outputs)
                
                total_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        history['val_losses'].append(avg_val_loss)

        # Log epoch results
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"  Learning rate: {current_lr:.2e}")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = output_dir / "best_clip_model.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'model_config': {
                    'image_input_dim': aion_dim,
                    'text_input_dim': text_dim,
                    'embedding_dim': embedding_dim,
                    'use_mean_embeddings': use_mean_embeddings,
                    'full_aion_shape': aion_embedding_shape if not use_mean_embeddings else None
                },
                'history': history
            }
            torch.save(checkpoint, best_model_path)
            logger.info(f"  Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_checkpoint_frequency == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'model_config': {
                    'image_input_dim': aion_dim,
                    'text_input_dim': text_dim,
                    'embedding_dim': embedding_dim,
                    'use_mean_embeddings': use_mean_embeddings,
                    'full_aion_shape': aion_embedding_shape if not use_mean_embeddings else None
                },
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"  Saved checkpoint at epoch {epoch + 1}")

        # Log to wandb
        if wandb.run:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr
            })
    
    # Save final model
    final_model_path = output_dir / "final_clip_model.pt"
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': {
            'image_input_dim': aion_dim,
            'text_input_dim': text_dim,
            'embedding_dim': embedding_dim,
            'use_mean_embeddings': use_mean_embeddings,
            'full_aion_shape': aion_embedding_shape if not use_mean_embeddings else None
        },
        'history': history
    }
    torch.save(final_checkpoint, final_model_path)
    logger.info(f"\nFinal model saved to: {final_model_path}")
    logger.info(f"Best model saved to: {output_dir / 'best_clip_model.pt'}")
    
    return {
        'model_path': final_model_path,
        'best_model_path': output_dir / 'best_clip_model.pt',
        'checkpoint_dir': output_dir,
        'checkpoints_dir': checkpoints_dir,
        'history': history,
        'best_val_loss': best_val_loss
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CLIP model for galaxy alignment")
    
    # Config file option (load first to set defaults)
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file")
    
    # Data paths
    parser.add_argument("--aion-embeddings", type=str, 
                       default="data/processed/galaxy_embeddings_unified.parquet",
                       help="Path to unified embeddings parquet file")
    
    # Training config
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                       help="Weight decay for AdamW optimizer")
    parser.add_argument("--embedding-dim", type=int, default=1024,
                       help="Shared embedding dimension")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Fraction of data for training")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cpu/cuda, auto-detect if None)")
    
    # Model config
    parser.add_argument("--use-mean-embeddings", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="Use mean-pooled AION embeddings")

    # Scheduler config
    parser.add_argument("--scheduler-t0", type=int, default=10,
                       help="Number of epochs for the first restart")
    parser.add_argument("--scheduler-t-mult", type=int, default=2,
                       help="Factor to increase T_0 after a restart")
    parser.add_argument("--scheduler-eta-min", type=float, default=5e-7,
                       help="Minimum learning rate for scheduler")
    parser.add_argument("--gradient-clip-max-norm", type=float, default=1.0,
                       help="Maximum norm for gradient clipping")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                       help="Number of epochs for linear warmup")
    parser.add_argument("--save-checkpoint-frequency", type=int, default=5,
                       help="Save checkpoint every N epochs")

    # Multi-text config
    parser.add_argument("--use-multi-text", action="store_true",
                       help="Use multi-text data loader for augmented training")
    parser.add_argument("--text-sampling-strategy", type=str, default="random",
                       choices=["original", "summaries-only", "specific_summary", "random", "round-robin"],
                       help="Strategy for sampling text embeddings")
    parser.add_argument("--num-embedding", type=int, default=None,
                       help="When using 'specific_summary' strategy, the index of the embedding to use")
    
    # Data scaling experiments
    parser.add_argument("--max-train-samples", type=int, default=None,
                       help="Maximum number of training samples (for data scaling experiments)")
    
    # Benchmark exclusion
    parser.add_argument("--exclude-crossmatch", type=str, default=None,
                       help="Path to crossmatch CSV file to exclude benchmark galaxies")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Output directory override
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (overrides auto-generated path)")
    
    # Parse args once to get config file path
    args, _ = parser.parse_known_args()
    
    # Load config file if provided to set defaults
    config_defaults = {}
    if args.config:
        with open(args.config, 'r') as f:
            config_yaml = yaml.safe_load(f)
            # Convert hyphenated keys to underscored for argparse
            for key, value in config_yaml.items():
                config_defaults[key.replace('-', '_')] = value
    
    # Set defaults from config (command line args will override these)
    parser.set_defaults(**config_defaults)
    
    # Parse args again with config defaults
    args = parser.parse_args()
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Output directory can be overridden by config or command line
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-generate output directory based on timestamp
        output_dir = Path("runs") / f"train_clip_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    
    # Configure logging to both file and console
    log_file = Path("data/logs") / f'train_clip_{timestamp}.log'
    
    # Use the existing setup_logging but with log file
    setup_logging(args.log_level, str(log_file))
    
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    
    # Log configuration
    if args.config:
        logger.info(f"Loaded config from: {args.config}")

    # Save config to output directory for reproducibility
    config_dict = vars(args)
    config_dict['output_dir'] = str(output_dir)
    config_path = output_dir / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Initialize wandb
    wandb.init(
        project="galaxy_clip",
        config=config_dict,
        name=f"train_clip_{timestamp}",
        dir=str(output_dir)
    )
    logger.info(f"Saved training config to: {config_path}")
    
    # Train model with pre-computed AION embeddings
    results = train_clip_model(
        aion_embeddings_path=args.aion_embeddings,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        train_ratio=args.train_ratio,
        device=args.device,
        scheduler_t0=args.scheduler_t0,
        scheduler_t_mult=args.scheduler_t_mult,
        scheduler_eta_min=args.scheduler_eta_min,
        gradient_clip_max_norm=args.gradient_clip_max_norm,
        use_mean_embeddings=args.use_mean_embeddings,
        warmup_epochs=args.warmup_epochs,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        # Multi-text parameters
        use_multi_text=args.use_multi_text,
        text_sampling_strategy=args.text_sampling_strategy,
        num_embedding=args.num_embedding,
        # Data scaling parameters
        max_train_samples=args.max_train_samples,
        # Benchmark exclusion
        exclude_crossmatch_path=args.exclude_crossmatch
    )
    
    print(f"\nTraining completed!")
    print(f"Final model saved to: {results['model_path']}")
    print(f"Best model saved to: {results['best_model_path']}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final train loss: {results['history']['train_losses'][-1]:.4f}")
    print(f"Final val loss: {results['history']['val_losses'][-1]:.4f}")

    if wandb.run is not None:
        wandb.finish()
        logger.info("WandB run finished and logged.")


if __name__ == "__main__":
    main()