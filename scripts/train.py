"""Training script for multi-modal reasoning model."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor

from src.data.dataset import MultiModalDataset, collate_fn, create_sample_dataset
from src.models.reasoning_model import MultiModalReasoningModel, ReasoningLoss
from src.eval.metrics import MultiModalEvaluator
from src.utils.config import load_config, resolve_config_paths
from src.utils.device import get_device, set_seed, get_device_info
from src.utils.logging import setup_logging, TensorBoardLogger


class Trainer:
    """Trainer for multi-modal reasoning model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        
        # Set up device and reproducibility
        set_seed(config.get('seed', 42))
        self.device = get_device()
        
        # Set up logging
        self.logger = setup_logging(
            log_level=config.get('logging', {}).get('level', 'INFO'),
            log_file=config.get('logging', {}).get('file'),
        )
        
        # Log device info
        device_info = get_device_info()
        self.logger.info(f"Using device: {device_info}")
        
        # Initialize model and processor
        self.processor = CLIPProcessor.from_pretrained(config['model']['name'])
        self.model = MultiModalReasoningModel(
            model_name=config['model']['name'],
            freeze_vision=config['model'].get('freeze_vision', False),
            freeze_text=config['model'].get('freeze_text', False),
            temperature=config['model'].get('temperature', 0.07),
        ).to(self.device)
        
        # Initialize loss function
        self.criterion = ReasoningLoss(
            contrastive_weight=1.0,
            classification_weight=0.1,
            temperature=config['model'].get('temperature', 0.07),
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01),
        )
        
        # Initialize scheduler
        total_steps = config['training']['num_epochs'] * config['training'].get('steps_per_epoch', 1000)
        warmup_steps = int(config['training'].get('warmup_ratio', 0.1) * total_steps)
        
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )
        
        # Initialize evaluator
        self.evaluator = MultiModalEvaluator(self.device)
        
        # Initialize TensorBoard logger
        self.tb_logger = TensorBoardLogger(config['output']['log_dir'])
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_score = 0.0
    
    def prepare_data(self) -> None:
        """Prepare training and validation data."""
        # Create sample dataset if data doesn't exist
        data_dir = Path(self.config['data']['train_path']).parent
        if not data_dir.exists() or not any(data_dir.glob('*.json')):
            self.logger.info("Creating sample dataset...")
            create_sample_dataset(data_dir, num_samples=1000)
        
        # Load datasets
        self.train_dataset = MultiModalDataset(
            data_path=self.config['data']['train_path'],
            image_dir=self.config['data']['image_dir'],
            processor=self.processor,
            max_length=self.config['data'].get('text_max_length', 77),
            image_size=self.config['data'].get('image_size', 224),
        )
        
        self.val_dataset = MultiModalDataset(
            data_path=self.config['data']['val_path'],
            image_dir=self.config['data']['image_dir'],
            processor=self.processor,
            max_length=self.config['data'].get('text_max_length', 77),
            image_size=self.config['data'].get('image_size', 224),
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=self.config['data'].get('shuffle', True),
            num_workers=self.config['data'].get('num_workers', 4),
            pin_memory=self.config['data'].get('pin_memory', True),
            collate_fn=collate_fn,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 4),
            pin_memory=self.config['data'].get('pin_memory', True),
            collate_fn=collate_fn,
        )
        
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
            )
            
            # Compute loss
            loss_dict = self.criterion(outputs)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config['training'].get('log_every', 100) == 0:
                self.tb_logger.log_scalars(
                    'train/loss',
                    {k: v.item() if isinstance(v, torch.Tensor) else v 
                     for k, v in loss_dict.items()},
                    self.global_step,
                )
                
                self.tb_logger.log_scalar(
                    'train/learning_rate',
                    self.scheduler.get_last_lr()[0],
                    self.global_step,
                )
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0],
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        self.evaluator.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                )
                
                # Compute loss
                loss_dict = self.criterion(outputs)
                loss = loss_dict['total_loss']
                
                # Update evaluator
                self.evaluator.update(outputs)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
        
        # Compute all metrics
        metrics = self.evaluator.get_all_metrics()
        metrics['val_loss'] = total_loss / num_batches
        
        # Log metrics
        for key, value in metrics.items():
            self.tb_logger.log_scalar(f'val/{key}', value, self.global_step)
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            metrics: Validation metrics.
            is_best: Whether this is the best checkpoint.
        """
        checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with score: {metrics.get('recall_at_1_i2t', 0.0):.4f}")
    
    def train(self) -> None:
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Prepare data
        self.prepare_data()
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Recall@1: {val_metrics.get('recall_at_1_i2t', 0.0):.4f}"
            )
            
            # Save checkpoint
            current_score = val_metrics.get('recall_at_1_i2t', 0.0)
            is_best = current_score > self.best_score
            if is_best:
                self.best_score = current_score
            
            if epoch % self.config['training'].get('save_every', 1) == 0:
                self.save_checkpoint(val_metrics, is_best)
        
        self.logger.info("Training completed!")
        self.tb_logger.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train multi-modal reasoning model')
    parser.add_argument('--config', type=str, default='configs/train/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='assets',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override paths if provided
    if args.data_dir:
        config['data']['train_path'] = f"{args.data_dir}/train.json"
        config['data']['val_path'] = f"{args.data_dir}/val.json"
        config['data']['image_dir'] = f"{args.data_dir}/images"
    
    if args.output_dir:
        config['output']['checkpoint_dir'] = f"{args.output_dir}/checkpoints"
        config['output']['log_dir'] = f"{args.output_dir}/logs"
        config['output']['results_dir'] = f"{args.output_dir}/results"
    
    # Resolve paths
    config = resolve_config_paths(config)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
