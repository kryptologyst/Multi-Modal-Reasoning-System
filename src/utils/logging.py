"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file path.
        log_format: Optional custom log format.
        
    Returns:
        Configured logger.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger = logging.getLogger()
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(self, log_dir: str):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.
        
        Args:
            tag: Tag for the scalar.
            value: Scalar value.
            step: Step number.
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        """Log multiple scalar values.
        
        Args:
            main_tag: Main tag for the scalars.
            tag_scalar_dict: Dictionary of tag-value pairs.
            step: Step number.
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int) -> None:
        """Log an image.
        
        Args:
            tag: Tag for the image.
            img_tensor: Image tensor.
            step: Step number.
        """
        self.writer.add_image(tag, img_tensor, step)
    
    def log_images(self, tag: str, img_tensor: torch.Tensor, step: int) -> None:
        """Log multiple images.
        
        Args:
            tag: Tag for the images.
            img_tensor: Image tensor with batch dimension.
            step: Step number.
        """
        self.writer.add_images(tag, img_tensor, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram.
        
        Args:
            tag: Tag for the histogram.
            values: Values for the histogram.
            step: Step number.
        """
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text_string: str, step: int) -> None:
        """Log text.
        
        Args:
            tag: Tag for the text.
            text_string: Text to log.
            step: Step number.
        """
        self.writer.add_text(tag, text_string, step)
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
