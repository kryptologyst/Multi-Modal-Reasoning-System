"""Data loading and preprocessing utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class MultiModalDataset(Dataset):
    """Multi-modal dataset for vision-language tasks."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        image_dir: Union[str, Path],
        processor: CLIPProcessor,
        max_length: int = 77,
        image_size: int = 224,
        transform: Optional[Any] = None,
    ):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the JSON data file.
            image_dir: Directory containing images.
            processor: CLIP processor for text and image processing.
            max_length: Maximum text length.
            image_size: Image size for processing.
            transform: Optional image transforms.
        """
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file.
        
        Returns:
            List of data samples.
        """
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Validate data format
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            raise ValueError("Invalid data format. Expected list or dict with 'data' key.")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a data sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing processed data.
        """
        sample = self.data[idx]
        
        # Load image
        image_path = self.image_dir / sample['image']
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Process text
        text = sample['text']
        if isinstance(text, list):
            text = ' '.join(text)
        
        # Process with CLIP processor
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # Remove batch dimension
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].squeeze(0)
        
        # Add metadata
        inputs['image_path'] = str(image_path)
        inputs['text'] = text
        inputs['idx'] = idx
        
        return inputs


def create_sample_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 100,
    image_size: Tuple[int, int] = (224, 224),
) -> None:
    """Create a sample dataset for testing.
    
    Args:
        output_dir: Directory to save the dataset.
        num_samples: Number of samples to generate.
        image_size: Size of generated images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image directory
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    # Sample data
    sample_data = []
    categories = [
        "cat", "dog", "car", "tree", "house", "person", "bird", "flower",
        "book", "phone", "computer", "chair", "table", "food", "water"
    ]
    
    colors = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
    
    for i in range(num_samples):
        # Generate random image
        image_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Save image
        image_path = f"sample_{i:04d}.jpg"
        image.save(image_dir / image_path)
        
        # Generate text descriptions
        category = np.random.choice(categories)
        color = np.random.choice(colors)
        
        # Create different types of text descriptions
        text_types = [
            f"A {color} {category}",
            f"This is a {category}",
            f"I see a {color} {category} in the image",
            f"The {category} is {color}",
            f"Image shows a {category}",
        ]
        
        text = np.random.choice(text_types)
        
        sample_data.append({
            "image": image_path,
            "text": text,
            "category": category,
            "color": color,
            "id": i,
        })
    
    # Save data
    with open(output_dir / "data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Create train/val/test splits
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:train_size + val_size]
    test_data = sample_data[train_size + val_size:]
    
    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        with open(output_dir / f"{split}.json", 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"Created sample dataset with {num_samples} samples in {output_dir}")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.
    
    Args:
        batch: List of samples from dataset.
        
    Returns:
        Batched tensors.
    """
    # Separate different types of data
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'image_paths': [item['image_path'] for item in batch],
        'texts': [item['text'] for item in batch],
        'indices': torch.tensor([item['idx'] for item in batch]),
    }
