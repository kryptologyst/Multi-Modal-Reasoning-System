"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        output_path: Path where to save the configuration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        OmegaConf.save(config, f)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configurations.
    
    Args:
        *configs: Configuration objects to merge.
        
    Returns:
        Merged configuration object.
    """
    merged = OmegaConf.create()
    for config in configs:
        merged = OmegaConf.merge(merged, config)
    return merged


def resolve_config_paths(config: DictConfig, base_path: Optional[Union[str, Path]] = None) -> DictConfig:
    """Resolve relative paths in configuration.
    
    Args:
        config: Configuration object.
        base_path: Base path for resolving relative paths.
        
    Returns:
        Configuration with resolved paths.
    """
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)
    
    resolved_config = OmegaConf.create(config)
    
    # Resolve common path fields
    path_fields = ['data_path', 'model_path', 'checkpoint_path', 'output_path', 'log_path']
    
    def resolve_paths(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: resolve_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_paths(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('./') or obj.startswith('../'):
            return str(base_path / obj)
        else:
            return obj
    
    return OmegaConf.create(resolve_paths(resolved_config))


def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """Get a value from configuration with dot notation.
    
    Args:
        config: Configuration object.
        key: Dot-separated key path.
        default: Default value if key not found.
        
    Returns:
        Configuration value or default.
    """
    try:
        return OmegaConf.select(config, key)
    except:
        return default


def set_config_value(config: DictConfig, key: str, value: Any) -> None:
    """Set a value in configuration with dot notation.
    
    Args:
        config: Configuration object.
        key: Dot-separated key path.
        value: Value to set.
    """
    OmegaConf.set(config, key, value)
