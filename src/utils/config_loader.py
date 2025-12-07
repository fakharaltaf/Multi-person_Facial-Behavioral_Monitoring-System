"""
Configuration loader utility
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage system configuration"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration YAML file
        """
        if config_path is None:
            # Default to system_config.yaml in config directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "system_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'detection.confidence_threshold')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None) -> None:
        """
        Save configuration to file
        
        Args:
            path: Output path (uses original path if None)
        """
        output_path = Path(path) if path else self.config_path
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like assignment"""
        self.set(key, value)


# Global config instance
_config = None


def get_config(config_path: str = None) -> ConfigLoader:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to configuration file (only used on first call)
    
    Returns:
        ConfigLoader instance
    """
    global _config
    if _config is None:
        _config = ConfigLoader(config_path)
    return _config


def reload_config(config_path: str = None) -> ConfigLoader:
    """
    Force reload configuration
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        New ConfigLoader instance
    """
    global _config
    _config = ConfigLoader(config_path)
    return _config
