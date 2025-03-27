import os
import yaml
import json
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralized configuration manager for the application.
    Manages loading and accessing configuration from various sources.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Optional path to the configuration directory
        """
        # Find the config directory relative to this file
        if config_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_dir = os.path.join(base_dir, "assets")
        
        self.config_dir = config_dir
        logger.info(f"ConfigManager initialized with config_dir: {self.config_dir}")
        
        # Initialize configuration cache
        self._config_cache = {}
        
        # Default configurations
        self.defaults = {
            "max_iterations": 6,
            "reports_dir": "reports",
            "debug_dir": "debug",
            "use_hardcoded_cookies": False,
            "include_youtube_transcripts": True
        }
        
        # Load main configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from various sources.
        Priority order: Environment variables > YAML config files > Default values
        """
        try:
            # Try to load main config from yaml
            config_path = os.path.join(self.config_dir, "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self._config_cache.update(yaml.safe_load(f) or {})
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Main config file not found at {config_path}, using defaults")
                
            # Override with environment variables
            for key in self.defaults.keys():
                env_key = f"FIN_FORENSIC_{key.upper()}"
                if env_key in os.environ:
                    value = os.environ[env_key]
                    # Convert to appropriate type based on default
                    default_type = type(self.defaults[key])
                    if default_type == bool:
                        value = value.lower() in ("yes", "true", "t", "1")
                    elif default_type == int:
                        value = int(value)
                    self._config_cache[key] = value
                    logger.info(f"Loaded config from environment: {key}={value}")
                    
        except Exception as e:
            logger.error(f"Error loading configuration: {e}", exc_info=True)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key
            default: Default value if not found
            
        Returns:
            The configuration value
        """
        # Use provided default or system default if available
        default_value = default if default is not None else self.defaults.get(key)
        
        # Return from cache or default
        value = self._config_cache.get(key, default_value)
        return value
    
    def get_path(self, path_key: str, default: str = None) -> str:
        """
        Get a file system path, ensuring it exists.
        
        Args:
            path_key: The configuration key for the path
            default: Default path if not found
            
        Returns:
            The absolute path
        """
        # Get relative path
        rel_path = self.get_config(path_key, default)
        
        if not rel_path:
            logger.warning(f"No path configured for {path_key}, using current directory")
            rel_path = "."
            
        # Ensure the path is absolute
        if not os.path.isabs(rel_path):
            # Base the path on the working directory
            abs_path = os.path.abspath(rel_path)
        else:
            abs_path = rel_path
            
        # Ensure directory exists
        os.makedirs(abs_path, exist_ok=True)
        
        return abs_path
    
    def get_nse_config(self) -> Dict:
        """
        Get NSE-specific configuration.
        
        Returns:
            Dictionary of NSE configuration
        """
        try:
            config_path = os.path.join(self.config_dir, "nse_config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            else:
                logger.warning(f"NSE config file not found at {config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading NSE configuration: {e}", exc_info=True)
            return {}
    
    def get_corporate_governance(self, symbol: str = None) -> Dict:
        """
        Get corporate governance data for a specific symbol or all symbols.
        
        Args:
            symbol: Optional stock symbol to get specific data
            
        Returns:
            Dictionary of corporate governance data
        """
        try:
            json_path = os.path.join(self.config_dir, "corporate_governance.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                    
                if symbol:
                    # Try symbol as is
                    if symbol in data:
                        return data[symbol]
                        
                    # Try variants
                    variants = [
                        symbol.upper(),
                        symbol.lower(),
                        symbol.replace(" ", ""),
                        symbol.strip()
                    ]
                    
                    for variant in variants:
                        if variant in data:
                            return data[variant]
                            
                    logger.warning(f"Symbol '{symbol}' not found in corporate governance data")
                    return {}
                else:
                    return data
            else:
                logger.warning(f"Corporate governance file not found at {json_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading corporate governance data: {e}", exc_info=True)
            return {}