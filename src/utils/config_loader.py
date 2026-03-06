import yaml
import os
from pathlib import Path

def load_refinery_config():
    """
    Utility to load the centralized YAML configuration.
    """
    config_path = Path(__file__).parents[2] / "config" / "refinery_config.yaml"
    if not config_path.exists():
        return {}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
