import yaml
import os
from pathlib import Path

def load_refinery_config():
    """
    Utility to load the centralized YAML configuration.
    Loads both refinery_config.yaml and extraction_rules.yaml (from rubric folder).
    """
    config_dir = Path(__file__).parents[2] / "config"
    rubric_dir = Path(__file__).parents[2] / "rubric"
    
    # Load main config
    config_path = config_dir / "refinery_config.yaml"
    if not config_path.exists():
        config = {}
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    
    # Load extraction rules from rubric folder
    extraction_rules_path = rubric_dir / "extraction_rules.yaml"
    if extraction_rules_path.exists():
        with open(extraction_rules_path, "r") as f:
            extraction_rules = yaml.safe_load(f) or {}
            config["extraction_rules"] = extraction_rules
    
    return config

