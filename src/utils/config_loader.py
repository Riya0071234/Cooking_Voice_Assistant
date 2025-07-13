# src/utils/config_loader.py
"""
Handles loading, validation, and resolution of the project's central configuration.

This module uses Pydantic to enforce the structure of the config.yaml file and
substitutes environment variable placeholders (e.g., ${VAR_NAME}) with their
actual values for secure handling of secrets.
"""

import yaml
import logging
import os
import re
from functools import lru_cache
from pydantic import BaseModel, Field, HttpUrl, DirectoryPath, FilePath
from typing import Dict, List, Any
from pathlib import Path

# Load environment variables from a .env file at the project root
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')

# Regex to find all ${VAR_NAME} placeholders in the YAML file
ENV_VAR_PATTERN = re.compile(r"\$\{(.+?)\}")


def substitute_env_vars(config_item: Any) -> Any:
    """
    Recursively traverses the config and substitutes ${ENV_VAR} placeholders.
    """
    if isinstance(config_item, dict):
        return {key: substitute_env_vars(value) for key, value in config_item.items()}

    if isinstance(config_item, list):
        return [substitute_env_vars(item) for item in config_item]

    if isinstance(config_item, str):
        for match in ENV_VAR_PATTERN.finditer(config_item):
            env_var_name = match.group(1)
            env_var_value = os.getenv(env_var_name)
            if env_var_value is None:
                raise ValueError(f"Required environment variable '{env_var_name}' is not set!")
            config_item = config_item.replace(f"${{{env_var_name}}}", env_var_value)

    return config_item


# --- Pydantic Models for Config Validation ---
# These ensure your config file has the correct structure and types.
class DatabaseConfig(BaseModel):
    url: str


class ApiKeysConfig(BaseModel):
    openai: str
    youtube: str
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str


class RagConfig(BaseModel):
    embedding_model: str
    completion_model: str


# Define other Pydantic models for other sections if strict validation is needed...

class FullConfig(BaseModel):
    """The root Pydantic model for the entire config.yaml file."""
    database: DatabaseConfig
    api_keys: ApiKeysConfig
    rag: RagConfig

    # ... other sections would be defined here

    class Config:
        # Allow other fields not explicitly defined here for flexibility
        extra = 'allow'


@lru_cache()
def get_config(config_path: str = "config/config.yaml") -> FullConfig:
    """
    Loads, substitutes env vars, validates, and returns the application configuration.

    This function is cached to ensure the configuration is processed only once.
    """
    logging.info(f"Loading configuration from: {config_path}")
    try:
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'")

        with open(config_path_obj, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        # ** Step 1: Substitute environment variables **
        resolved_config = substitute_env_vars(raw_config)

        # ** Step 2: Validate with Pydantic **
        validated_config = FullConfig(**resolved_config)

        logging.info("✅ Configuration loaded, resolved, and validated successfully.")
        return validated_config

    except Exception as e:
        logging.exception(f"FATAL: Could not load configuration. Error: {e}")
        raise