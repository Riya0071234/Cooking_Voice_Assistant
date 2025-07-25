# src/utils/config_loader.py
# This definitive version contains a complete and non-contradictory Pydantic model.

import yaml, logging, os, re
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, HttpUrl, Field
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')
ENV_VAR_PATTERN = re.compile(r"\$\{(.+?)\}")

def substitute_env_vars(config_item: Any) -> Any:
    if isinstance(config_item, dict): return {key: substitute_env_vars(value) for key, value in config_item.items()}
    if isinstance(config_item, list): return [substitute_env_vars(item) for item in config_item]
    if isinstance(config_item, str):
        for match in ENV_VAR_PATTERN.finditer(config_item):
            env_var_name = match.group(1)
            env_var_value = os.getenv(env_var_name)
            if env_var_value is None: raise ValueError(f"Required environment variable '{env_var_name}' is not set!")
            config_item = config_item.replace(f"${{{env_var_name}}}", env_var_value)
    return config_item

# --- Pydantic Models for Full Config Validation ---
class DatabaseConfig(BaseModel): url: str
class ApiKeysConfig(BaseModel): openai: str; youtube: str; reddit_client_id: str; reddit_client_secret: str; reddit_user_agent: str
class RagConfig(BaseModel): embedding_model: str; completion_model: str
class InstagramConfig(BaseModel): enabled: bool; scrape_comments: bool; accounts: List[str]; hashtags: List[str]
class FacebookConfig(BaseModel): enabled: bool; scrape_comments: bool; pages: List[str]; groups: List[str]
class SocialMediaConfig(BaseModel): instagram: InstagramConfig; facebook: FacebookConfig
class RedditConfig(BaseModel): enabled: bool; scrape_comments: bool; subreddits: List[str]
class QuoraConfig(BaseModel): enabled: bool; scrape_answers: bool; topics: List[str]
class ForumsConfig(BaseModel): reddit: RedditConfig; quora: QuoraConfig
class YouTubeConfig(BaseModel): scrape_comments: bool; max_results_per_channel: int; channels: Dict[str, List[str]]
class ContextualSourcesConfig(BaseModel): social_media: SocialMediaConfig; youtube: YouTubeConfig; forums: ForumsConfig
class ScrapingConfig(BaseModel): delay_between_requests: int; max_retries: int; timeout: int; concurrent_workers: int; contextual_keywords: List[str]
class AutoTaggingParams(BaseModel): max_tags_per_item: int; min_word_length: int; top_n_keywords_per_cluster: int
class AutoTaggingConfig(BaseModel): enabled: bool; strategy: str; params: AutoTaggingParams
class ProcessingConfig(BaseModel): deduplication_similarity_threshold: float; auto_tagging: AutoTaggingConfig
class VisionDataConfig(BaseModel): enabled: bool; yolo_model_path: str; confidence_threshold: float; frame_sampling_interval: int
class ImagesConfig(BaseModel): download_enabled: bool; max_size_bytes: int; formats: List[str]
class StorageConfig(BaseModel): raw_data_path: str; processed_data_path: str; contextual_data_path: str; vision_data_path: str; images_path: str; log_path: str
class ValidationRule(BaseModel): min_length: Optional[int] = None; max_length: Optional[int] = None; min_count: Optional[int] = None; max_count: Optional[int] = None; accepted: Optional[List[str]] = None
class ContextualEntryValidation(BaseModel): question: ValidationRule; answer: ValidationRule; tags: ValidationRule; language: ValidationRule
class ValidationConfig(BaseModel): recipe_entry: Dict[str, ValidationRule]; contextual_entry: ContextualEntryValidation
class TrainingConfig(BaseModel): enabled: bool; openai_base_model: str; fine_tuned_model_id: str; dataset_path: str
class VisionTrainingConfig(BaseModel): enabled: bool; labeled_dataset_path: str; output_model_path: str; base_model: str; learning_rate: float; num_epochs: int; batch_size: int

class FullConfig(BaseModel):
    """The root Pydantic model that correctly structures the entire config file."""
    database: DatabaseConfig
    api_keys: ApiKeysConfig
    rag: RagConfig
    recipe_sites: Dict[str, List[HttpUrl]]
    contextual_sources: ContextualSourcesConfig
    # The redundant, top-level 'youtube' key has been removed from here.
    scraping: ScrapingConfig
    processing: ProcessingConfig
    vision_data: VisionDataConfig
    images: ImagesConfig
    storage: StorageConfig
    validation: ValidationConfig
    training: TrainingConfig
    vision_training: VisionTrainingConfig

@lru_cache()
def get_config(config_path: str = "config/config.yaml") -> FullConfig:
    try:
        config_path_obj = Path(config_path)
        if not config_path_obj.exists(): raise FileNotFoundError(f"Configuration file not found at '{config_path}'")
        with open(config_path_obj, 'r', encoding='utf-8') as f: raw_config = yaml.safe_load(f)
        resolved_config = substitute_env_vars(raw_config)
        validated_config = FullConfig(**resolved_config)
        logging.info("✅ Configuration loaded, resolved, and validated successfully.")
        return validated_config
    except Exception as e:
        logging.exception(f"FATAL: Could not load configuration. Error: {e}")
        raise