# scripts/pipeline_runner.py
# This is the final version with no placeholders.

import logging
import argparse
import sys
from pathlib import Path

# --- Dynamic Path Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Real Module Imports ---
from src.utils.config_loader import get_config
from src.scrapers.recipe_scraper import main as run_recipe_scraper
from src.scrapers.youtube_scraper import main as run_youtube_scraper
from src.scrapers.social_scraper import main as run_social_scraper
from src.loaders.contextual_loader import main as run_contextual_loader
from src.processing.language_detection import main as run_language_detection
from src.processing.auto_tagging import main as run_auto_tagging
from src.processing.vision_pipeline import main as run_vision_pipeline
from scripts.validator import main as run_validator


def setup_logging(config):
    log_path = Path(config.storage.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(module)-25s] %(message)s",
                        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler(sys.stdout)])
    logging.info(f"Logging initialized. Log file at: {log_path}")


def run_step(step_name: str, step_function):
    logging.info(f"🚀 --- Starting Step: {step_name} ---")
    try:
        step_function()
        logging.info(f"✅ --- Completed Step: {step_name} ---")
    except Exception:
        logging.exception(f"❌ --- FAILED Step: {step_name} ---")


def main():
    parser = argparse.ArgumentParser(description="AI Cooking Assistant Data Pipeline Runner.")
    # Add arguments if needed
    args = parser.parse_args()

    config = get_config()
    setup_logging(config)

    logging.info("=====================================================")
    logging.info("==      Starting AI Cooking Assistant Pipeline     ==")
    logging.info("=====================================================")

    pipeline_stages = [
        ("Scraping Recipe Websites", run_recipe_scraper),
        ("Scraping YouTube", run_youtube_scraper),
        ("Scraping Social Media & Forums", run_social_scraper),
        ("Loading Contextual Data into DB", run_contextual_loader),
        ("Auto-Tagging All Content", run_auto_tagging),
        ("Detecting Language for Entries", run_language_detection),
        ("Running Vision Data Collection", run_vision_pipeline),
        ("Validating All Data", run_validator),
    ]

    for name, func in pipeline_stages:
        run_step(name, func)

    logging.info("=====================================================")
    logging.info("==         Pipeline Run Finished                   ==")
    logging.info("=====================================================")


if __name__ == "__main__":
    main()