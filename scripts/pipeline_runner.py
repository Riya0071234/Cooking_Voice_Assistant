# scripts/pipeline_runner.py
"""
The master orchestration script for the AI Cooking Assistant's data pipeline.

This script is the single entry point for all data gathering and preparation tasks,
running each stage in the correct sequence by importing the main functions
from their respective modules.
"""

import logging
import argparse
import sys
from pathlib import Path

# --- Dynamic Path Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Module Imports ---
# Import the main functions from our source code modules.
from src.utils.config_loader import get_config
from src.scrapers.recipe_scraper import main as run_recipe_scraper
from src.scrapers.youtube_scraper import main as run_youtube_scraper
from src.scrapers.social_scraper import main as run_social_scraper
from src.loaders.contextual_loader import main as run_contextual_loader
from src.processing.language_detection import main as run_language_detection
from src.processing.auto_tagging import main as run_auto_tagging
from src.processing.vision_pipeline import main as run_vision_pipeline
from scripts.validator import main as run_validator  # This script is already in the scripts folder


def setup_logging(config):
    """Configures centralized logging for the pipeline run."""
    log_path = Path(config.storage.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] [%(module)-25s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file at: {log_path}")


def run_step(step_name: str, step_function):
    """Wrapper to run and log a single pipeline step with error handling."""
    logging.info(f"🚀 --- Starting Step: {step_name} ---")
    try:
        step_function()
        logging.info(f"✅ --- Completed Step: {step_name} ---")
    except Exception as e:
        logging.exception(f"❌ --- FAILED Step: {step_name} ---")


def main():
    """Parses arguments and runs the selected pipeline stages."""
    parser = argparse.ArgumentParser(description="AI Cooking Assistant Data Pipeline Runner.")
    parser.add_argument(
        '--run-only',
        choices=['scraping', 'loading', 'processing', 'vision', 'validation'],
        help='Run only a single stage of the pipeline.'
    )
    # Add other skip arguments if needed
    args = parser.parse_args()

    config = get_config()
    setup_logging(config)

    logging.info("=====================================================")
    logging.info("==      Starting AI Cooking Assistant Pipeline     ==")
    logging.info("=====================================================")

    pipeline_stages = {
        'scraping': [
            ("Scraping Recipe Websites", run_recipe_scraper),
            ("Scraping YouTube", run_youtube_scraper),
            ("Scraping Social Media & Forums", run_social_scraper),
        ],
        'loading': [
            ("Loading Contextual Data into DB", run_contextual_loader),
        ],
        'processing': [
            ("Auto-Tagging All Content", run_auto_tagging),
            ("Detecting Language for Entries", run_language_detection),
        ],
        'vision': [
            ("Running Vision Data Collection", run_vision_pipeline),
        ],
        'validation': [
            ("Validating All Data", run_validator),
        ]
    }

    if args.run_only:
        logging.info(f"Executing ONLY stage: '{args.run_only}'")
        for name, func in pipeline_stages.get(args.run_only, []):
            run_step(name, func)
    else:
        for stage in pipeline_stages.values():
            for name, func in stage:
                run_step(name, func)

    logging.info("=====================================================")
    logging.info("==         Pipeline Run Finished                   ==")
    logging.info("=====================================================")


if __name__ == "__main__":
    main()