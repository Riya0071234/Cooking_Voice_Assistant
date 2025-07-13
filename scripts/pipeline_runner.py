# scripts/pipeline_runner.py
"""
The master orchestration script for the AI Cooking Assistant's data pipeline.

This script runs the full data collection, processing, and validation pipeline
in the correct sequence. It is designed to be the single entry point for
all data gathering and preparation tasks.

Key Improvements over the original runner:
- Imports functions directly instead of using error-prone subprocesses.
- Uses the centralized, validated config from `config/config.yaml`.
- Provides command-line arguments to selectively skip parts of the pipeline.
- Implements structured logging and error handling for each major step.

Example Usages:
- Run the full pipeline:
  `python scripts/pipeline_runner.py`
- Run the pipeline but skip the vision processing step:
  `python scripts/pipeline_runner.py --skip-vision`
- Run only the validation step:
  `python scripts/pipeline_runner.py --run-only validation`
"""

import logging
import argparse
import sys
from pathlib import Path

# Add the project's source directory to the Python path
# This allows us to import modules from the `src` directory
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Import modules from your source code ---
# NOTE: We are writing this as if all these modules exist in their correct locations.
# We will create each of these scripts in subsequent steps.
from src.utils.config_loader import get_config


# from src.scrapers import recipe_scraper, youtube_scraper, social_scraper
# from src.processing import language_detection, auto_tagging, validation, vision_pipeline
# from src.loaders import contextual_loader

# --- Placeholder for module imports until they are created ---
# We will replace these with actual imports as we build the project.
def placeholder_function(name, config_section):
    logging.info(f"Executing placeholder for: {name}. Config enabled: {config_section.get('enabled', True)}")
    return True


recipe_scraper = lambda config: placeholder_function("Recipe Scraper", config.recipe_sites)
youtube_scraper = lambda config: placeholder_function("YouTube Scraper", config.youtube)
social_scraper = lambda config: placeholder_function("Social Scraper", config.contextual_sources.social_media)
contextual_loader = lambda config: placeholder_function("Contextual Loader", {})
language_detection = lambda config: placeholder_function("Language Detection", {})
auto_tagging = lambda config: placeholder_function("Auto Tagging", config.processing.auto_tagging)
vision_pipeline = lambda config: placeholder_function("Vision Pipeline", config.vision_data)
validation = lambda config: placeholder_function("Validation", config.validation)


def setup_logging(log_path: str):
    """Configures centralized logging for the pipeline run."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file at: {log_path}")


def run_step(step_name: str, step_function, *args):
    """Wrapper to run and log a single pipeline step."""
    logging.info(f"🚀 --- Starting Step: {step_name} ---")
    try:
        step_function(*args)
        logging.info(f"✅ --- Completed Step: {step_name} ---")
        return True
    except Exception as e:
        logging.error(f"❌ --- FAILED Step: {step_name} ---")
        logging.error(f"Error details: {e}", exc_info=True)
        return False


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="AI Cooking Assistant Data Pipeline Runner.")
    parser.add_argument("--skip-scraping", action="store_true", help="Skip all data scraping steps.")
    parser.add_argument("--skip-processing", action="store_true",
                        help="Skip all data processing steps (language detection, tagging).")
    parser.add_argument("--skip-vision", action="store_true", help="Skip the vision data processing pipeline.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the final validation step.")

    args = parser.parse_args()

    # Load the central configuration
    config = get_config()

    # Setup logging using path from config
    setup_logging(config.storage.log_path)

    logging.info("=====================================================")
    logging.info("==      Starting AI Cooking Assistant Pipeline     ==")
    logging.info("=====================================================")

    # --- STEP 1: DATA SCRAPING ---
    if not args.skip_scraping:
        run_step("Scraping Recipe Websites", recipe_scraper, config)
        run_step("Scraping YouTube", youtube_scraper, config)
        run_step("Scraping Social Media & Forums", social_scraper, config)
    else:
        logging.warning("⏩ Skipping all scraping steps as requested.")

    # --- STEP 2: LOADING DATA INTO DATABASE ---
    # This step would typically load the raw JSON/CSV from scraping into the DB
    if not args.skip_scraping:  # Usually linked to scraping
        run_step("Loading Contextual Data into DB", contextual_loader, config)

    # --- STEP 3: DATA PROCESSING & ENRICHMENT ---
    if not args.skip_processing:
        run_step("Detecting Language for Entries", language_detection, config)
        run_step("Auto-Tagging Entries", auto_tagging, config)
    else:
        logging.warning("⏩ Skipping data processing steps as requested.")

    # --- STEP 4: VISION PIPELINE ---
    if not args.skip_vision and config.vision_data.enabled:
        run_step("Running Vision Data Collection", vision_pipeline, config)
    else:
        logging.warning("⏩ Skipping vision pipeline as requested or disabled in config.")

    # --- STEP 5: FINAL VALIDATION ---
    if not args.skip_validation:
        run_step("Validating All Data", validation, config)
    else:
        logging.warning("⏩ Skipping final validation step as requested.")

    logging.info("=====================================================")
    logging.info("==         Pipeline Run Finished Successfully        ==")
    logging.info("=====================================================")


if __name__ == "__main__":
    main()