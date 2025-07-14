# scripts/pipeline_runner.py
# This version uses the corrected paths to access config values.

import logging, argparse, sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config_loader import get_config
from src.scrapers.recipe_scraper import main as run_recipe_scraper
from src.scrapers.youtube_scraper import main as run_youtube_scraper
from src.scrapers.social_scraper import main as run_social_scraper


# ... other imports ...
# Placeholders for unwritten scripts
def placeholder_function(name): logging.info(f"Executing placeholder for: {name}")


run_contextual_loader = lambda: placeholder_function("Contextual Loader")
run_language_detection = lambda: placeholder_function("Language Detection")
run_auto_tagging = lambda: placeholder_function("Auto Tagging")
run_vision_pipeline = lambda: placeholder_function("Vision Pipeline")
run_validator = lambda: placeholder_function("Validator")


def setup_logging(config):
    # This function remains the same
    log_path = Path(config.storage.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(module)-25s] %(message)s",
                        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler(sys.stdout)])
    logging.info(f"Logging initialized. Log file at: {log_path}")


def run_step(step_name: str, step_function):
    # This function remains the same
    logging.info(f"🚀 --- Starting Step: {step_name} ---")
    try:
        step_function()
        logging.info(f"✅ --- Completed Step: {step_name} ---")
    except Exception as e:
        logging.exception(f"❌ --- FAILED Step: {step_name} ---")


def main():
    parser = argparse.ArgumentParser(description="AI Cooking Assistant Data Pipeline Runner.")
    parser.add_argument('--run-only', choices=['scraping', 'loading', 'processing', 'vision', 'validation'],
                        help='Run only a single stage of the pipeline.')
    args = parser.parse_args()

    config = get_config()
    setup_logging(config)

    logging.info("=====================================================")
    logging.info("==      Starting AI Cooking Assistant Pipeline     ==")
    logging.info("=====================================================")

    # This dictionary is now just for organizing the functions.
    # The config object is loaded once and accessed by each module directly.
    pipeline_stages = {
        'scraping': [
            ("Scraping Recipe Websites", run_recipe_scraper),
            ("Scraping YouTube", run_youtube_scraper),
            ("Scraping Social Media & Forums", run_social_scraper),
        ],
        'loading': [("Loading Contextual Data", run_contextual_loader)],
        'processing': [("Auto-Tagging Content", run_auto_tagging), ("Detecting Language", run_language_detection)],
        'vision': [("Running Vision Data Collection", run_vision_pipeline)],
        'validation': [("Validating All Data", run_validator)]
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