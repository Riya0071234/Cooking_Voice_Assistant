# scripts/validator.py
"""
Validates data integrity in the database against rules in the config.

This script connects to the PostgreSQL database and checks entries in key tables
(e.g., recipes, contextual_entries) to ensure they meet the quality standards
defined in the `validation` section of the `config.yaml` file.

It's a crucial utility for maintaining a high-quality dataset for your AI models.
"""

import logging
import sys
from pathlib import Path
from sqlalchemy.orm import Session

# --- Dynamic Path Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Module Imports ---
from src.utils.config_loader import get_config
from src.models.sql_models import get_db_session, Recipe, ContextualEntry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')


class Validator:
    """A class to validate database records based on dynamic rules."""

    def __init__(self, config):
        self.config = config
        self.session: Session = get_db_session(config.database.url)
        # Load validation rules directly from the config
        self.validation_rules = config.validation

    def _validate_recipes(self):
        """Validates all entries in the 'recipes' table."""
        logging.info("--- 🔎 Starting Recipe Validation ---")
        rules = self.validation_rules.recipe_entry
        recipes = self.session.query(Recipe).all()

        valid_count = 0
        broken_entries = []

        for recipe in recipes:
            errors = []
            if len(recipe.title) < rules.title.min_length:
                errors.append(f"title too short (min: {rules.title.min_length})")

            if not (rules.ingredients.min_count <= len(recipe.ingredients) <= rules.ingredients.max_count):
                errors.append(
                    f"ingredient count out of range (min: {rules.ingredients.min_count}, max: {rules.ingredients.max_count})")

            if not (rules.instructions.min_count <= len(recipe.instructions) <= rules.instructions.max_count):
                errors.append(
                    f"instruction count out of range (min: {rules.instructions.min_count}, max: {rules.instructions.max_count})")

            if errors:
                broken_entries.append({'id': recipe.id, 'title': recipe.title, 'errors': errors})
            else:
                valid_count += 1

        logging.info(f"Recipe Validation Summary: ✅ Valid: {valid_count} | ❌ Broken: {len(broken_entries)}")
        if broken_entries:
            logging.warning("--- Broken Recipe Details ---")
            for entry in broken_entries:
                logging.warning(f"  - ID {entry['id']} ('{entry['title']}') failed: {', '.join(entry['errors'])}")

    def _validate_contextual_entries(self):
        """Validates all entries in the 'contextual_entries' table."""
        logging.info("--- 🔎 Starting Contextual Entry Validation ---")
        rules = self.validation_rules.contextual_entry
        entries = self.session.query(ContextualEntry).all()

        valid_count = 0
        broken_entries = []

        for entry in entries:
            errors = []
            q_rules = rules.question
            a_rules = rules.answer

            if not (q_rules.min_length <= len(entry.question) <= q_rules.max_length):
                errors.append(f"question length out of range (min: {q_rules.min_length}, max: {q_rules.max_length})")

            if not (a_rules.min_length <= len(entry.answer) <= a_rules.max_length):
                errors.append(f"answer length out of range (min: {a_rules.min_length}, max: {a_rules.max_length})")

            if entry.tags and len(entry.tags) < rules.tags.min_count:
                errors.append(f"tag count too low (min: {rules.tags.min_count})")

            if entry.language not in rules.language.accepted:
                errors.append(f"language '{entry.language}' not accepted")

            if errors:
                broken_entries.append({'id': entry.id, 'question': entry.question[:50], 'errors': errors})
            else:
                valid_count += 1

        logging.info(f"Contextual Entry Validation Summary: ✅ Valid: {valid_count} | ❌ Broken: {len(broken_entries)}")
        if broken_entries:
            logging.warning("--- Broken Contextual Entry Details ---")
            for entry in broken_entries:
                logging.warning(f"  - ID {entry['id']} ('{entry['question']}...') failed: {', '.join(entry['errors'])}")

    def run(self):
        """Runs all validation checks."""
        logging.info("=============================================")
        logging.info("==         Running Data Validator          ==")
        logging.info("=============================================")

        self._validate_recipes()
        self._validate_contextual_entries()

        self.session.close()
        logging.info("Validator run complete.")


def main():
    """Main entry point for the script."""
    try:
        config = get_config()
        validator = Validator(config)
        validator.run()
    except Exception as e:
        logging.exception(f"The validator encountered a fatal error: {e}")


if __name__ == "__main__":
    main()