# src/processing/language_detection.py
"""
Detects the language of processed text data, with special handling for Hinglish.

This script reads the tagged data from the 'processed' directory, analyzes
the text content to determine the language (en, hi, hi-en), and enriches the
data by adding a 'language' field.

Key Features:
- Uses the reliable `langdetect` library for baseline detection.
- Implements a custom heuristic to identify "Hinglish" (Hindi words written
  in Roman script mixed with English), a key requirement for the target audience.
- Processes all tagged files and overwrites them with the enriched data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# For consistent detection results
DetectorFactory.seed = 42


class LanguageDetector:
    """
    Analyzes and assigns a language code to text data.
    Supports English ('en'), Hindi ('hi'), and Hinglish ('hi-en').
    """

    def __init__(self, config):
        self.config = config
        self.processed_data_path = Path(config.storage.processed_data_path)
        # A small set of common Romanized Hindi words to help identify Hinglish
        self.hinglish_markers = {
            'aur', 'acha', 'gaya', 'kar', 'nahi', 'sab', 'bhi', 'kya',
            'masala', 'paneer', 'roti', 'dal', 'sabzi', 'tadka', 'ghee',
            'namak', 'jaldi', 'thoda', 'zyada', 'chahiye', 'kaise', 'banate'
        }

    def _detect_language(self, text: str) -> str:
        """
        Detects language with a heuristic for identifying Hinglish.
        """
        if not text or not text.strip():
            return "unknown"

        try:
            # Step 1: Use the standard library to get a baseline detection
            lang = detect(text)

            # Step 2: Apply the Hinglish heuristic
            # If the library detects English, we check for our markers.
            if lang == 'en':
                words = set(text.lower().split())
                # If there's a significant overlap with our markers, classify as Hinglish
                if len(words.intersection(self.hinglish_markers)) >= 2:
                    return 'hi-en'

            return lang

        except LangDetectException:
            # This can happen for very short or ambiguous text
            logging.warning(f"Could not detect language for text snippet: '{text[:50]}...'. Marking as 'unknown'.")
            return "unknown"

    def process_file(self, file_path: Path):
        """Loads a JSON file, detects language for each item, and saves it back."""
        logging.info(f"Processing language detection for {file_path.name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            updated_items = 0
            for item in data:
                # Aggregate text from relevant fields for accurate detection
                text_to_analyze = " ".join([
                    str(item.get('title', '')),
                    str(item.get('question', '')),
                    str(item.get('answer', '')),
                    str(item.get('description', ''))
                ])

                lang = self._detect_language(text_to_analyze)
                item['language'] = lang
                updated_items += 1

            # Overwrite the file with the new language data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logging.info(f"Updated {updated_items} items in {file_path.name} with language codes.")

        except json.JSONDecodeError:
            logging.error(f"Could not decode JSON from file: {file_path.name}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing {file_path.name}: {e}")

    def run(self):
        """
        Runs the language detection process on all tagged files in the
        processed data directory.
        """
        if not self.processed_data_path.exists():
            logging.error(f"Processed data directory not found: {self.processed_data_path}. Aborting.")
            return

        logging.info(f"Starting language detection for files in: {self.processed_data_path}")

        processed_files = list(self.processed_data_path.glob("tagged_*.json"))
        if not processed_files:
            logging.warning("No 'tagged_' JSON files found in the processed directory to analyze.")
            return

        for file_path in processed_files:
            self.process_file(file_path)

        logging.info("✅ Language detection process completed successfully.")


def main():
    """Main entry point for the script."""
    config = get_config()
    detector = LanguageDetector(config)
    detector.run()


if __name__ == "__main__":
    main()