# src/loaders/contextual_loader.py
"""
Loads raw contextual Q&A data into the database.

This script is responsible for the 'Transform' and 'Load' stages of the ETL
process for contextual data.

Key Features:
- Reads raw JSON files from the location specified in the config.
- Uses a sentence-transformer model for semantic deduplication of questions.
- Validates data against a Pydantic model before processing.
- Loads clean, unique data into the PostgreSQL database in efficient batches.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer, util

from src.utils.config_loader import get_config
from src.models.sql_models import ContextualEntry, get_db_session
from src.scrapers.social_scraper import ContextualPost  # Re-using the Pydantic model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class ContextualLoader:
    """Handles the loading and deduplication of contextual data."""

    def __init__(self, config):
        self.config = config
        self.raw_data_path = Path(config.storage.raw_data_path)
        self.db_session = get_db_session(config.database.url)
        logging.info("Loading sentence transformer model for deduplication...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Model loaded successfully.")

    def _load_from_file(self) -> List[ContextualPost]:
        """Loads the raw contextual posts from the scraped JSON file."""
        file_path = self.raw_data_path / "scraped_contextual_posts.json"
        if not file_path.exists():
            logging.warning(f"Raw data file not found: {file_path}. Skipping load.")
            return []

        posts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                try:
                    # Validate each item against our Pydantic model
                    posts.append(ContextualPost(**item))
                except Exception as e:
                    logging.warning(f"Skipping malformed record: {item}. Error: {e}")
        return posts

    def _deduplicate(self, posts: List[ContextualPost]) -> List[ContextualPost]:
        """
        Removes semantically similar questions to ensure data quality.
        From a cluster of similar questions, it keeps the one with the highest score.
        """
        if not posts:
            return []

        logging.info(f"Deduplicating {len(posts)} posts using semantic similarity...")

        # Encode all questions into vectors
        corpus = [post.question for post in posts]
        embeddings = self.model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

        # Use built-in clustering utility from sentence-transformers
        clusters = util.community_detection(embeddings, min_community_size=2, threshold=0.85)

        unique_posts = []
        is_in_a_cluster = {idx for cluster in clusters for idx in cluster}

        # First, add all posts that are not in any cluster (i.e., are unique)
        for i in range(len(posts)):
            if i not in is_in_a_cluster:
                unique_posts.append(posts[i])

        # Then, for each cluster, find the best post (highest score) and add it
        for cluster in clusters:
            cluster_posts = [posts[i] for i in cluster]
            best_post = max(cluster_posts, key=lambda p: p.score)
            unique_posts.append(best_post)

        logging.info(f"Reduced {len(posts)} posts to {len(unique_posts)} unique entries.")
        return unique_posts

    def run(self):
        """Executes the full loading and deduplication pipeline."""
        raw_posts = self._load_from_file()
        if not raw_posts:
            return

        unique_posts = self._deduplicate(raw_posts)

        logging.info(f"Loading {len(unique_posts)} unique posts into the database...")

        # Get existing URLs from the DB to avoid inserting duplicates
        existing_urls = {res[0] for res in self.db_session.query(ContextualEntry.source_url).all()}

        new_entries = []
        for post in unique_posts:
            if str(post.source_url) not in existing_urls:
                new_entries.append(ContextualEntry(
                    question=post.question,
                    answer=post.answer,
                    source_platform=post.source_platform,
                    source_url=str(post.source_url),
                    score=post.score
                ))

        if new_entries:
            self.db_session.bulk_save_objects(new_entries)
            self.db_session.commit()
            logging.info(f"✅ Successfully inserted {len(new_entries)} new contextual entries into the database.")
        else:
            logging.info("No new contextual entries to add to the database.")

        self.db_session.close()


def main():
    """Main entry point for the script."""
    config = get_config()
    loader = ContextualLoader(config)
    loader.run()


if __name__ == "__main__":
    main()