# src/processing/auto_tagging.py
"""
Dynamically generates and applies tags to all scraped data.

This script implements a sophisticated auto-tagging strategy:
1.  Loads all raw data (recipes, YouTube videos, social posts).
2.  Aggregates and cleans the text content from these sources.
3.  Uses TF-IDF to identify important keywords in each document.
4.  Applies K-Means clustering to group documents into distinct topics.
5.  Extracts the top keywords from each topic cluster to create a dynamic tag set.
6.  Assigns these tags back to the original data items.
7.  Saves the newly enriched and tagged data to the processed data directory.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class AutoTagger:
    """
    A class to dynamically analyze and tag content from various sources.
    """

    def __init__(self, config):
        self.config = config
        self.raw_data_path = Path(config.storage.raw_data_path)
        self.processed_data_path = Path(config.storage.processed_data_path)
        self.params = config.processing.auto_tagging.params

        # English and common Hinglish/transliterated stopwords
        self.stopwords = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
            'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'out', 'on', 'off', 'over',
            'under', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
            'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
            'ka', 'ke', 'ki', 'ko', 'hai', 'mein', 'se', 'aur', 'kya', 'recipe', 'video', 'watch', 'http'
        ])

        # Initialize ML models
        self.vectorizer = TfidfVectorizer(
            max_df=0.8,
            min_df=5,
            stop_words=list(self.stopwords),
            ngram_range=(1, 2)
        )
        # Using 50 clusters to find a good number of topics. This can be tuned.
        self.clusterer = KMeans(n_clusters=50, random_state=42, n_init=10)
        # LSA for dimensionality reduction to improve clustering quality
        self.lsa = TruncatedSVD(n_components=100, random_state=42)

    def _clean_text(self, text: str) -> str:
        """Applies basic text cleaning."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    def _load_and_prepare_data(self) -> List[Dict[str, Any]]:
        """Loads all raw JSON files and aggregates text for processing."""
        documents = []
        if not self.raw_data_path.exists():
            logging.warning(f"Raw data path does not exist: {self.raw_data_path}")
            return []

        for file_path in self.raw_data_path.glob("*.json"):
            logging.info(f"Loading data from: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    # Combine all relevant text fields into one string for analysis
                    text_parts = [
                        item.get('title', ''),
                        item.get('description', ''),
                        " ".join(item.get('ingredients', [])),
                        " ".join(item.get('instructions', [])),
                        item.get('transcript', ''),
                        item.get('question', ''),
                        item.get('answer', '')
                    ]
                    full_text = self._clean_text(" ".join(text_parts))

                    # Keep a reference to the original item
                    documents.append({
                        "id": f"{file_path.stem}_{i}",
                        "original_item": item,
                        "source_file": file_path.name,
                        "text": full_text
                    })
        return documents

    def run(self):
        """Executes the full auto-tagging pipeline."""
        documents = self._load_and_prepare_data()
        if not documents:
            logging.info("No documents found to process. Exiting auto-tagging.")
            return

        logging.info(f"Processing {len(documents)} documents for tagging.")

        # --- Step 1: Vectorize Text ---
        corpus = [doc['text'] for doc in documents]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # --- Step 2: Reduce Dimensionality and Cluster ---
        lsa_matrix = self.lsa.fit_transform(tfidf_matrix)
        clusters = self.clusterer.fit_predict(lsa_matrix)

        # --- Step 3: Discover Tags from Clusters ---
        logging.info("Discovering tags from topic clusters...")
        terms = self.vectorizer.get_feature_names_out()
        topic_tags = {}
        for i in range(self.clusterer.n_clusters):
            # Find the original TF-IDF vectors for the current cluster
            cluster_indices = [idx for idx, label in enumerate(clusters) if label == i]
            if not cluster_indices:
                continue

            # Get the average TF-IDF score for each term within this cluster
            cluster_vector = tfidf_matrix[cluster_indices].mean(axis=0)

            # Get top N terms for this cluster
            top_term_indices = cluster_vector.argsort()[0, -self.params.top_n_keywords_per_cluster:]
            # Reverse to get descending order
            top_terms = [terms[idx] for idx in reversed(top_term_indices.tolist()[0])]
            topic_tags[i] = top_terms

        logging.info(f"Discovered {len(topic_tags)} topics with tags.")

        # --- Step 4: Apply Tags and Save Processed Data ---
        tagged_data_by_source = {}
        for doc, cluster_label in zip(documents, clusters):
            source_file = doc['source_file']
            if source_file not in tagged_data_by_source:
                tagged_data_by_source[source_file] = []

            # Assign the discovered tags to the original item
            tagged_item = doc['original_item']
            tagged_item['tags'] = topic_tags.get(cluster_label, [])
            tagged_data_by_source[source_file].append(tagged_item)

        # --- Step 5: Save to Processed Directory ---
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        for source_file, data in tagged_data_by_source.items():
            output_filename = self.processed_data_path / f"tagged_{source_file}"
            logging.info(f"Saving {len(data)} tagged items to {output_filename}")
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        logging.info("✅ Auto-tagging process completed successfully.")


def main():
    """Main entry point for the script."""
    config = get_config()
    if not config.processing.auto_tagging.enabled:
        logging.warning("Auto-tagging is disabled in the configuration. Skipping.")
        return

    tagger = AutoTagger(config)
    tagger.run()


if __name__ == "__main__":
    main()