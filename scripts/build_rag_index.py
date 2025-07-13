# scripts/build_rag_index.py
"""
Builds and populates the vector database index for the RAG system.

This script reads the final, processed contextual data, generates vector
embeddings for each Q&A pair using OpenAI's API, and then
upserts these embeddings along with their metadata into Pinecone.
"""

import logging
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from openai import OpenAI
import pinecone

# --- Dynamic Path Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(module)-20s] %(message)s')


class RAGIndexer:
    """Handles the creation and population of a RAG vector index using OpenAI and Pinecone."""

    def __init__(self, config):
        self.config = config
        self.processed_data_path = Path(config.storage.processed_data_path)

        # Add PINECONE_API_KEY and PINECONE_ENVIRONMENT to your .env file
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index_name = "cooking-assistant-rag"

        if not self.pinecone_api_key or not self.pinecone_env:
            raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set in your .env file.")

        try:
            self.openai_client = OpenAI(api_key=config.api_keys.openai)
            self.pinecone_index = self._initialize_pinecone()
            logging.info("OpenAI and Pinecone clients initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize clients: {e}")
            raise

    def _initialize_pinecone(self):
        """Initializes the Pinecone client and creates the index if it doesn't exist."""
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)

        # OpenAI's text-embedding-3-small has 1536 dimensions
        embedding_dimension = 1536

        if self.pinecone_index_name not in pinecone.list_indexes():
            logging.info(
                f"Creating new Pinecone index: '{self.pinecone_index_name}' with {embedding_dimension} dimensions.")
            pinecone.create_index(
                name=self.pinecone_index_name,
                dimension=embedding_dimension,
                metric='cosine'  # Cosine similarity is great for text embeddings
            )
        else:
            logging.info(f"Connecting to existing Pinecone index: '{self.pinecone_index_name}'")

        return pinecone.Index(self.pinecone_index_name)

    def _load_processed_data(self) -> List[Dict[str, Any]]:
        """Loads the final processed contextual Q&A data."""
        # This filename should match the output of your last processing step
        file_path = self.processed_data_path / "lang_tagged_scraped_contextual_posts.json"

        if not file_path.exists():
            logging.error(f"Processed data file not found: {file_path}. Cannot build index.")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run(self):
        """Executes the full indexing pipeline."""
        documents = self._load_processed_data()
        if not documents:
            return

        logging.info(f"Starting to index {len(documents)} documents into Pinecone.")

        # Upsert to Pinecone in batches for efficiency and reliability
        batch_size = 100
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing Batches"):
            batch_docs = documents[i: i + batch_size]

            # Create a concise text chunk for each document for embedding
            texts_to_embed = [
                f"Question: {doc.get('question', '')}\nAnswer: {doc.get('answer', '')}" for doc in batch_docs
            ]

            # 1. Generate embeddings using OpenAI
            try:
                res = self.openai_client.embeddings.create(
                    input=texts_to_embed,
                    model=self.config.rag.embedding_model
                )
                embeddings = [record.embedding for record in res.data]
            except Exception as e:
                logging.error(f"Failed to generate embeddings for batch {i // batch_size}: {e}")
                continue

            # 2. Prepare data for upsert
            vectors_to_upsert = []
            for j, doc in enumerate(batch_docs):
                vector_id = str(doc.get('source_url'))  # Use a unique ID string
                metadata = {
                    "question": doc.get('question', ''),
                    "answer": doc.get('answer', ''),
                    "source_platform": doc.get('source_platform', 'unknown'),
                    "tags": doc.get('tags', []),
                    "language": doc.get('language', 'unknown'),
                    "text_chunk": texts_to_embed[j]
                }
                vectors_to_upsert.append((vector_id, embeddings[j], metadata))

            # 3. Upsert the batch to Pinecone
            try:
                self.pinecone_index.upsert(vectors=vectors_to_upsert)
            except Exception as e:
                logging.error(f"Failed to upsert batch {i // batch_size} to Pinecone: {e}")

        index_stats = self.pinecone_index.describe_index_stats()
        logging.info(f"✅ Indexing complete. Index now contains {index_stats['total_vector_count']} vectors.")


def main():
    """Main entry point for the script."""
    config = get_config()
    indexer = RAGIndexer(config)
    indexer.run()


if __name__ == "__main__":
    main()