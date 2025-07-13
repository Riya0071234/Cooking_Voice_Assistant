# src/core/rag_client.py
"""
A client to handle the Retrieval-Augmented Generation (RAG) process.

This module is responsible for:
1.  Taking a user query.
2.  Converting the query into a vector embedding using OpenAI's API.
3.  Querying a vector database (Pinecone) to find relevant documents.
4.  Constructing a detailed prompt containing the user's query and the
    retrieved context.
5.  Sending the prompt to an OpenAI completion model to generate a final,
    context-aware answer.
"""

import logging
from typing import List

from openai import OpenAI
import pinecone

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(module)-15s] %(message)s')


# You would add these to your .env and config.yaml files
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = "cooking-assistant-rag"

class RAGClient:
    """Manages the full RAG workflow from query to final answer."""

    def __init__(self, config):
        self.config = config
        self.rag_config = config.rag

        # Initialize API clients
        try:
            self.openai_client = OpenAI(api_key=config.api_keys.openai)
            # self.pinecone_index = self._initialize_pinecone()
            logging.info("RAG Client initialized. (Pinecone client is conceptual).")
        except Exception as e:
            logging.error(f"Failed to initialize RAG clients: {e}")
            raise

    def _initialize_pinecone(self):
        """Initializes and returns the Pinecone client and index."""
        # pinecone.init(api_key=PINECONE_API_KEY)
        # return pinecone.Index(PINECONE_INDEX_NAME)
        pass  # Placeholder for actual initialization

    def _get_query_embedding(self, query_text: str) -> List[float]:
        """Creates a vector embedding for the user's query using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=[query_text],
                model=self.rag_config.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Failed to create query embedding: {e}")
            return []

    def _find_relevant_context(self, query_vector: List[float], top_k: int = 3) -> List[str]:
        """Queries the vector database to find the most relevant document chunks."""
        # if not self.pinecone_index:
        #     logging.warning("Pinecone index not initialized. Returning empty context.")
        #     return []
        # try:
        #     results = self.pinecone_index.query(
        #         vector=query_vector,
        #         top_k=top_k,
        #         include_metadata=True
        #     )
        #     context = [match['metadata']['text_chunk'] for match in results['matches']]
        #     return context
        # except Exception as e:
        #     logging.error(f"Failed to query vector database: {e}")
        #     return []
        # This is a placeholder since the pinecone client isn't fully set up yet.
        return [
            "Context 1: Burnt food is often caused by excessively high heat or cooking for too long.",
            "Context 2: To fix burnt onions, remove the burnt parts, deglaze the pan with a little water or stock, and continue with the recipe."
        ]

    def generate_response(self, user_query: str) -> str:
        """
        Executes the full RAG chain: embed, retrieve, and generate.
        """
        logging.info(f"Executing RAG chain for query: '{user_query}'")

        # 1. Embed the user's query
        query_embedding = self._get_query_embedding(user_query)
        if not query_embedding:
            return "I'm sorry, I had trouble understanding your question. Could you please rephrase?"

        # 2. Retrieve relevant context from the vector DB
        retrieved_context = self._find_relevant_context(query_embedding)

        # 3. Engineer the prompt and generate the final answer
        context_str = "\n\n".join(f"- {item}" for item in retrieved_context)
        system_prompt = "You are a helpful and friendly cooking assistant. Answer the user's question based on the provided context. If the context isn't relevant, use your general cooking knowledge."

        user_prompt = f"""
        Context:
        {context_str}

        User's Question: {user_query}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.rag_config.completion_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )
            final_answer = response.choices[0].message.content
            return final_answer
        except Exception as e:
            logging.error(f"Failed to generate final response from OpenAI: {e}")
            return "I'm sorry, I encountered an error while formulating a response. Please try again."


if __name__ == '__main__':
    # Example of how to use the RAGClient
    try:
        config = get_config()
        rag_client = RAGClient(config)

        test_query = "My onions got a little burnt while making butter chicken, what should I do?"
        final_answer = rag_client.generate_response(test_query)

        print("--- RAG Client Test ---")
        print(f"Test Query: {test_query}")
        print("\nGenerated Answer:")
        print(final_answer)

    except Exception as e:
        logging.exception(f"An error occurred during the RAG client test: {e}")