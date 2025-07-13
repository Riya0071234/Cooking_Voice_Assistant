# src/core/orchestrator.py
"""
The core logic engine (the "brain") of the AI assistant.

This final version of the orchestrator fully integrates all AI components,
including the RAG client and the custom fine-tuned model, to provide
intelligent, context-aware responses.
"""

import logging
from openai import OpenAI

# Import the RAG client we built
from src.core.rag_client import RAGClient


class QueryOrchestrator:
    """
    Handles a user query by intelligently routing it to the best system.
    Implements the final hybrid logic: Emergency -> Intent Classification -> RAG / Fine-tuned LLM.
    """

    def __init__(self, config):
        self.config = config
        self.openai_client = OpenAI(api_key=config.api_keys.openai)

        # Instantiate our functional RAG client
        self.rag_client = RAGClient(config)

        # Get the ID of our custom-trained model from the config
        self.expert_model_id = config.training.fine_tuned_model_id

        self.emergency_keywords = {"fire", "smoke", "burning", "help", "emergency", "spill", "danger"}

    def _classify_intent(self, query_text: str) -> str:
        """Uses a powerful base model to quickly classify the user's intent."""
        logging.info(f"Classifying intent for query: '{query_text}'")
        try:
            # Using a powerful model like gpt-4-turbo for classification ensures high accuracy
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are an API that classifies a user's cooking-related query. Respond with only one of these two categories: 'Troubleshooting/Q&A' for specific problems, errors, or factual questions, or 'Creative/Instructional' for open-ended requests like asking for a recipe, ideas, or general guidance."},
                    {"role": "user", "content": query_text}
                ],
                temperature=0,
                max_tokens=20
            )
            intent = response.choices[0].message.content.strip().replace("'", "")

            # Ensure the response is one of the two expected categories
            if intent in ["Troubleshooting/Q&A", "Creative/Instructional"]:
                logging.info(f"Detected intent: '{intent}'")
                return intent
            else:
                logging.warning(f"Unexpected classification result: '{intent}'. Defaulting to Troubleshooting.")
                return "Troubleshooting/Q&A"

        except Exception as e:
            logging.error(f"Could not classify intent due to API error: {e}")
            return "Troubleshooting/Q&A"

    def handle_query(self, query_text: str) -> dict:
        """Executes the full, intelligent query-handling pipeline."""

        # 1. Emergency Check (Highest Priority)
        if any(keyword in query_text.lower() for keyword in self.emergency_keywords):
            return {
                "response_text": "EMERGENCY DETECTED. Please ensure your immediate safety. Turn off all cooking appliances. If there is a fire, use a fire extinguisher. Do not use water on a grease fire.",
                "intent": "emergency_response",
                "source": "Emergency System"
            }

        # 2. Intent Classification
        intent = self._classify_intent(query_text)

        # 3. Route to the appropriate system
        if intent == "Troubleshooting/Q&A":
            logging.info("Routing to RAG System for a fact-based answer.")
            response_text = self.rag_client.generate_response(query_text)
            return {
                "response_text": response_text,
                "intent": intent,
                "source": "RAG System"
            }

        else:  # Handles "Creative/Instructional" and any fallback
            logging.info(f"Routing to Fine-Tuned Expert LLM ('{self.expert_model_id}').")
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.expert_model_id,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful and friendly cooking assistant. Handle creative requests and generate recipes or ideas with flair."},
                        {"role": "user", "content": query_text}
                    ],
                    temperature=0.8,
                )
                response_text = response.choices[0].message.content
                return {
                    "response_text": response_text,
                    "intent": intent,
                    "source": "Fine-Tuned LLM"
                }
            except Exception as e:
                logging.error(f"Error querying fine-tuned model: {e}. Falling back to RAG.")
                response_text = self.rag_client.generate_response(query_text)
                return {
                    "response_text": response_text,
                    "intent": intent,
                    "source": "RAG System (Fallback)"
                }