# scripts/train_model.py
"""
Manages the fine-tuning of an OpenAI model on our scraped Q&A data.

This script orchestrates the entire OpenAI fine-tuning workflow:
1.  Loads the processed and cleaned contextual Q&A data.
2.  Formats the data into the required JSONL structure with conversational roles.
3.  Uploads the prepared dataset file to OpenAI.
4.  Creates a new fine-tuning job using the uploaded file.
5.  Monitors the job's progress until completion.
6.  Reports the final, fine-tuned model ID upon success.
"""

import logging
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

# --- Dynamic Path Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] [%(module)-20s] %(message)s')


class OpenAIFineTuner:
    """Manages the process of fine-tuning a model using the OpenAI API."""

    def __init__(self, config):
        self.config = config
        self.training_config = config.training
        self.processed_data_path = Path(config.storage.processed_data_path)
        self.training_file_path = self.processed_data_path / "openai_training_data.jsonl"

        try:
            self.client = OpenAI(api_key=config.api_keys.openai)
            logging.info("OpenAI client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client. Ensure OPENAI_API_KEY is set correctly. Error: {e}")
            raise

    def _prepare_dataset(self) -> bool:
        """
        Loads processed Q&A data and formats it into the JSONL format
        required by OpenAI for fine-tuning.
        """
        source_file = self.processed_data_path / "lang_tagged_scraped_contextual_posts.json"
        if not source_file.exists():
            logging.error(f"Source data file not found: {source_file}")
            return False

        logging.info(f"Preparing dataset from {source_file} for OpenAI fine-tuning...")

        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        formatted_records = 0
        with open(self.training_file_path, 'w', encoding='utf-8') as f_out:
            for item in data:
                question = item.get("question")
                answer = item.get("answer")

                if question and answer:
                    # Create the JSON object with the required message structure
                    record = {
                        "messages": [
                            {"role": "system",
                             "content": "You are a helpful and friendly cooking assistant specializing in troubleshooting cooking problems."},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                    f_out.write(json.dumps(record) + "\n")
                    formatted_records += 1

        logging.info(
            f"Successfully created training file at {self.training_file_path} with {formatted_records} records.")
        return True

    def run(self):
        """Executes the full fine-tuning workflow."""
        if not self.training_config.enabled:
            logging.warning("Training is disabled in the configuration. Skipping.")
            return

        # Step 1: Prepare the dataset file
        if not self._prepare_dataset():
            return

        # Step 2: Upload the file to OpenAI
        logging.info(f"Uploading training file '{self.training_file_path.name}' to OpenAI...")
        with open(self.training_file_path, "rb") as f:
            training_file = self.client.files.create(file=f, purpose="fine-tune")
        logging.info(f"File uploaded successfully. File ID: {training_file.id}")

        # Step 3: Create the fine-tuning job
        base_model = self.training_config.model_to_fine_tune
        logging.info(f"Creating fine-tuning job with model '{base_model}'...")
        job = self.client.fine_tuning.jobs.create(
            training_file=training_file.id,
            model=base_model
        )
        logging.info(f"Fine-tuning job created successfully. Job ID: {job.id}")

        # Step 4: Monitor the job's progress
        logging.info("Monitoring job progress... (This may take some time)")
        while True:
            job_status = self.client.fine_tuning.jobs.retrieve(job.id)
            status = job_status.status
            logging.info(f"Current job status: {status}")

            if status == "succeeded":
                logging.info("✅ Fine-tuning job completed successfully!")
                logging.info(f"Your new fine-tuned model ID is: {job_status.fine_tuned_model}")
                break
            elif status in ["failed", "cancelled"]:
                logging.error(f"❌ Fine-tuning job {status}. Details: {job_status.error}")
                break

            time.sleep(60)  # Wait for 60 seconds before checking the status again


def main():
    """Main entry point for the script."""
    try:
        config = get_config()
        tuner = OpenAIFineTuner(config)
        tuner.run()
    except Exception as e:
        logging.exception(f"The OpenAI fine-tuner encountered a fatal error: {e}")


if __name__ == "__main__":
    main()