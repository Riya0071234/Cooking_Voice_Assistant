# src/scrapers/recipe_scraper.py
"""
Scrapes structured recipe content using the `recipe-scrapers` library.

This version is more robust and uses a specialized library that supports
thousands of recipe sites. It also integrates with boto3 to save the
final output directly to an S3 bucket as defined in the config.
"""

import json
import logging
import time
import boto3
from recipe_scrapers import scrape_me
from botocore.exceptions import ClientError

from src.utils.config_loader import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')


class RecipeScraper:
    """A class to handle scraping recipes from a list of websites."""

    def __init__(self, config):
        self.config = config
        self.recipe_sites = [url for category in self.config.recipe_sites.values() for url in category]
        self.s3_client = boto3.client('s3')

    def scrape_and_format(self, url: str):
        """Uses the scrape_me library to get structured recipe data."""
        try:
            scraper = scrape_me(url, wild_mode=True)
            return {
                "title": scraper.title(),
                "url": url,
                "yields": scraper.yields(),
                "ingredients": scraper.ingredients(),
                "instructions": scraper.instructions_list(),
                "image": scraper.image(),
                "total_time": scraper.total_time(),
                "cuisine": scraper.cuisine(),
                "category": scraper.category()
            }
        except Exception as e:
            logging.error(f"Could not scrape {url}: {e}")
            return None

    def save_to_s3(self, data, s3_path: str):
        """Saves the final data as a JSON file to the specified S3 path."""
        try:
            # S3 paths are in the format "s3://bucket-name/key"
            bucket_name, key = s3_path.replace("s3://", "").split("/", 1)

            logging.info(f"Uploading data to S3 bucket '{bucket_name}' with key '{key}'...")
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=json.dumps(data, indent=2, ensure_ascii=False),
                ContentType='application/json'
            )
            logging.info("✅ Successfully saved data to S3.")
        except ClientError as e:
            logging.error(f"Failed to upload to S3: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during S3 upload: {e}")

    def run(self):
        """Runs the full scraping process for all configured sites."""
        all_scraped_recipes = []
        logging.info(f"Starting recipe scraping from {len(self.recipe_sites)} base URLs.")

        for site_url in self.recipe_sites:
            logging.info(f"Scraping: {site_url}")
            recipe = self.scrape_and_format(site_url)
            if recipe:
                all_scraped_recipes.append(recipe)

            time.sleep(self.config.scraping.delay_between_requests)

        if not all_scraped_recipes:
            logging.warning("No recipes were successfully scraped in this run.")
            return

        # Save the output to the configured raw data path in S3
        output_path_s3 = self.config.storage.raw_data_path + "/scraped_recipes.json"
        self.save_to_s3(all_scraped_recipes, output_path_s3)


def main():
    """Main entry point for the script."""
    config = get_config()
    scraper = RecipeScraper(config)
    scraper.run()


if __name__ == "__main__":
    main()