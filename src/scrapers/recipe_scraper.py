# src/scrapers/recipe_scraper.py
"""
Scrapes structured recipe content from the websites listed in the config.

This scraper is designed to be robust:
1. It first attempts to parse for a 'Recipe' JSON-LD script tag, which is the most
   reliable source of structured data.
2. If JSON-LD is not found, it falls back to parsing common HTML tags (`<ul>`, `<p>`).
3. It validates the extracted data against a Pydantic model to ensure quality.
4. It is driven entirely by the `config.yaml` file for URLs and settings.
"""

import requests
import json
import logging
import time
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl, Field, ValidationError
from typing import List, Optional, Dict

# Get the configuration
from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class ScrapedRecipe(BaseModel):
    """Pydantic model to validate the structure of a scraped recipe."""
    title: str = Field(..., min_length=5)
    url: HttpUrl
    ingredients: List[str] = Field(..., min_length=2)
    instructions: List[str] = Field(..., min_length=2)
    cuisine: Optional[str] = "Unknown"


class RecipeScraper:
    """A class to handle scraping recipes from a list of websites."""

    def __init__(self, config):
        self.config = config
        self.recipe_sites = []
        for sites in self.config.recipe_sites.values():
            self.recipe_sites.extend(sites)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def _parse_from_json_ld(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Attempts to find and parse a 'Recipe' type JSON-LD script."""
        scripts = soup.find_all("script", {"type": "application/ld+json"})
        for script in scripts:
            try:
                data = json.loads(script.string)
                # The @graph key is common for nested JSON-LD
                if '@graph' in data:
                    for item in data['@graph']:
                        if item.get('@type') == 'Recipe':
                            return item
                if data.get('@type') == 'Recipe':
                    return data
            except (json.JSONDecodeError, AttributeError):
                continue
        return None

    def _extract_from_html(self, soup: BeautifulSoup) -> Dict:
        """Fallback method to extract recipe data from common HTML tags."""
        # This contains basic extraction logic; it can be made more sophisticated.
        title = soup.title.string.strip() if soup.title else "Untitled"

        ingredients = []
        # A simple heuristic to find ingredient lists
        for tag in soup.find_all(['ul', 'ol']):
            if 'ingredient' in tag.get('class', '') or 'ingredient' in tag.get('id', ''):
                ingredients.extend([li.get_text(strip=True) for li in tag.find_all('li')])
                break

        instructions = []
        # Simple heuristic for instructions
        for tag in soup.find_all(['ol']):
            if 'instruction' in tag.get('class', '') or 'instruction' in tag.get('id', ''):
                instructions.extend([li.get_text(strip=True) for li in tag.find_all('li')])
                break

        return {
            "name": title,
            "recipeIngredient": ingredients,
            "recipeInstructions": [{"text": step} for step in instructions]  # Mimic JSON-LD structure
        }

    def scrape_recipe_page(self, url: str) -> Optional[ScrapedRecipe]:
        """Scrapes a single recipe page and returns a validated ScrapedRecipe object."""
        try:
            response = self.session.get(url, timeout=self.config.scraping.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            recipe_data = self._parse_from_json_ld(soup)
            if not recipe_data:
                logging.warning(f"No JSON-LD found for {url}. Falling back to HTML parsing.")
                recipe_data = self._extract_from_html(soup)

            # Normalize data and validate with Pydantic
            validated_data = ScrapedRecipe(
                title=recipe_data.get("name", "Untitled"),
                url=url,
                ingredients=recipe_data.get("recipeIngredient", []),
                instructions=[step.get("text", "") for step in recipe_data.get("recipeInstructions", []) if
                              step.get("text")],
                cuisine=recipe_data.get("recipeCuisine", "Unknown")
            )
            return validated_data

        except requests.RequestException as e:
            logging.error(f"Request failed for {url}: {e}")
        except ValidationError as e:
            logging.error(f"Data validation failed for {url}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while scraping {url}: {e}")

        return None

    def run(self):
        """Runs the full scraping process for all configured sites."""
        all_scraped_recipes = []
        logging.info(f"Starting recipe scraping from {len(self.recipe_sites)} base URLs.")

        for site_url in self.recipe_sites:
            # In a real scenario, you'd have a spider here to find all recipe links.
            # For this example, we'll just scrape the base URL as a single page.
            # A more advanced version would use a library like Scrapy or a custom link finder.
            logging.info(f"Scraping: {site_url}")
            recipe = self.scrape_recipe_page(site_url)
            if recipe:
                all_scraped_recipes.append(recipe.dict())

            time.sleep(self.config.scraping.delay_between_requests)

        # Save the output to the configured raw data path
        output_path_str = self.config.storage.raw_data_path
        # A simple check to handle local vs. S3 paths conceptually
        if output_path_str.startswith("s3://"):
            # In a real AWS setup, you'd use the `boto3` library here to upload to S3.
            logging.info(f"Saving {len(all_scraped_recipes)} recipes to S3 (conceptual).")
            # s3_client.put_object(Bucket=..., Key=..., Body=json.dumps(...))
        else:
            from pathlib import Path
            output_path = Path(output_path_str)
            output_path.mkdir(parents=True, exist_ok=True)
            file_path = output_path / "scraped_recipes.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(all_scraped_recipes, f, indent=2, ensure_ascii=False)
            logging.info(f"✅ Saved {len(all_scraped_recipes)} recipes to {file_path}")


def main():
    """Main entry point for the script."""
    config = get_config()
    scraper = RecipeScraper(config)
    scraper.run()


if __name__ == "__main__":
    main()