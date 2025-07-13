# src/scrapers/social_scraper.py
"""
Scrapes contextual Question & Answer data from social media and forums.

This module is responsible for gathering real-world cooking problems,
questions, and solutions from sources like Reddit, Quora, and Instagram.

--- PLATFORM RELIABILITY ---
- Reddit:   HIGH. Uses the official PRAW API and is reliable.
- Instagram: LOW. Requires login, is heavily rate-limited, and can get your
             account flagged. Use with extreme caution. Provided conceptually.
- Quora:     LOW. Relies on web scraping, which can break easily if Quora
             changes its website's HTML structure.
- Facebook:  VERY LOW. Not implemented due to strong anti-scraping measures
             that make reliable scraping nearly impossible without private APIs.
"""

import logging
import json
import time
from pathlib import Path
from typing import List, Optional

import praw
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl, Field

# Uncomment the line below if you decide to use the Instagram scraper
# import instaloader

from src.utils.config_loader import get_config
from src.models.sql_models import ContextualEntry  # Used for structure reference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class ContextualPost(BaseModel):
    """Pydantic model to validate scraped Q&A-style data."""
    question: str = Field(..., min_length=15)
    answer: str = Field(..., min_length=20)
    source_platform: str
    source_url: HttpUrl
    intent: str = "troubleshooting"
    score: int = 0


class SocialScraper:
    """
    A class to handle scraping contextual Q&A from various social platforms.
    """

    def __init__(self, config):
        self.config = config
        self.reddit_config = config.contextual_sources.forums.reddit
        self.insta_config = config.contextual_sources.social_media.instagram
        self.keywords = config.scraping.contextual_keywords
        self.reddit_client = self._initialize_reddit_client()
        self.insta_client = self._initialize_insta_client()
        self.http_session = requests.Session()
        self.http_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def _initialize_reddit_client(self):
        """Initializes the PRAW Reddit client using credentials from the config."""
        if not self.reddit_config.enabled:
            return None
        try:
            client = praw.Reddit(
                client_id=self.config.api_keys.reddit_client_id,
                client_secret=self.config.api_keys.reddit_client_secret,
                user_agent=self.config.api_keys.reddit_user_agent,
                read_only=True
            )
            logging.info("Reddit (PRAW) client initialized successfully.")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize Reddit client: {e}")
            return None

    def _initialize_insta_client(self):
        """Initializes the Instaloader client. Requires manual configuration."""
        if not self.insta_config.enabled:
            return None
        logging.warning(
            "Instagram scraping is highly unstable and can get your account banned. The functionality is provided conceptually. To enable, uncomment the code in this function and in your requirements.txt, then provide your credentials.")
        # try:
        #     L = instaloader.Instaloader()
        #     # ---- DANGER: THIS REQUIRES YOUR REAL USERNAME AND PASSWORD ----
        #     # L.login("YOUR_INSTAGRAM_USERNAME", "YOUR_INSTAGRAM_PASSWORD")
        #     logging.info("Instaloader client initialized.")
        #     return L
        # except Exception as e:
        #     logging.error(f"Failed to initialize Instaloader: {e}")
        return None

    def _scrape_reddit(self) -> List[ContextualPost]:
        """Scrapes relevant posts and their top comments from configured subreddits."""
        if not self.reddit_client:
            return []

        all_posts = []
        search_query = " OR ".join(f'"{kw}"' for kw in self.keywords)

        for sub_name in self.reddit_config.subreddits:
            logging.info(f"Scraping subreddit: r/{sub_name}")
            try:
                subreddit = self.reddit_client.subreddit(sub_name)
                for submission in subreddit.search(search_query, limit=50, sort='comments'):
                    if submission.is_self and not submission.stickied and submission.num_comments > 0:
                        submission.comments.replace_more(limit=0)
                        if not submission.comments: continue

                        top_comment = submission.comments[0]
                        if top_comment.body and len(top_comment.body) > 20:
                            post = ContextualPost(
                                question=submission.title,
                                answer=top_comment.body,
                                source_platform="Reddit",
                                source_url=f"https://www.reddit.com{submission.permalink}",
                                score=submission.score
                            )
                            all_posts.append(post)
                time.sleep(self.config.scraping.delay_between_requests)
            except Exception as e:
                logging.error(f"Failed to scrape r/{sub_name}: {e}")

        logging.info(f"Scraped {len(all_posts)} posts from Reddit.")
        return all_posts

    def _scrape_instagram(self) -> List[ContextualPost]:
        """Conceptual function for scraping Instagram."""
        if not self.insta_client:
            return []
        logging.info("Executing conceptual Instagram scraping function.")
        # The logic here would be complex: iterate through accounts, get posts,
        # filter by keywords in captions, and then fetch comments for matched posts.
        # This is left as a placeholder due to the high risk and instability.
        return []

    def _scrape_quora(self) -> List[ContextualPost]:
        """Conceptual function for scraping Quora."""
        logging.warning("Quora scraping is brittle and left as a placeholder.")
        return []

    def _scrape_facebook(self) -> List[ContextualPost]:
        """Conceptual function for scraping Facebook."""
        logging.warning("Facebook scraping is not feasible and is disabled.")
        return []

    def run(self):
        """Runs the full social scraping process and saves data to a file."""
        logging.info("Starting social media and forum scraping.")

        # Execute each scraper function
        reddit_posts = self._scrape_reddit()
        instagram_posts = self._scrape_instagram()
        quora_posts = self._scrape_quora()
        facebook_posts = self._scrape_facebook()

        all_contextual_posts = reddit_posts + instagram_posts + quora_posts + facebook_posts

        if not all_contextual_posts:
            logging.warning("No contextual posts were scraped in this run.")
            return

        output_path = Path(self.config.storage.raw_data_path)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / "scraped_contextual_posts.json"

        posts_as_dicts = [post.dict() for post in all_contextual_posts]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(posts_as_dicts, f, indent=2, ensure_ascii=False)

        logging.info(f"✅ Saved {len(all_contextual_posts)} total contextual posts to {file_path}")


def main():
    """Main entry point for the script."""
    config = get_config()
    scraper = SocialScraper(config)
    scraper.run()


if __name__ == "__main__":
    main()