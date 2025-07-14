# src/scrapers/social_scraper.py
"""
Scrapes contextual Q&A data from Reddit, Instagram, and Quora,
and saves the aggregated results directly to Amazon S3.
"""

import logging
import json
import time
import os
import boto3
from pathlib import Path
from typing import List, Optional
from botocore.exceptions import ClientError
from itertools import islice

import praw
import instaloader
import requests
from bs4 import BeautifulSoup

from src.utils.config_loader import get_config
from pydantic import BaseModel, HttpUrl, Field, ValidationError

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] [%(module)-20s] %(message)s')


class ContextualPost(BaseModel):
    """Pydantic model to validate scraped Q&A-style data."""
    question: str
    answer: str
    source_platform: str
    source_url: HttpUrl
    intent: str = "troubleshooting"
    score: int = 0


class SocialScraper:
    """Handles scraping from various social platforms and saves to S3."""

    def __init__(self, config):
        self.config = config
        self.reddit_config = config.contextual_sources.forums.reddit
        self.quora_config = config.contextual_sources.forums.quora
        self.insta_config = config.contextual_sources.social_media.instagram
        self.keywords = config.scraping.contextual_keywords

        self.s3_client = boto3.client('s3')
        self.reddit_client = self._initialize_reddit_client()
        self.insta_client = self._initialize_insta_client()
        self.http_session = requests.Session()
        self.http_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def _initialize_reddit_client(self):
        if not self.reddit_config.enabled: return None
        try:
            client = praw.Reddit(client_id=self.config.api_keys.reddit_client_id,
                                 client_secret=self.config.api_keys.reddit_client_secret,
                                 user_agent=self.config.api_keys.reddit_user_agent, read_only=True)
            logging.info("✅ Reddit (PRAW) client initialized successfully.")
            return client
        except Exception as e:
            logging.error(f"❌ Failed to initialize Reddit client: {e}")
            return None

    def _initialize_insta_client(self):
        if not self.insta_config.enabled: return None
        user, pwd = os.getenv("INSTAGRAM_USER"), os.getenv("INSTAGRAM_PASSWORD")
        if not user or not pwd:
            logging.error("❌ INSTAGRAM_USER and INSTAGRAM_PASSWORD not found. Disabling Instagram scraping.")
            return None
        try:
            L = instaloader.Instaloader()
            L.login(user, pwd)
            logging.info(f"✅ Instagram client initialized and logged in as '{user}'.")
            return L
        except Exception as e:
            logging.error(f"❌ Failed to initialize Instagram client: {e}")
            return None

    def _scrape_reddit(self) -> List[ContextualPost]:
        if not self.reddit_client: return []
        posts = []
        search_query = " OR ".join(f'"{kw}"' for kw in self.keywords)
        for sub_name in self.reddit_config.subreddits:
            try:
                subreddit = self.reddit_client.subreddit(sub_name)
                for submission in subreddit.search(search_query, limit=25, sort='comments'):
                    if submission.is_self and not submission.stickied and submission.num_comments > 0:
                        submission.comments.replace_more(limit=0)
                        if submission.comments and submission.comments[0].body and len(
                                submission.comments[0].body) > 20:
                            posts.append(ContextualPost(question=submission.title, answer=submission.comments[0].body,
                                                        source_platform="Reddit",
                                                        source_url=f"https://www.reddit.com{submission.permalink}",
                                                        score=submission.score))
                time.sleep(self.config.scraping.delay_between_requests)
            except Exception as e:
                logging.error(f"Failed to scrape r/{sub_name}: {e}")
        logging.info(f"Scraped {len(posts)} posts from Reddit.")
        return posts

    def _scrape_instagram(self) -> List[ContextualPost]:
        if not self.insta_client: return []
        posts = []
        for account in self.insta_config.accounts:
            try:
                profile = instaloader.Profile.from_username(self.insta_client.context, account)
                for post in islice(profile.get_posts(), 15):  # Limit to recent posts
                    if post.caption and any(kw in post.caption.lower() for kw in self.keywords):
                        top_comment = next(post.get_comments(), None)
                        posts.append(ContextualPost(question=post.caption[:300],
                                                    answer=top_comment.text if top_comment else "No answer found.",
                                                    source_platform="Instagram",
                                                    source_url=f"https://www.instagram.com/p/{post.shortcode}/",
                                                    score=post.likes))
                time.sleep(self.config.scraping.delay_between_requests * 5)
            except Exception as e:
                logging.error(f"Failed to scrape Instagram account {account}: {e}")
        logging.info(f"Scraped {len(posts)} posts from Instagram.")
        return posts

    def _scrape_quora(self) -> List[ContextualPost]:
        if not self.quora_config.enabled: return []
        posts = []
        logging.info("Starting Quora scraping (best-effort).")
        for topic in self.quora_config.topics:
            try:
                search_url = f"https://www.quora.com/search?q=cooking+{topic.lower()}"
                response = self.http_session.get(search_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                # This selector is brittle and may need updating if Quora changes their site
                question_links = soup.select('a.q-box.qu-cursor--pointer')
                for link in islice(question_links, 5):
                    question_url = "https://www.quora.com" + link['href']
                    question_text = link.get_text(strip=True)
                    if len(question_text) > 15:
                        posts.append(ContextualPost(question=question_text,
                                                    answer="Answer context would be scraped from the linked page.",
                                                    source_platform="Quora", source_url=question_url))
                time.sleep(self.config.scraping.delay_between_requests)
            except Exception as e:
                logging.error(f"Failed to scrape Quora topic '{topic}': {e}")
        logging.info(f"Scraped {len(posts)} questions from Quora.")
        return posts

    def run(self):
        """Runs the full social scraping process and saves data to S3."""
        reddit_posts = self._scrape_reddit()
        instagram_posts = self._scrape_instagram()
        quora_posts = self._scrape_quora()
        all_posts = reddit_posts + instagram_posts + quora_posts

        if not all_posts:
            logging.warning("No contextual posts were scraped in this run.")
            return

        posts_as_dicts = []
        for post in all_posts:
            try:
                post_dict = post.dict()
                post_dict['source_url'] = str(post.source_url)  # Ensure URL is a string for JSON
                posts_as_dicts.append(post_dict)
            except ValidationError as e:
                logging.warning(f"Skipping a post due to validation error: {e}")

        # Construct S3 path and save data
        output_s3_path = self.config.storage.contextual_data_path + "/scraped_social_posts.json"
        try:
            bucket_name, key = output_s3_path.replace("s3://", "").split("/", 1)
            logging.info(f"Uploading {len(posts_as_dicts)} posts to S3 bucket '{bucket_name}'...")
            self.s3_client.put_object(
                Bucket=bucket_name, Key=key,
                Body=json.dumps(posts_as_dicts, indent=2, ensure_ascii=False),
                ContentType='application/json'
            )
            logging.info("✅ Successfully saved social data to S3.")
        except ClientError as e:
            logging.error(f"Failed to upload to S3: {e}")


def main():
    config = get_config()
    SocialScraper(config).run()


if __name__ == "__main__":
    main()