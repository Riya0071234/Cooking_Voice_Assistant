# src/scrapers/youtube_scraper.py
"""
Scrapes YouTube for video metadata, transcripts, and comments.

This module uses the YouTube Data API v3 to find videos from specified channels
and then uses community libraries to fetch high-quality data for each video.

Key Features:
- Driven by the `config.yaml` file for channel lists and API keys.
- Fetches video metadata, transcripts (multi-language), and top comments.
- Validates all collected data with Pydantic for structural integrity.
- Saves structured data to the raw data path for downstream processing.
"""

import logging
import json
import time
from typing import List, Optional, Dict
from pathlib import Path
from pydantic import ValidationError

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pydantic import BaseModel, HttpUrl, Field

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class YouTubeComment(BaseModel):
    author: str
    text: str
    like_count: int


class YouTubeVideoData(BaseModel):
    video_id: str
    title: str
    url: HttpUrl
    description: Optional[str]
    channel_name: str
    view_count: int
    length_seconds: int
    publish_date: str
    transcript: Optional[str]
    comments: List[YouTubeComment] = []


class YouTubeScraper:
    """A class to handle scraping video data from YouTube."""

    def __init__(self, config):
        self.config = config
        self.api_key = config.api_keys.youtube
        self.max_results = config.youtube.max_results_per_channel
        self.youtube_service = self._get_youtube_service()
        self.channel_ids = [channel_id for category in config.youtube.channels.values() for channel_id in category]

    def _get_youtube_service(self):
        """Initializes and returns the YouTube Data API service client."""
        try:
            return build('youtube', 'v3', developerKey=self.api_key)
        except Exception as e:
            logging.error(f"Failed to build YouTube service client: {e}")
            raise

    def get_video_ids_from_channels(self) -> List[str]:
        """Gets a list of video IDs from the configured channels."""
        video_ids = []
        if not self.youtube_service:
            return []

        logging.info(f"Fetching video IDs from {len(self.channel_ids)} channels.")
        for channel_id in self.channel_ids:
            try:
                response = self.youtube_service.search().list(
                    part='id',
                    channelId=channel_id,
                    maxResults=self.max_results,
                    order='date',
                    type='video'
                ).execute()

                for item in response.get('items', []):
                    video_ids.append(item['id']['videoId'])

                time.sleep(self.config.scraping.delay_between_requests)
            except Exception as e:
                logging.error(f"Could not fetch videos for channel {channel_id}: {e}")

        logging.info(f"Found a total of {len(video_ids)} video IDs to process.")
        return list(set(video_ids))  # Return unique video IDs

    def _get_transcript(self, video_id: str) -> Optional[str]:
        """Fetches the full transcript for a video, trying English then Hindi."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
            return " ".join([item['text'] for item in transcript_list])
        except (TranscriptsDisabled, NoTranscriptFound):
            logging.warning(f"No transcript found for video ID: {video_id}")
        except Exception as e:
            logging.error(f"An unexpected error occurred fetching transcript for {video_id}: {e}")
        return None

    def _get_comments(self, video_id: str) -> List[YouTubeComment]:
        """Fetches the top-level comments for a video."""
        comments = []
        if not self.config.contextual_sources.youtube.scrape_comments:
            return comments
        try:
            response = self.youtube_service.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=20,  # Fetch a reasonable number of top comments
                textFormat='plainText'
            ).execute()

            for item in response.get('items', []):
                comment_snippet = item['snippet']['topLevelComment']['snippet']
                comments.append(YouTubeComment(
                    author=comment_snippet.get('authorDisplayName'),
                    text=comment_snippet.get('textDisplay'),
                    like_count=comment_snippet.get('likeCount', 0)
                ))
            return comments
        except Exception as e:
            # Often fails if comments are disabled
            logging.warning(f"Could not fetch comments for video {video_id}: {e}")
            return []

    def get_video_details(self, video_id: str) -> Optional[YouTubeVideoData]:
        """Fetches all details for a single video and validates the data."""
        try:
            response = self.youtube_service.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            ).execute()

            if not response.get('items'):
                logging.warning(f"No video details found for ID: {video_id}")
                return None

            item = response['items'][0]
            snippet = item['snippet']
            # A simple regex to parse ISO 8601 duration format (e.g., PT1M35S)
            import isodate
            duration_seconds = isodate.parse_duration(item['contentDetails']['duration']).total_seconds()

            video_data = YouTubeVideoData(
                video_id=video_id,
                title=snippet['title'],
                url=f"https://www.youtube.com/watch?v={video_id}",
                description=snippet.get('description'),
                channel_name=snippet.get('channelTitle'),
                view_count=int(item['statistics'].get('viewCount', 0)),
                length_seconds=int(duration_seconds),
                publish_date=snippet.get('publishedAt'),
                transcript=self._get_transcript(video_id),
                comments=self._get_comments(video_id)
            )
            return video_data
        except ValidationError as e:
            logging.error(f"Data validation failed for video {video_id}: {e}")
        except Exception as e:
            logging.error(f"Failed to get details for video {video_id}: {e}")
        return None

    def run(self):
        """Runs the full YouTube scraping process."""
        video_ids = self.get_video_ids_from_channels()
        all_video_data = []

        for video_id in video_ids:
            logging.info(f"Processing video: {video_id}")
            details = self.get_video_details(video_id)
            if details:
                all_video_data.append(details.dict())
            time.sleep(self.config.scraping.delay_between_requests)

        # Save to raw data path from config
        output_path = Path(self.config.storage.raw_data_path)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / "scraped_youtube_videos.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(all_video_data, f, indent=2, ensure_ascii=False)
        logging.info(f"✅ Saved data for {len(all_video_data)} videos to {file_path}")


def main():
    """Main entry point for the script."""
    config = get_config()
    if not config.api_keys.youtube or "YOUR_YOUTUBE_API_KEY" in config.api_keys.youtube:
        logging.error("YouTube API key is not configured in config.yaml. Aborting scrape.")
        return

    scraper = YouTubeScraper(config)
    scraper.run()


if __name__ == "__main__":
    main()