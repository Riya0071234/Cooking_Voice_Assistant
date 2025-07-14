# src/scrapers/youtube_scraper.py
"""
Scrapes YouTube for video metadata, transcripts, and comments,
and saves the output directly to Amazon S3.
"""
import logging
import json
import time
import isodate
import boto3
from pathlib import Path
from typing import List, Optional
from botocore.exceptions import ClientError

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pydantic import BaseModel, HttpUrl, ValidationError

from src.utils.config_loader import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] [%(module)-20s] %(message)s')


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
    """A class to handle scraping video data from YouTube and saving to S3."""

    def __init__(self, config):
        self.config = config
        self.api_key = config.api_keys.youtube
        self.youtube_config = config.contextual_sources.youtube
        self.max_results = self.youtube_config.max_results_per_channel
        self.channel_ids = [channel_id for category in self.youtube_config.channels.values() for channel_id in category]
        self.s3_client = boto3.client('s3')
        self.youtube_service = self._get_youtube_service()

    def _get_youtube_service(self):
        try:
            return build('youtube', 'v3', developerKey=self.api_key)
        except Exception as e:
            logging.error(f"Failed to build YouTube service client: {e}")
            raise

    def get_video_ids_from_channels(self) -> List[str]:
        video_ids = []
        if not self.youtube_service: return []
        logging.info(f"Fetching video IDs from {len(self.channel_ids)} channels.")
        for channel_id in self.channel_ids:
            try:
                response = self.youtube_service.search().list(
                    part='id', channelId=channel_id, maxResults=self.max_results, order='date', type='video'
                ).execute()
                video_ids.extend([item['id']['videoId'] for item in response.get('items', [])])
                time.sleep(self.config.scraping.delay_between_requests)
            except Exception as e:
                logging.error(f"Could not fetch videos for channel {channel_id}: {e}")
        unique_ids = list(set(video_ids))
        logging.info(f"Found a total of {len(unique_ids)} unique video IDs to process.")
        return unique_ids

    def _get_transcript(self, video_id: str) -> Optional[str]:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
            return " ".join([item['text'] for item in transcript_list])
        except (TranscriptsDisabled, NoTranscriptFound):
            logging.warning(f"No transcript found for video ID: {video_id}")
        except Exception as e:
            logging.error(f"An unexpected error fetching transcript for {video_id}: {e}")
        return None

    def _get_comments(self, video_id: str) -> List[YouTubeComment]:
        comments = []
        if not self.youtube_config.scrape_comments: return comments
        try:
            response = self.youtube_service.commentThreads().list(
                part='snippet', videoId=video_id, maxResults=20, textFormat='plainText'
            ).execute()
            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append(YouTubeComment(author=snippet.get('authorDisplayName'), text=snippet.get('textDisplay'), like_count=snippet.get('likeCount', 0)))
        except Exception as e:
            logging.warning(f"Could not fetch comments for video {video_id}: {e}")
        return comments

    def get_video_details(self, video_id: str) -> Optional[dict]:
        try:
            response = self.youtube_service.videos().list(part='snippet,contentDetails,statistics', id=video_id).execute()
            if not response.get('items'):
                logging.warning(f"No video details found for ID: {video_id}")
                return None
            item = response['items'][0]
            snippet = item['snippet']
            duration_seconds = isodate.parse_duration(item['contentDetails']['duration']).total_seconds()
            validated_data = YouTubeVideoData(
                video_id=video_id, title=snippet['title'], url=f"https://www.youtube.com/watch?v={video_id}",
                description=snippet.get('description'), channel_name=snippet.get('channelTitle'),
                view_count=int(item['statistics'].get('viewCount', 0)), length_seconds=int(duration_seconds),
                publish_date=snippet.get('publishedAt'), transcript=self._get_transcript(video_id),
                comments=self._get_comments(video_id)
            )
            # Convert to dict and ensure URL is a string for JSON serialization
            final_dict = validated_data.dict()
            final_dict['url'] = str(final_dict['url'])
            return final_dict
        except ValidationError as e:
            logging.error(f"Data validation failed for video {video_id}: {e}")
        except Exception as e:
            logging.error(f"Failed to get details for video {video_id}: {e}")
        return None

    def save_to_s3(self, data, s3_path: str):
        if not data:
            logging.warning("No YouTube data to save to S3.")
            return
        try:
            bucket_name, key = s3_path.replace("s3://", "").split("/", 1)
            logging.info(f"Uploading data for {len(data)} videos to S3 bucket '{bucket_name}'...")
            self.s3_client.put_object(Bucket=bucket_name, Key=key, Body=json.dumps(data, indent=2, ensure_ascii=False), ContentType='application/json')
            logging.info("✅ Successfully saved YouTube data to S3.")
        except ClientError as e:
            logging.error(f"Failed to upload YouTube data to S3: {e}")

    def run(self):
        video_ids = self.get_video_ids_from_channels()
        all_video_data = [details for video_id in video_ids if (details := self.get_video_details(video_id))]
        output_s3_path = self.config.storage.raw_data_path + "/scraped_youtube_videos.json"
        self.save_to_s3(all_video_data, output_s3_path)

def main():
    config = get_config()
    if not config.api_keys.youtube or "YOUR_YOUTUBE_API_KEY" in config.api_keys.youtube:
        logging.error("YouTube API key is not configured. Aborting scrape.")
        return
    YouTubeScraper(config).run()

if __name__ == "__main__":
    main()