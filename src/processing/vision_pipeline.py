# src/processing/vision_pipeline.py
"""
Processes scraped YouTube videos to extract visual data and saves it to S3.

This pipeline downloads videos, samples frames at a configured interval,
and runs a YOLO object detection model on each frame to identify ingredients
and cooking utensils. The structured visual data and the frame images are
then uploaded to an S3 bucket.
"""

import logging
import json
import sys
import cv2
import boto3
from pathlib import Path
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError

from ultralytics import YOLO
from pytube import YouTube
from pydantic import BaseModel, Field

# --- Dynamic Path Setup ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] [%(module)-25s] %(message)s')


class DetectedObject(BaseModel):
    """Represents a single object detected in a frame."""
    label: str
    confidence: float = Field(..., ge=0, le=1)
    box: List[int]  # [x1, y1, x2, y2]


class VisionFrameData(BaseModel):
    """Represents the complete visual analysis of a single video frame."""
    video_id: str
    frame_s3_key: str  # The S3 key for the saved frame image
    timestamp_seconds: float
    detections: List[DetectedObject]


class VisionDataCollector:
    """Orchestrates the download, analysis, and upload of video data."""

    def __init__(self, config):
        self.config = config
        self.vision_config = config.vision_data
        self.storage_config = config.storage
        self.raw_data_path = self.storage_config.raw_data_path
        self.vision_output_path = self.storage_config.vision_data_path

        # In a production system, you might download the model from S3 if it doesn't exist locally
        logging.info(f"Loading YOLO model from: {self.vision_config.yolo_model_path}")
        self.yolo_model = YOLO(self.vision_config.yolo_model_path)
        self.s3_client = boto3.client('s3')
        logging.info("YOLO model and S3 client initialized successfully.")

    def _load_scraped_videos(self) -> List[Dict[str, Any]]:
        """Loads the list of videos scraped by youtube_scraper.py from S3."""
        s3_path = self.raw_data_path + "/scraped_youtube_videos.json"
        try:
            bucket_name, key = s3_path.replace("s3://", "").split("/", 1)
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            content = response['Body'].read().decode('utf-8')
            return json.loads(content)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logging.error(f"Video data file not found at {s3_path}. Please run the YouTube scraper first.")
            else:
                logging.error(f"Failed to load video data from S3: {e}")
        return []

    def _process_video(self, video_info: Dict[str, Any]) -> List[VisionFrameData]:
        """Downloads, analyzes, and cleans up a single video."""
        video_id = video_info.get("video_id")
        video_url = video_info.get("url")
        if not video_id or not video_url: return []

        logging.info(f"--- Processing video ID: {video_id} ---")
        video_frames_data = []
        video_path: Optional[Path] = None

        try:
            # Step 1: Download the video
            yt = YouTube(video_url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not stream:
                logging.error(f"No suitable stream found for video {video_id}")
                return []

            # Use a temporary local directory for downloads
            temp_dir = Path("/tmp/video_downloads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Downloading '{yt.title}' to temporary location...")
            video_path = Path(stream.download(output_path=str(temp_dir)))

            # Step 2: Process frames and upload to S3
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * self.vision_config.frame_sampling_interval)

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break

                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps

                    # Run YOLO detection
                    results = self.yolo_model(frame, verbose=False)
                    detections = [
                        DetectedObject(label=res.names[int(box.cls)], confidence=float(box.conf),
                                       box=[int(c) for c in box.xyxy[0]])
                        for res in results for box in res.boxes
                    ]

                    # Encode frame to memory buffer and upload to S3
                    success, buffer = cv2.imencode('.jpg', frame)
                    if not success: continue

                    frame_filename = f"frame_{frame_count}.jpg"
                    s3_bucket, base_key = self.vision_output_path.replace("s3://", "").split("/", 1)
                    frame_s3_key = f"{base_key}/frames/{video_id}/{frame_filename}"

                    self.s3_client.put_object(Bucket=s3_bucket, Key=frame_s3_key, Body=buffer.tobytes(),
                                              ContentType='image/jpeg')

                    video_frames_data.append(VisionFrameData(
                        video_id=video_id,
                        frame_s3_key=frame_s3_key,
                        timestamp_seconds=timestamp,
                        detections=detections
                    ))
                frame_count += 1

            cap.release()
            logging.info(f"  Processed and uploaded {len(video_frames_data)} frames for video {video_id}.")

        except Exception as e:
            logging.exception(f"An error occurred while processing video {video_id}: {e}")
        finally:
            if video_path and video_path.exists():
                video_path.unlink()  # Clean up downloaded video file

        return video_frames_data

    def run(self):
        """Main function to run the entire vision data collection process."""
        videos_to_process = self._load_scraped_videos()
        if not videos_to_process: return

        all_vision_data = []
        for video_info in videos_to_process:
            vision_data_for_video = self._process_video(video_info)
            all_vision_data.extend(vision_data_for_video)

        if not all_vision_data:
            logging.warning("No vision data was generated in this run.")
            return

        # Save all metadata to a single JSON file in S3
        output_s3_path = self.vision_output_path + "/vision_metadata.json"
        try:
            bucket_name, key = output_s3_path.replace("s3://", "").split("/", 1)
            data_to_save = [item.dict() for item in all_vision_data]
            self.s3_client.put_object(Bucket=bucket_name, Key=key, Body=json.dumps(data_to_save, indent=2),
                                      ContentType='application/json')
            logging.info(
                f"✅ Vision pipeline complete. Saved metadata for {len(all_vision_data)} frames to {output_s3_path}.")
        except ClientError as e:
            logging.error(f"Failed to upload vision metadata to S3: {e}")


def main():
    """Main entry point for the script."""
    config = get_config()
    if not config.vision_data.enabled:
        logging.warning("Vision pipeline is disabled in the configuration. Skipping.")
        return

    VisionDataCollector(config).run()


if __name__ == "__main__":
    main()