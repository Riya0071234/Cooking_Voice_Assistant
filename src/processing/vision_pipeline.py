# src/processing/vision_pipeline.py
"""
Processes scraped YouTube videos to extract visual data.

This pipeline downloads videos, samples frames at a configured interval,
and runs a YOLO object detection model on each frame to identify ingredients
and cooking utensils. The structured visual data is then saved for later use.

Key Features:
- Integrates with the output of the `youtube_scraper.py`.
- Uses the powerful `ultralytics` YOLOv8 for object detection.
- Fully configurable via the `vision_data` section of `config.yaml`.
- Structures output using Pydantic models for data integrity.
- Manages file cleanup to conserve disk space.
"""

import logging
import json
import sys
import cv2
from pathlib import Path
from typing import List, Dict, Any

from ultralytics import YOLO
from pytube import YouTube
from pydantic import BaseModel, Field

# --- Dynamic Path Setup ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(module)-25s] %(message)s')


class DetectedObject(BaseModel):
    """Represents a single object detected in a frame."""
    label: str
    confidence: float = Field(..., ge=0, le=1)
    box: List[int]  # [x1, y1, x2, y2]


class VisionFrameData(BaseModel):
    """Represents the complete visual analysis of a single video frame."""
    video_id: str
    frame_filename: str
    timestamp_seconds: float
    detections: List[DetectedObject]


class VisionDataCollector:
    """Orchestrates the download and processing of video data."""

    def __init__(self, config):
        self.config = config
        self.vision_config = config.vision_data
        self.storage_config = config.storage
        self.raw_data_path = Path(self.storage_config.raw_data_path)
        self.vision_output_path = Path(self.storage_config.vision_data_path)

        logging.info(f"Loading YOLO model from: {self.vision_config.model_path}")
        self.yolo_model = YOLO(self.vision_config.model_path)
        logging.info("YOLO model loaded successfully.")

    def _load_scraped_videos(self) -> List[Dict[str, Any]]:
        """Loads the list of videos scraped by youtube_scraper.py."""
        video_file = self.raw_data_path / "scraped_youtube_videos.json"
        if not video_file.exists():
            logging.error(f"Video data file not found at {video_file}. Please run the YouTube scraper first.")
            return []
        with open(video_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _process_video(self, video_info: Dict[str, Any]) -> List[VisionFrameData]:
        """Downloads, analyzes, and cleans up a single video."""
        video_id = video_info.get("video_id")
        video_url = video_info.get("url")
        if not video_id or not video_url:
            return []

        logging.info(f"--- Processing video ID: {video_id} ---")
        video_frames_data = []
        video_path = None

        try:
            # Step 1: Download the video
            yt = YouTube(video_url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not stream:
                logging.error(f"No suitable stream found for video {video_id}")
                return []

            download_dir = self.vision_output_path / "temp_videos"
            download_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Downloading '{yt.title}'...")
            video_path = Path(stream.download(output_path=str(download_dir)))

            # Step 2: Process frames
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * self.vision_config.frame_sampling_interval)

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    logging.info(f"  - Analyzing frame at {timestamp:.2f}s")

                    # Run YOLO detection
                    results = self.yolo_model(frame, verbose=False)
                    detected_objects = []
                    for res in results:
                        for box in res.boxes:
                            detected_objects.append(DetectedObject(
                                label=res.names[int(box.cls)],
                                confidence=float(box.conf),
                                box=[int(coord) for coord in box.xyxy[0]]
                            ))

                    # Save frame image
                    frame_dir = self.vision_output_path / "frames" / video_id
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    frame_filename = f"frame_{frame_count}.jpg"
                    frame_filepath = frame_dir / frame_filename
                    cv2.imwrite(str(frame_filepath), frame)

                    video_frames_data.append(VisionFrameData(
                        video_id=video_id,
                        frame_filename=frame_filename,
                        timestamp_seconds=timestamp,
                        detections=detected_objects
                    ))
                frame_count += 1

            cap.release()

        except Exception as e:
            logging.exception(f"An error occurred while processing video {video_id}: {e}")
        finally:
            # Step 3: Clean up downloaded video file
            if video_path and video_path.exists():
                video_path.unlink()
                logging.info(f"Cleaned up video file: {video_path.name}")

        return video_frames_data

    def run(self):
        """Main function to run the entire vision data collection process."""
        videos_to_process = self._load_scraped_videos()
        if not videos_to_process:
            return

        all_vision_data = []
        for video_info in videos_to_process:
            vision_data_for_video = self._process_video(video_info)
            all_vision_data.extend(vision_data_for_video)

        if not all_vision_data:
            logging.warning("No vision data was generated in this run.")
            return

        # Save all metadata to a single JSON file
        output_file = self.vision_output_path / "vision_metadata.json"
        # Convert Pydantic objects to dicts for JSON serialization
        data_to_save = [item.dict() for item in all_vision_data]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)

        logging.info(f"✅ Vision pipeline complete. Saved metadata for {len(all_vision_data)} frames to {output_file}.")


def main():
    """Main entry point for the script."""
    config = get_config()
    if not config.vision_data.enabled:
        logging.warning("Vision pipeline is disabled in the configuration. Skipping.")
        return

    collector = VisionDataCollector(config)
    collector.run()


if __name__ == "__main__":
    main()