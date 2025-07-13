# scripts/prepare_vision_dataset.py
"""
Automates the creation of a labeled image dataset from vision pipeline output.

This script reads the master `vision_metadata.json` file, and for each
detection, it copies the corresponding frame into a new directory named after
the detected object's label.

This process transforms our raw vision output into a structured, labeled dataset
ready for training a custom vision classifier model.
"""

import logging
import json
import sys
import shutil
from pathlib import Path
from collections import defaultdict

# --- Dynamic Path Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')


class VisionDatasetPreparer:
    """Sorts image frames into labeled directories based on YOLO detections."""

    def __init__(self, config):
        self.config = config
        self.vision_path = Path(config.storage.vision_data_path)
        self.metadata_file = self.vision_path / "vision_metadata.json"
        self.frames_base_path = self.vision_path / "frames"

        # The new location for our structured, labeled dataset
        self.output_dataset_path = self.vision_path / "labeled_dataset"

        self.min_confidence = config.vision_data.confidence_threshold

    def run(self):
        """Executes the full dataset preparation process."""
        if not self.metadata_file.exists():
            logging.error(
                f"Vision metadata file not found at {self.metadata_file}. Please run the vision pipeline first.")
            return

        logging.info(f"Loading vision metadata from {self.metadata_file}...")
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            all_frames_data = json.load(f)

        if not all_frames_data:
            logging.warning("Vision metadata file is empty. No dataset to prepare.")
            return

        logging.info(f"Found metadata for {len(all_frames_data)} frames. Starting segregation...")

        # Create the main output directory
        self.output_dataset_path.mkdir(parents=True, exist_ok=True)

        copy_counts = defaultdict(int)

        for frame_data in all_frames_data:
            video_id = frame_data.get('video_id')
            frame_filename = frame_data.get('frame_filename')
            source_path = self.frames_base_path / video_id / frame_filename

            if not source_path.exists():
                logging.warning(f"Source frame not found, skipping: {source_path}")
                continue

            for detection in frame_data.get('detections', []):
                if detection['confidence'] >= self.min_confidence:
                    label = detection['label']

                    # Create a directory for the label if it doesn't exist
                    label_dir = self.output_dataset_path / label
                    label_dir.mkdir(exist_ok=True)

                    # Copy the frame into the correct labeled directory
                    destination_path = label_dir / f"{video_id}_{frame_filename}"
                    shutil.copy(source_path, destination_path)
                    copy_counts[label] += 1

        logging.info("✅ Dataset preparation complete.")
        logging.info("--- Image Segregation Summary ---")
        for label, count in sorted(copy_counts.items()):
            logging.info(f"  - Created {count} images for label: '{label}'")


def main():
    """Main entry point for the script."""
    try:
        config = get_config()
        preparer = VisionDatasetPreparer(config)
        preparer.run()
    except Exception as e:
        logging.exception(f"The dataset preparer encountered a fatal error: {e}")


if __name__ == "__main__":
    main()