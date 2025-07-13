# scripts/train_vision_classifier.py
"""
Trains a custom computer vision classifier for cooking stages.

This script uses transfer learning on a pre-trained vision model to classify
images of cooking stages (e.g., 'raw_onions', 'golden_onions').

The workflow is as follows:
1.  Loads the labeled image dataset created by `prepare_vision_dataset.py`.
2.  Applies data augmentation and normalization.
3.  Loads a pre-trained model (e.g., EfficientNet) from torchvision.
4.  Replaces the final layer to match the number of our custom cooking stages.
5.  Trains the new classification layer on our data.
6.  Saves the best performing model weights.
"""

import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from pathlib import Path
from tqdm import tqdm

# --- Dynamic Path Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')


class VisionClassifierTrainer:
    """Orchestrates the training of a custom vision classifier."""

    def __init__(self, config):
        self.config = config.vision_training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Training will use device: {self.device}")

        self.dataset_path = Path(self.config.labeled_dataset_path)
        self.output_path = Path(self.config.output_model_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _prepare_dataloaders(self):
        """Prepares datasets and dataloaders with data augmentation."""
        if not self.dataset_path.exists() or not any(self.dataset_path.iterdir()):
            raise FileNotFoundError(f"Labeled dataset not found or is empty at: {self.dataset_path}")

        # Data augmentation for training, simple resize/normalize for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        full_dataset = datasets.ImageFolder(self.dataset_path)
        # 80/20 split for training and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        train_dataset.dataset.transform = data_transforms['train']
        val_dataset.dataset.transform = data_transforms['val']

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        }

        self.class_names = full_dataset.classes
        logging.info(f"Dataset prepared. Found {len(self.class_names)} classes: {self.class_names}")
        return dataloaders

    def _get_model(self):
        """Loads a pre-trained model and replaces its final layer."""
        logging.info(f"Loading pre-trained base model: {self.config.base_model}")
        # Using efficientnet_b0 as an example
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Freeze all the pre-trained layers
        for param in model.parameters():
            param.requires_grad = False

        # Replace the final classifier layer
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(self.class_names))

        return model.to(self.device)

    def run(self):
        """Executes the full training and validation loop."""
        dataloaders = self._prepare_dataloaders()
        model = self._get_model()

        criterion = nn.CrossEntropyLoss()
        # Observe that only parameters of final layer are being optimized
        optimizer = optim.SGD(model.classifier[1].parameters(), lr=self.config.learning_rate, momentum=0.9)

        best_acc = 0.0

        for epoch in range(self.config.num_epochs):
            logging.info(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")

            # Training phase
            model.train()
            for inputs, labels in tqdm(dataloaders['train'], desc="Training"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            running_corrects = 0
            with torch.no_grad():
                for inputs, labels in tqdm(dataloaders['val'], desc="Validating"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)
            logging.info(f"Validation Accuracy: {epoch_acc:.4f}")

            # Save the best model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), self.output_path)
                logging.info(f"New best model saved to {self.output_path} with accuracy: {best_acc:.4f}")

        logging.info("✅ Vision model training complete.")


def main():
    """Main entry point for the script."""
    config = get_config()
    if not config.vision_training.enabled:
        logging.warning("Vision model training is disabled in the configuration. Skipping.")
        return

    trainer = VisionClassifierTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()