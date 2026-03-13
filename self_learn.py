"""Self-Learning Pipeline for Camera Detection

Collects user feedback on detections, builds training datasets,
fine-tunes YOLOv8, and manages model versions with rollback support.
"""

import json
import uuid
import shutil
import logging
import random
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────

FACE_DATA_DIR = Path("face_data")
TRAINING_DIR = FACE_DATA_DIR / "training"
MODELS_DIR = Path("models")
FEEDBACK_FILE = FACE_DATA_DIR / "feedback.json"
MODEL_REGISTRY_FILE = MODELS_DIR / "registry.json"

for _dir in [TRAINING_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─── Data Classes ────────────────────────────────────────

@dataclass
class Feedback:
    id: str
    detection_id: str
    correct: bool
    label: str
    image_path: str
    timestamp: str


@dataclass
class ModelVersion:
    version: str
    path: str
    accuracy: float
    feedback_count: int
    created_at: str
    is_active: bool


# ─── Feedback Collector ────────────────────────────────────

class FeedbackCollector:
    """Collects and stores user feedback on detections."""

    def __init__(self):
        self.feedbacks: list[Feedback] = []
        self._load()

    def _load(self):
        if FEEDBACK_FILE.exists():
            try:
                data = json.loads(FEEDBACK_FILE.read_text())
                self.feedbacks = [Feedback(**f) for f in data]
            except Exception as e:
                logger.error(f"Failed to load feedback: {e}")
                self.feedbacks = []

    def _save(self):
        FEEDBACK_FILE.write_text(
            json.dumps([asdict(f) for f in self.feedbacks], indent=2)
        )

    def add_feedback(self, detection_id: str, correct: bool, label: str,
                     image_data: Optional[bytes] = None) -> Feedback:
        """Store feedback + optional crop image."""
        feedback_id = str(uuid.uuid4())[:8]

        # Save crop image if provided
        image_path = ""
        if image_data:
            safe_label = label.replace("/", "_").replace("..", "_").replace("\\", "_").strip()
            if not safe_label:
                safe_label = "unknown"
            class_dir = TRAINING_DIR / safe_label
            class_dir.mkdir(parents=True, exist_ok=True)
            img_filename = f"{feedback_id}.jpg"
            img_path = class_dir / img_filename
            img_path.write_bytes(image_data)
            image_path = str(img_path)

        feedback = Feedback(
            id=feedback_id,
            detection_id=detection_id,
            correct=correct,
            label=label,
            image_path=image_path,
            timestamp=datetime.now().astimezone().isoformat(),
        )

        self.feedbacks.append(feedback)
        self._save()
        logger.info(f"Feedback #{feedback_id}: detection={detection_id} correct={correct} label={label}")
        return feedback

    def get_summary(self) -> dict:
        """Return feedback counts per class and correct/incorrect totals."""
        summary: dict[str, dict[str, int]] = {}
        total_correct = 0
        total_incorrect = 0

        for f in self.feedbacks:
            if f.label not in summary:
                summary[f.label] = {"correct": 0, "incorrect": 0, "total": 0}
            if f.correct:
                summary[f.label]["correct"] += 1
                total_correct += 1
            else:
                summary[f.label]["incorrect"] += 1
                total_incorrect += 1
            summary[f.label]["total"] += 1

        return {
            "per_class": summary,
            "total_correct": total_correct,
            "total_incorrect": total_incorrect,
            "total": len(self.feedbacks),
        }

    def get_pending_count(self) -> int:
        """Count feedback entries with images available for training."""
        return sum(1 for f in self.feedbacks if f.image_path)


# ─── Dataset Builder ────────────────────────────────────

class DatasetBuilder:
    """Builds YOLO-format dataset from collected feedback images."""

    def __init__(self, train_ratio: float = 0.8):
        self.train_ratio = train_ratio

    def build(self) -> Optional[Path]:
        """Build dataset.yaml from training images. Returns path or None if insufficient data."""
        classes: dict[str, list[Path]] = {}

        for class_dir in TRAINING_DIR.iterdir():
            if not class_dir.is_dir():
                continue
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if images:
                classes[class_dir.name] = images

        if not classes:
            logger.warning("No training images found")
            return None

        total_images = sum(len(imgs) for imgs in classes.values())
        if total_images < 50:
            logger.info(f"Only {total_images} images, need >= 50 for retrain")
            return None

        # Create dataset structure
        dataset_dir = FACE_DATA_DIR / "dataset"
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"

        # Clean previous dataset
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        for split_dir in [train_dir, val_dir]:
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)

        class_names = sorted(classes.keys())

        for cls_idx, cls_name in enumerate(class_names):
            images = classes[cls_name]
            random.shuffle(images)
            split_point = int(len(images) * self.train_ratio)
            train_imgs = images[:split_point]
            val_imgs = images[split_point:]

            for img_path in train_imgs:
                self._copy_with_label(img_path, train_dir, cls_idx)
            for img_path in val_imgs:
                self._copy_with_label(img_path, val_dir, cls_idx)

        # Augment training images
        self._augment(train_dir / "images")

        # Write dataset.yaml
        yaml_path = dataset_dir / "dataset.yaml"
        yaml_content = (
            f"path: {dataset_dir.resolve()}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"nc: {len(class_names)}\n"
            f"names: {class_names}\n"
        )
        yaml_path.write_text(yaml_content)

        logger.info(f"Dataset built: {total_images} images, {len(class_names)} classes at {yaml_path}")
        return yaml_path

    def _copy_with_label(self, img_path: Path, split_dir: Path, class_idx: int):
        """Copy image and create YOLO label (full-image bounding box)."""
        dest_img = split_dir / "images" / img_path.name
        shutil.copy2(img_path, dest_img)

        # Create label with full-image bbox (crop images are already the object)
        label_path = split_dir / "labels" / img_path.with_suffix(".txt").name
        label_path.write_text(f"{class_idx} 0.5 0.5 1.0 1.0\n")

    def _augment(self, images_dir: Path):
        """Simple augmentation: horizontal flip + brightness adjustment."""
        for img_path in list(images_dir.glob("*.jpg")):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Horizontal flip
                flipped = cv2.flip(img, 1)
                flip_path = images_dir / f"aug_flip_{img_path.name}"
                cv2.imwrite(str(flip_path), flipped)

                # Copy label for augmented image
                label_dir = images_dir.parent / "labels"
                orig_label = label_dir / img_path.with_suffix(".txt").name
                if orig_label.exists():
                    flip_label = label_dir / f"aug_flip_{img_path.with_suffix('.txt').name}"
                    shutil.copy2(orig_label, flip_label)

                # Brightness adjustment
                bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
                bright_path = images_dir / f"aug_bright_{img_path.name}"
                cv2.imwrite(str(bright_path), bright)
                if orig_label.exists():
                    bright_label = label_dir / f"aug_bright_{img_path.with_suffix('.txt').name}"
                    shutil.copy2(orig_label, bright_label)

            except Exception as e:
                logger.warning(f"Augmentation failed for {img_path.name}: {e}")


# ─── Model Manager ────────────────────────────────────

class ModelManager:
    """Manages model versions with rollback support."""

    def __init__(self):
        self.versions: list[ModelVersion] = []
        self._load()

    def _load(self):
        if MODEL_REGISTRY_FILE.exists():
            try:
                data = json.loads(MODEL_REGISTRY_FILE.read_text())
                self.versions = [ModelVersion(**v) for v in data]
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
                self.versions = []

    def _save(self):
        MODEL_REGISTRY_FILE.write_text(
            json.dumps([asdict(v) for v in self.versions], indent=2)
        )

    def get_active(self) -> Optional[ModelVersion]:
        for v in self.versions:
            if v.is_active:
                return v
        return None

    def register(self, model_path: str, accuracy: float, feedback_count: int) -> ModelVersion:
        """Register a new model version."""
        version_num = len(self.versions) + 1
        version_name = f"custom_v{version_num}"

        # Copy model to versioned path
        dest = MODELS_DIR / f"{version_name}.pt"
        shutil.copy2(model_path, dest)

        # Deactivate all previous
        for v in self.versions:
            v.is_active = False

        new_version = ModelVersion(
            version=version_name,
            path=str(dest),
            accuracy=accuracy,
            feedback_count=feedback_count,
            created_at=datetime.now().astimezone().isoformat(),
            is_active=True,
        )

        self.versions.append(new_version)
        self._save()
        logger.info(f"Registered model {version_name}: accuracy={accuracy:.4f}")
        return new_version

    def rollback(self, version_name: str) -> Optional[ModelVersion]:
        """Rollback to a specific version."""
        target = None
        for v in self.versions:
            if v.version == version_name:
                target = v
                break

        if not target:
            return None

        if not Path(target.path).exists():
            logger.error(f"Model file not found: {target.path}")
            return None

        for v in self.versions:
            v.is_active = False
        target.is_active = True
        self._save()
        logger.info(f"Rolled back to {version_name}")
        return target

    def get_stats(self) -> dict:
        """Return model stats summary."""
        active = self.get_active()
        return {
            "total_versions": len(self.versions),
            "active_version": active.version if active else "base (yolov8n)",
            "active_accuracy": active.accuracy if active else None,
            "active_path": active.path if active else None,
            "last_retrain": active.created_at if active else None,
            "versions": [
                {
                    "version": v.version,
                    "accuracy": v.accuracy,
                    "feedback_count": v.feedback_count,
                    "created_at": v.created_at,
                    "is_active": v.is_active,
                }
                for v in self.versions
            ],
        }


# ─── Model Trainer ────────────────────────────────────

class ModelTrainer:
    """Fine-tunes YOLOv8 on custom feedback data."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self._lock = threading.Lock()
        self.is_training = False

    def train(self, dataset_yaml: Path, epochs: int = 50, imgsz: int = 640) -> Optional[ModelVersion]:
        """Fine-tune YOLOv8 with dataset. Returns new ModelVersion or None on failure."""
        if not self._lock.acquire(blocking=False):
            logger.warning("Training already in progress")
            return None

        self.is_training = True
        try:
            from ultralytics import YOLO

            # Start from active custom model or base model
            active = self.model_manager.get_active()
            base_model = active.path if active and Path(active.path).exists() else "yolov8n.pt"

            logger.info(f"Starting training: base={base_model}, epochs={epochs}, imgsz={imgsz}")
            model = YOLO(base_model)
            results = model.train(
                data=str(dataset_yaml),
                epochs=epochs,
                imgsz=imgsz,
                device="cpu",
                verbose=False,
            )

            # Get best model path from training output
            best_path = Path(results.save_dir) / "weights" / "best.pt"
            if not best_path.exists():
                logger.error("Training completed but best.pt not found")
                return None

            # Evaluate on validation set
            val_results = model.val()
            accuracy = float(val_results.box.map50) if hasattr(val_results.box, "map50") else 0.0

            # Compare with current model
            current = self.model_manager.get_active()
            if current and accuracy < current.accuracy:
                logger.warning(
                    f"New model ({accuracy:.4f}) worse than current ({current.accuracy:.4f}), skipping deploy"
                )
                return None

            # Count feedback images used
            feedback_count = sum(
                len(list(d.glob("*.jpg")) + list(d.glob("*.png")))
                for d in TRAINING_DIR.iterdir()
                if d.is_dir()
            )

            version = self.model_manager.register(str(best_path), accuracy, feedback_count)
            logger.info(f"Training complete: {version.version} accuracy={accuracy:.4f}")
            return version

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
        finally:
            self.is_training = False
            self._lock.release()


# ─── Main Pipeline ────────────────────────────────────

class SelfLearnPipeline:
    """Orchestrates the full self-learning pipeline."""

    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.dataset_builder = DatasetBuilder()
        self.model_manager = ModelManager()
        self.model_trainer = ModelTrainer(self.model_manager)

    def submit_feedback(self, detection_id: str, correct: bool, label: str,
                        image_data: Optional[bytes] = None) -> Feedback:
        return self.feedback_collector.add_feedback(detection_id, correct, label, image_data)

    def should_retrain(self) -> bool:
        return self.feedback_collector.get_pending_count() >= 50

    def retrain(self, force: bool = False) -> dict:
        """Trigger retrain if enough data or forced."""
        pending = self.feedback_collector.get_pending_count()
        if not force and pending < 50:
            return {"status": "skipped", "reason": f"Only {pending} images, need >= 50"}

        if self.model_trainer.is_training:
            return {"status": "skipped", "reason": "Training already in progress"}

        dataset_yaml = self.dataset_builder.build()
        if not dataset_yaml:
            return {"status": "failed", "reason": "Failed to build dataset"}

        version = self.model_trainer.train(dataset_yaml)
        if not version:
            return {"status": "failed", "reason": "Training failed or new model worse than current"}

        return {
            "status": "success",
            "version": version.version,
            "accuracy": version.accuracy,
            "feedback_count": version.feedback_count,
        }

    def get_model_stats(self) -> dict:
        stats = self.model_manager.get_stats()
        feedback_summary = self.feedback_collector.get_summary()
        return {
            **stats,
            "total_feedback": feedback_summary["total"],
            "feedback_summary": feedback_summary,
            "pending_images": self.feedback_collector.get_pending_count(),
            "ready_to_retrain": self.should_retrain(),
        }
