"""Small, explainable baseline for aligned machined-part images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

CLASSES = ("normal", "scratch", "dent", "hole")


@dataclass(frozen=True)
class Prediction:
    decision: str
    known_class: str
    class_confidence: float
    anomaly_score: float
    anomaly_threshold: float
    anomaly_map: np.ndarray


def read_image(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def feature_vector(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return intensity/edge/residual statistics for a fixed inspection zone."""
    image = image.astype(np.float32, copy=False)
    pixels = image[mask]
    edges = cv2.Canny((image * 255).astype(np.uint8), 40, 110) > 0
    laplacian = np.abs(cv2.Laplacian(image, cv2.CV_32F))
    values = np.array([
        pixels.mean(), pixels.std(), np.percentile(pixels, 5), np.percentile(pixels, 95),
        edges[mask].mean(), laplacian[mask].mean(), np.percentile(laplacian[mask], 95),
    ], dtype=np.float32)
    return values


class InspectionModel:
    def __init__(self, size: int = 256, confidence_threshold: float = 0.45) -> None:
        self.size = size
        self.confidence_threshold = confidence_threshold

    def _mask(self) -> np.ndarray:
        yy, xx = np.mgrid[:self.size, :self.size]
        distance = np.sqrt((xx - self.size / 2) ** 2 + (yy - self.size / 2) ** 2)
        return (distance < self.size * 0.31) & (distance > self.size * 0.11)

    def fit(self, samples: Iterable[tuple[str, np.ndarray]], anomaly_quantile: float = 0.995) -> "InspectionModel":
        samples = list(samples)
        if not samples:
            raise ValueError("No training samples supplied")
        mask = self._mask()
        normal_images = [image for label, image in samples if label == "normal"]
        if len(normal_images) < 2:
            raise ValueError("At least two normal images are required for anomaly training")
        stack = np.stack(normal_images)
        self.normal_mean = stack.mean(axis=0)
        self.normal_std = np.maximum(stack.std(axis=0), 0.02)
        normal_scores = [self._anomaly_score(image, mask)[0] for image in normal_images]
        self.anomaly_threshold = float(np.quantile(normal_scores, anomaly_quantile))

        features = [(label, feature_vector(image, mask)) for label, image in samples]
        self.feature_mean = np.stack([value for _, value in features]).mean(axis=0)
        self.feature_std = np.maximum(np.stack([value for _, value in features]).std(axis=0), 1e-4)
        self.centroids = {}
        for label in CLASSES:
            class_features = [value for sample_label, value in features if sample_label == label]
            if not class_features:
                raise ValueError(f"Missing class in training data: {label}")
            self.centroids[label] = np.stack(class_features).mean(axis=0)
        return self

    def _anomaly_score(self, image: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.abs(image - self.normal_mean) / self.normal_std
        z[~mask] = 0
        return float(np.quantile(z[mask], 0.995)), z

    def predict(self, image: np.ndarray) -> Prediction:
        if not hasattr(self, "normal_mean"):
            raise RuntimeError("Model has not been fitted")
        mask = self._mask()
        score, anomaly_map = self._anomaly_score(image, mask)
        feature = feature_vector(image, mask)
        distances = {label: float(np.linalg.norm((feature - center) / self.feature_std)) for label, center in self.centroids.items()}
        ordered = sorted(distances.items(), key=lambda pair: pair[1])
        known_class, best = ordered[0]
        runner_up = ordered[1][1]
        confidence = float(np.clip((runner_up - best) / max(runner_up, 1e-6), 0, 1))
        if score > self.anomaly_threshold and confidence < self.confidence_threshold:
            decision = "unknown_anomaly"
        elif confidence < self.confidence_threshold:
            decision = "uncertain"
        else:
            decision = known_class
        return Prediction(decision, known_class, confidence, score, self.anomaly_threshold, anomaly_map)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, size=self.size, confidence_threshold=self.confidence_threshold,
                            normal_mean=self.normal_mean, normal_std=self.normal_std,
                            anomaly_threshold=self.anomaly_threshold, feature_mean=self.feature_mean,
                            feature_std=self.feature_std, **{f"centroid_{key}": value for key, value in self.centroids.items()})

    @classmethod
    def load(cls, path: Path) -> "InspectionModel":
        data = np.load(path)
        model = cls(int(data["size"]), float(data["confidence_threshold"]))
        for key in ("normal_mean", "normal_std", "feature_mean", "feature_std"):
            setattr(model, key, data[key])
        model.anomaly_threshold = float(data["anomaly_threshold"])
        model.centroids = {label: data[f"centroid_{label}"] for label in CLASSES}
        return model
