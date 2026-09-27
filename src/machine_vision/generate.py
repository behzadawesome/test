"""Generate aligned synthetic parts and defect masks for prototype experiments."""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from .model import CLASSES

SIZE = 256


def make_part(label: str) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((SIZE, SIZE, 3), random.randint(25, 45), dtype=np.float32)
    annotation = np.zeros((SIZE, SIZE), dtype=np.uint8)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cx, cy, radius = SIZE / 2 + random.uniform(-4, 4), SIZE / 2 + random.uniform(-4, 4), random.uniform(78, 88)
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    part_mask = distance <= radius
    angle = random.uniform(0, 2 * np.pi)
    intensity = 110 + 45 * np.clip(1 - distance / radius, 0, 1)
    intensity += 18 * ((xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)) / SIZE
    intensity += cv2.GaussianBlur(np.random.normal(0, 5, (SIZE, SIZE)).astype(np.float32), (0, 0), 1.5)
    for channel in range(3):
        image[:, :, channel][part_mask] = intensity[part_mask]
    center = (int(cx), int(cy))
    cv2.circle(image, center, int(radius), (180, 183, 185), 2)
    for groove in range(int(radius * 0.35), int(radius * 0.9), random.randint(7, 10)):
        cv2.circle(image, center, groove, (125, 128, 130), 1)
    bore = int(random.uniform(18, 25))
    cv2.circle(image, center, bore, (35, 38, 40), -1)
    cv2.circle(image, center, bore, (190, 192, 194), 2)
    valid = ((distance < radius - 10) & (distance > bore + 10)).astype(np.uint8)
    bolt_radius, hole_radius, rotation = random.uniform(45, 52), random.uniform(4, 6), random.uniform(0, np.pi / 2)
    for index in range(4):
        px, py = int(cx + bolt_radius * np.cos(rotation + index * np.pi / 2)), int(cy + bolt_radius * np.sin(rotation + index * np.pi / 2))
        cv2.circle(image, (px, py), int(hole_radius), (38, 40, 42), -1)
        cv2.circle(image, (px, py), int(hole_radius) + 1, (175, 178, 180), 1)
        cv2.circle(valid, (px, py), int(hole_radius) + 8, 0, -1)

    points = np.argwhere(valid > 0)
    if label != "normal":
        py, px = random.choice(points)
        if label == "scratch":
            length = random.randint(18, 36)
            defect_angle = random.uniform(0, 2 * np.pi)
            end = (int(px + length * np.cos(defect_angle)), int(py + length * np.sin(defect_angle)))
            cv2.line(image, (int(px), int(py)), end, (35, 38, 40), 2)
            cv2.line(annotation, (int(px), int(py)), end, 255, 3)
        elif label == "dent":
            axes = (random.randint(7, 12), random.randint(4, 8))
            cv2.ellipse(image, (int(px), int(py)), axes, random.uniform(0, 180), 0, 360, (55, 58, 60), -1)
            cv2.ellipse(annotation, (int(px), int(py)), axes, 0, 0, 360, 255, -1)
        else:
            hole = random.randint(6, 10)
            cv2.circle(image, (int(px), int(py)), hole, (20, 23, 25), -1)
            cv2.circle(annotation, (int(px), int(py)), hole + 1, 255, -1)
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), random.uniform(0.2, 0.9))
    image = np.clip(image.astype(np.int16) + np.random.normal(0, random.uniform(1.5, 4), image.shape), 0, 255).astype(np.uint8)
    transform = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), random.uniform(-5, 5), random.uniform(0.96, 1.04))
    return cv2.warpAffine(image, transform, (SIZE, SIZE), borderValue=(30, 30, 30)), cv2.warpAffine(annotation, transform, (SIZE, SIZE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--val", type=int, default=60)
    parser.add_argument("--test", type=int, default=60)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.output.exists():
        shutil.rmtree(args.output)
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        for label in CLASSES:
            image_dir, mask_dir = args.output / split / label, args.output / "masks" / split / label
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image, mask = make_part(label)
                stem = f"{label}_{index:04d}"
                cv2.imwrite(str(image_dir / f"{stem}.jpg"), image)
                cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)
    print(f"Generated dataset at {args.output}")


if __name__ == "__main__":
    main()
