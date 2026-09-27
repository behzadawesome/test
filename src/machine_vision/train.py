"""Train and persist the inspection model."""
import argparse
from pathlib import Path

from .model import CLASSES, InspectionModel, read_image


def load_split(root: Path, split: str, size: int):
    for label in CLASSES:
        for path in sorted((root / split / label).glob("*.jpg")):
            yield label, read_image(path, size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    model = InspectionModel(args.size).fit(load_split(args.data, "train", args.size))
    model.save(args.output)
    print(f"Saved model to {args.output}; anomaly threshold={model.anomaly_threshold:.3f}")


if __name__ == "__main__":
    main()
