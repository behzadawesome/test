"""Evaluate known-class and operational binary inspection metrics."""
import argparse
from pathlib import Path

from .model import CLASSES, InspectionModel, read_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    model = InspectionModel.load(args.model)
    matrix = {actual: {predicted: 0 for predicted in CLASSES + ("unknown_anomaly", "uncertain")} for actual in CLASSES}
    total_normal = rejected_normal = total_defect = escaped_defect = 0
    for actual in CLASSES:
        for path in sorted((args.data / args.split / actual).glob("*.jpg")):
            decision = model.predict(read_image(path, model.size)).decision
            matrix[actual][decision] += 1
            if actual == "normal":
                total_normal += 1
                rejected_normal += decision != "normal"
            else:
                total_defect += 1
                escaped_defect += decision == "normal"
    print("actual -> decision counts")
    for actual, row in matrix.items():
        print(actual, row)
    print(f"false_reject_rate={rejected_normal / max(total_normal, 1):.4f}")
    print(f"defect_escape_rate={escaped_defect / max(total_defect, 1):.4f}")


if __name__ == "__main__":
    main()
