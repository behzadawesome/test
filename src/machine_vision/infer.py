"""Run one inspection and print a machine-readable decision."""
import argparse
import json
from pathlib import Path

from .model import InspectionModel, read_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    model = InspectionModel.load(args.model)
    prediction = model.predict(read_image(args.image, model.size))
    print(json.dumps({"decision": prediction.decision, "known_class": prediction.known_class,
                      "class_confidence": round(prediction.class_confidence, 4),
                      "anomaly_score": round(prediction.anomaly_score, 4),
                      "anomaly_threshold": round(prediction.anomaly_threshold, 4)}))


if __name__ == "__main__":
    main()
