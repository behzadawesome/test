import numpy as np

from machine_vision.model import InspectionModel


def test_model_detects_strong_unseen_anomaly():
    rng = np.random.default_rng(7)
    normal = [np.full((64, 64), 0.5, dtype=np.float32) + rng.normal(0, 0.002, (64, 64)) for _ in range(8)]
    samples = [("normal", image) for image in normal]
    samples += [(label, image + offset) for label, offset in (("scratch", 0.02), ("dent", -0.02), ("hole", 0.04)) for image in normal[:3]]
    model = InspectionModel(size=64, confidence_threshold=0.95).fit(samples)
    abnormal = normal[0].copy()
    abnormal[25:35, 25:35] = 0.0
    prediction = model.predict(abnormal)
    assert prediction.anomaly_score > prediction.anomaly_threshold
    assert prediction.decision == "unknown_anomaly"
