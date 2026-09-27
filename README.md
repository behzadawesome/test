# Machine Vision Defect Prototype

A reproducible, inspection-oriented prototype for circular machined parts.  It combines two complementary decisions:

1. **Known-defect classification** for `scratch`, `dent`, and `hole`, using a compact nearest-centroid feature model.
2. **One-class anomaly detection** trained only on conforming (`normal`) parts, so unfamiliar surface anomalies can be rejected instead of being forced into a known class.

The project includes a synthetic-data generator, train/evaluate CLI, persisted model artifacts, and a JSON inference interface. It is a prototype—not a replacement for qualification using production cameras, lighting, part fixtures, and labeled factory data.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

python -m machine_vision.generate --output data/synthetic --seed 42
python -m machine_vision.train --data data/synthetic --output artifacts/model.npz
python -m machine_vision.evaluate --data data/synthetic --model artifacts/model.npz
python -m machine_vision.infer --model artifacts/model.npz \
  --image data/synthetic/test/scratch/scratch_0000.jpg
```

## Dataset layout

```
data/synthetic/
  train/{normal,scratch,dent,hole}/*.jpg
  val/{normal,scratch,dent,hole}/*.jpg
  test/{normal,scratch,dent,hole}/*.jpg
```

The generator saves a same-named PNG annotation mask for each non-normal image in `masks/<split>/<class>/`. These masks are intentionally excluded from model inputs and are available to measure future segmentation/localization models.

## Decision policy

`infer` returns `normal`, a known defect class, `unknown_anomaly`, or `uncertain`.

* A high normal-template residual produces `unknown_anomaly` when the classifier is not confident in a known class.
* A confident known class is returned even if the anomaly detector is high.
* Low classifier confidence without a material anomaly becomes `uncertain` for manual review.

Tune the anomaly and confidence quantiles on held-out production validation data before deployment. Keep acquisition geometry fixed: centering, scale, illumination and exposure must match the normal training images.

## Evaluation

The evaluator reports a known-class confusion matrix plus binary defect escape and normal false-reject rates. The latter are the operational measures to track on a separately collected, production-representative test set.

## Safety and production notes

* Do not train or approve a release solely on these synthetic images.
* Store images, decisions, scores, camera parameters, and model version for traceability.
* Define an escalation route for `uncertain` and `unknown_anomaly` parts; never silently pass them.
* Revalidate after changes to part revision, lens, lighting, camera firmware, or fixture.
