"""Business logic of a prediction: preprocessing, calibration, decision and review flag.

* P(defective) = softmax(logits / temperature)[defective]   (temperature fit on validation)
* predicted_class = defective if P(defective) >= threshold  (threshold chosen on validation)
* needs_review = review_band.low <= P(defective) < review_band.high
* confidence = probability of the predicted class

Preprocessing is the same ``core.preprocessing.preprocess`` used to build training tensors,
so the model sees identical inputs in both places.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image

from defect_detection.core.constants import CLASS_NAMES, DEFECTIVE_LABEL
from defect_detection.core.model_meta import ModelMeta
from defect_detection.core.preprocessing import PreprocessSpec, preprocess
from defect_detection.serving.inference.engine import Predictor


@dataclass(frozen=True)
class Prediction:
    """One image's outcome."""

    predicted_class: str
    confidence: float
    defect_probability: float
    threshold: float
    needs_review: bool
    model_version: str


def defect_probabilities(
    logits: npt.NDArray[np.float32], temperature: float
) -> npt.NDArray[np.float64]:
    """Numerically stable softmax over the 2 logits after temperature scaling."""
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    probs: npt.NDArray[np.float64] = (exp / exp.sum(axis=1, keepdims=True))[:, DEFECTIVE_LABEL]
    return probs


class PredictionService:
    """Turns decoded images into Predictions using a Predictor and the model metadata."""

    def __init__(self, predictor: Predictor, meta: ModelMeta) -> None:
        self.predictor = predictor
        self.meta = meta
        self.spec = PreprocessSpec(meta.input.image_size, meta.input.mean, meta.input.std)

    def predict(self, images: list[Image.Image]) -> list[Prediction]:
        """Predict a batch of already-validated images."""
        batch = np.stack([preprocess(img, self.spec) for img in images])
        probs = defect_probabilities(self.predictor.predict_logits(batch), self.meta.temperature)
        band = self.meta.review_band
        results = []
        for p in probs.tolist():
            defective = p >= self.meta.threshold
            results.append(
                Prediction(
                    predicted_class=CLASS_NAMES[
                        DEFECTIVE_LABEL if defective else 1 - DEFECTIVE_LABEL
                    ],
                    confidence=p if defective else 1.0 - p,
                    defect_probability=p,
                    threshold=self.meta.threshold,
                    needs_review=band.low <= p < band.high,
                    model_version=self.meta.model_version,
                )
            )
        return results
