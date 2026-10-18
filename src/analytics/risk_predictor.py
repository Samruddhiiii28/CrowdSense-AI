
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = PROJECT_ROOT / "models" / "crowd_risk_model.joblib"


class CrowdRiskPredictor:
    """Loads the trained model and predicts prototype crowd-risk labels."""

    def __init__(self, model_path=MODEL_FILE):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}. "
                "Complete Day 11 first."
            )

        artifact = joblib.load(model_path)

        self.model = artifact["model"]
        self.features = artifact["features"]
        self.labels = artifact["labels"]

    def predict(self, feature_data):
        """
        feature_data may be a dictionary of features or a DataFrame.
        Returns one prediction per input row.
        """
        if isinstance(feature_data, dict):
            data = pd.DataFrame([feature_data])
        else:
            data = pd.DataFrame(feature_data).copy()

        missing = [
            name for name in self.features
            if name not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing model features: {missing}"
            )

        # Keep the exact training feature order.
        data = data[self.features].copy()

        for column in self.features:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

        if data.isna().all(axis=1).any():
            raise ValueError(
                "At least one row contains no usable model features."
            )

        predicted_labels = self.model.predict(data)

        result = pd.DataFrame({
            "risk_label": predicted_labels
        })

        # Probability estimates are model scores, not calibrated
        # probabilities of a real-world crowd emergency.
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(data)
            classifier = self.model.named_steps["classifier"]

            for index, label in enumerate(classifier.classes_):
                result[f"score_{label.lower()}"] = probabilities[:, index]

        return result
