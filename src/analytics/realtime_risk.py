from collections import deque
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.risk_predictor import CrowdRiskPredictor


class RealtimeRiskEngine:

    def __init__(self, window_seconds=5):
        self.window_seconds = window_seconds
        self.history = deque()

        self.predictor = CrowdRiskPredictor()

    def add_observation(
        self,
        timestamp,
        total_people,
        zone_a,
        zone_b,
        zone_c,
        zone_d,
        avg_movement
    ):

        observation = {
            "timestamp": timestamp,
            "total_people": total_people,
            "zone_a": zone_a,
            "zone_b": zone_b,
            "zone_c": zone_c,
            "zone_d": zone_d,
            "avg_movement": avg_movement
        }

        self.history.append(observation)

        # Keep only recent window
        cutoff = timestamp.timestamp() - self.window_seconds

        while self.history:
            oldest_time = self.history[0]["timestamp"].timestamp()

            if oldest_time < cutoff:
                self.history.popleft()
            else:
                break

        return self._calculate_risk()

    def _calculate_risk(self):

        if len(self.history) < 2:
            return {
                "risk_label": "WAITING",
                "message": "Collecting crowd data..."
            }

        df = pd.DataFrame(self.history)

        total_people = df["total_people"]

        features = {
            "total_people_mean": total_people.mean(),
            "total_people_max": total_people.max(),
            "total_people_min": total_people.min(),

            "zone_a_mean": df["zone_a"].mean(),
            "zone_b_mean": df["zone_b"].mean(),
            "zone_c_mean": df["zone_c"].mean(),
            "zone_d_mean": df["zone_d"].mean(),

            "avg_movement_mean": df["avg_movement"].mean(),

            "people_change": (
                total_people.iloc[-1] - total_people.iloc[0]
            ),

            "rolling_people_mean": total_people.mean(),

            "observations": len(df)
        }

        duration = (
            df["timestamp"].iloc[-1]
            - df["timestamp"].iloc[0]
        ).total_seconds()

        if duration > 0:
            features["people_change_per_minute"] = (
                features["people_change"] / duration * 60
            )
        else:
            features["people_change_per_minute"] = 0.0

        prediction = self.predictor.predict(
            pd.DataFrame([features])
        )

        result = prediction.iloc[0].to_dict()

        result["features"] = features

        return result