import time
from datetime import datetime
from pathlib import Path
import sys

# Add CrowdSense project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.realtime_risk import RealtimeRiskEngine


def main():

    print("=" * 50)
    print("CrowdSense AI - Day 13 Real-Time Risk Test")
    print("=" * 50)

    engine = RealtimeRiskEngine(
        window_seconds=5
    )

    print("\nCollecting simulated live observations...\n")

    # Simulated observations.
    # Later these values will come directly from
    # YOLO + ByteTrack + crowd analytics.

    observations = [
        (3, 1, 1, 0, 1, 0.5),
        (4, 1, 1, 1, 1, 0.6),
        (5, 2, 1, 1, 1, 0.8),
        (7, 3, 1, 2, 1, 1.0),
        (9, 4, 2, 2, 1, 1.2),
        (11, 5, 2, 2, 2, 1.5),
    ]

    for values in observations:

        (
            total_people,
            zone_a,
            zone_b,
            zone_c,
            zone_d,
            movement,
        ) = values

        result = engine.add_observation(
            timestamp=datetime.now(),
            total_people=total_people,
            zone_a=zone_a,
            zone_b=zone_b,
            zone_c=zone_c,
            zone_d=zone_d,
            avg_movement=movement,
        )

        print(
            f"People: {total_people:2d} | "
            f"Risk: {result['risk_label']}"
        )

        time.sleep(1)

    print("\nTest completed.")


if __name__ == "__main__":
    main()