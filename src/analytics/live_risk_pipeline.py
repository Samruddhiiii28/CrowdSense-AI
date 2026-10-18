from datetime import datetime
from pathlib import Path
import sys
import time

import cv2

# --------------------------------------------------
# Make CrowdSense project root available
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analytics.realtime_risk import RealtimeRiskEngine


class LiveRiskPipeline:

    def __init__(self, camera_index=0):

        self.camera_index = camera_index

        self.risk_engine = RealtimeRiskEngine(
            window_seconds=5
        )

        self.last_prediction = "WAITING"

        self.last_prediction_time = 0

        self.prediction_interval = 1.0

    def start(self):

        print("=" * 60)
        print("CrowdSense AI - Live Risk Pipeline")
        print("=" * 60)

        print("\nOpening camera...")

        cap = cv2.VideoCapture(
            self.camera_index
        )

        if not cap.isOpened():

            raise RuntimeError(
                "Could not open camera.\n"
                "Check your webcam or camera index."
            )

        print("Camera opened successfully.")
        print("Press Q to stop.\n")

        while True:

            success, frame = cap.read()

            if not success:

                print(
                    "Could not read frame from camera."
                )

                break

            # --------------------------------------------------
            # TEMPORARY TEST VALUES
            # --------------------------------------------------
            #
            # These values will be replaced by the actual
            # YOLO + ByteTrack analytics in the next step.
            #
            total_people = 0

            zone_a = 0
            zone_b = 0
            zone_c = 0
            zone_d = 0

            avg_movement = 0.0

            # --------------------------------------------------
            # Risk prediction
            # --------------------------------------------------

            current_time = time.time()

            if (
                current_time
                - self.last_prediction_time
                >= self.prediction_interval
            ):

                result = self.risk_engine.add_observation(
                    timestamp=datetime.now(),
                    total_people=total_people,
                    zone_a=zone_a,
                    zone_b=zone_b,
                    zone_c=zone_c,
                    zone_d=zone_d,
                    avg_movement=avg_movement,
                )

                self.last_prediction = result[
                    "risk_label"
                ]

                self.last_prediction_time = (
                    current_time
                )

            # --------------------------------------------------
            # Display
            # --------------------------------------------------

            cv2.putText(
                frame,
                "CrowdSense AI",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Risk: {self.last_prediction}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "YOLO/ByteTrack integration pending",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow(
                "CrowdSense AI - Live Risk",
                frame
            )

            # Q = quit
            if (
                cv2.waitKey(1)
                & 0xFF
                == ord("q")
            ):
                break

        cap.release()

        cv2.destroyAllWindows()


def main():

    pipeline = LiveRiskPipeline(
        camera_index=0
    )

    pipeline.start()


if __name__ == "__main__":
    main()