from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analytics.live_risk_pipeline import (
    LiveRiskPipeline
)

def main():

    pipeline = LiveRiskPipeline(
        camera_index=0
    )

    pipeline.start()


if __name__ == "__main__":
    main()