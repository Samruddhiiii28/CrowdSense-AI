
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

LIVE_FILE = DATA_DIR / "live_crowd_data.csv"
PREDICTIONS_FILE = DATA_DIR / "day21_live_predictions.csv"
ALERTS_FILE = DATA_DIR / "risk_alerts.csv"

st.set_page_config(
    page_title="CrowdSense AI",
    page_icon="👥",
    layout="wide",
)

st.title("CrowdSense AI")
st.caption(
    "Predictive Crowd Intelligence & Safety Monitoring"
)

st.sidebar.header("Dashboard Controls")
refresh = st.sidebar.button("Refresh data")

st.sidebar.info(
    "This dashboard displays recorded observations and model "
    "predictions. It is a prototype, not a certified safety system."
)


def load_csv(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


live_df = load_csv(LIVE_FILE)
pred_df = load_csv(PREDICTIONS_FILE)
alerts_df = load_csv(ALERTS_FILE)

if live_df.empty:
    st.warning(
        "No live camera records found. Run your existing "
        "Day 16 logging pipeline to generate observations."
    )
else:
    live_df = live_df.copy()

    numeric_columns = [
        "total_people",
        "zone_a",
        "zone_b",
        "zone_c",
        "zone_d",
        "avg_movement",
        "entries",
        "exits",
    ]

    for column in numeric_columns:
        if column in live_df.columns:
            live_df[column] = pd.to_numeric(
                live_df[column], errors="coerce"
            )

    latest = live_df.iloc[-1]

    total_people = (
        latest.get("total_people", 0)
    )
    movement = latest.get("avg_movement", 0)
    entries = latest.get("entries", 0)
    exits = latest.get("exits", 0)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Detected People",
        int(total_people) if pd.notna(total_people) else "N/A",
    )
    c2.metric(
        "Movement (pixels/frame)",
        f"{movement:.2f}" if pd.notna(movement) else "N/A",
    )
    c3.metric(
        "Entries",
        int(entries) if pd.notna(entries) else "N/A",
    )
    c4.metric(
        "Exits",
        int(exits) if pd.notna(exits) else "N/A",
    )

    st.subheader("Zone-wise Crowd Count")

    zone_columns = [
        col for col in ["zone_a", "zone_b", "zone_c", "zone_d"]
        if col in live_df.columns
    ]

    if zone_columns:
        zone_values = pd.DataFrame({
            "Zone": zone_columns,
            "People": [
                latest[col] if pd.notna(latest[col]) else 0
                for col in zone_columns
            ],
        }).set_index("Zone")

        st.bar_chart(zone_values)
    else:
        st.info("Zone count columns are unavailable.")

    st.subheader("Recent Camera Observations")

    display_columns = [
        col for col in [
            "timestamp",
            "total_people",
            "zone_a",
            "zone_b",
            "zone_c",
            "zone_d",
            "avg_movement",
            "entries",
            "exits",
        ]
        if col in live_df.columns
    ]

    st.dataframe(
        live_df[display_columns].tail(20),
        width="stretch",
        hide_index=True,
    )

st.divider()
st.subheader("Risk Predictions")

if pred_df.empty:
    st.info(
        "No historical prediction file found. Run the existing "
        "Day 21 prediction script to generate it."
    )
else:
    possible_label_columns = [
        "predicted_risk",
        "predicted_risk_label",
        "risk_prediction",
        "risk_label",
        "prediction",
    ]

    label_col = next(
        (
            col for col in possible_label_columns
            if col in pred_df.columns
        ),
        None,
    )

    if label_col is None:
        st.warning(
            "Could not identify the risk label column. "
            f"Available columns: {list(pred_df.columns)}"
        )
    else:
        labels = (
            pred_df[label_col]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        distribution = labels.value_counts().rename_axis(
            "Predicted Risk"
        ).to_frame("Windows")

        st.caption(
            "Historical model predictions—not verified safety labels."
        )
        st.bar_chart(distribution)

        latest_label = labels.dropna()

        if not latest_label.empty:
            st.metric(
                "Last Recorded Prediction",
                latest_label.iloc[-1],
            )

st.divider()
st.subheader("Alert History")

if alerts_df.empty:
    st.info(
        "No alert events found yet. Alert history appears after "
        "the Day 28 AlertManager logs events."
    )
else:
    st.dataframe(
        alerts_df.tail(20),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "CrowdSense AI | Predictions may be inaccurate and require "
    "independent validation before operational use."
)
