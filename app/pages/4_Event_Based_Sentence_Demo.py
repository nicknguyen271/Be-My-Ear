
import sys
from pathlib import Path
from collections import deque
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mss
    MSS_AVAILABLE = True
except Exception:
    MSS_AVAILABLE = False

from src.inference.wlasl2000_engine import WLASL2000Engine
from src.inference.event_sentence_logic import multi_window_predict, merge_event_predictions
from src.nlp.candidate_sentence_mode import candidate_sentence_mode

st.set_page_config(page_title="Event Sentence Demo", page_icon="🧠", layout="wide")
st.title("🧠 Event-Based Sentence Demo")
st.caption("Testing mode: short sign events → event-level Top-5 → sentence buffer → NLP sentence")

if not MSS_AVAILABLE:
    st.warning("mss is not installed.")
    st.code("pip install mss", language="powershell")
    st.stop()

@st.cache_resource
def load_engine():
    return WLASL2000Engine(project_root="E:/Be_My_Ear")

engine = load_engine()

st.info(
    "This is a project demo/testing page. It is not the final clean app UI. "
    "The goal is to test whether short sign events can be collected into sentence candidates."
)

# ============================================================
# Screen capture helpers
# ============================================================

def capture_full_monitor(monitor_index=1):
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        screenshot = np.array(sct.grab(monitor))

    frame_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    return frame_bgr, monitor


def select_screen_region_with_mouse(monitor_index=1):
    full_frame, monitor = capture_full_monitor(monitor_index)

    window_name = "Select Screen Region - drag box, ENTER/SPACE confirm, C cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    roi = cv2.selectROI(
        window_name,
        full_frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(window_name)

    x, y, w, h = roi

    if w == 0 or h == 0:
        return None

    return {
        "left": int(monitor["left"] + x),
        "top": int(monitor["top"] + y),
        "width": int(w),
        "height": int(h)
    }


def capture_screen_region(region):
    with mss.mss() as sct:
        screenshot = np.array(sct.grab(region))
    return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)


def summarise_events(events):
    rows = []
    for event in events:
        rows.append({
            "event_id": event["event_id"],
            "time": event["time"],
            "duration": event["duration"],
            "top1": event["top_k"][0]["gloss"],
            "top1_score": round(float(event["top_k"][0]["probability"]), 3),
            "top5": ", ".join([x["gloss"] for x in event["top_k"]])
        })
    return pd.DataFrame(rows)


def summarise_completed_sentences(completed_sentences):
    return pd.DataFrame([
        {
            "sentence_id": s["sentence_id"],
            "sentence": s["sentence"],
            "confidence": s["confidence"],
            "selected_glosses": " → ".join(s["selected_glosses"])
        }
        for s in completed_sentences
    ])


# ============================================================
# Region selection
# ============================================================

with mss.mss() as sct:
    monitors = sct.monitors

st.subheader("1. Select signer/video region")

monitor_options = []
for i, monitor in enumerate(monitors):
    monitor_options.append(f"{i} - {monitor}")

selected_monitor_label = st.selectbox(
    "Monitor",
    monitor_options,
    index=1 if len(monitor_options) > 1 else 0
)

MONITOR_INDEX = int(selected_monitor_label.split(" - ")[0])

col_a, col_b = st.columns(2)

with col_a:
    if st.button("Select region with mouse", type="primary"):
        selected = select_screen_region_with_mouse(MONITOR_INDEX)
        if selected:
            st.session_state["event_screen_region"] = selected
            st.success("Region selected.")
            st.json(selected)
        else:
            st.error("No region selected.")

with col_b:
    if st.button("Use whole monitor"):
        with mss.mss() as sct:
            st.session_state["event_screen_region"] = dict(sct.monitors[MONITOR_INDEX])
        st.success("Whole monitor selected.")
        st.json(st.session_state["event_screen_region"])

if "event_screen_region" not in st.session_state:
    st.session_state["event_screen_region"] = {
        "left": 120,
        "top": 120,
        "width": 640,
        "height": 480
    }

region = st.session_state["event_screen_region"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    region["left"] = int(st.number_input("Left", value=int(region["left"]), step=10))
with col2:
    region["top"] = int(st.number_input("Top", value=int(region["top"]), step=10))
with col3:
    region["width"] = int(st.number_input("Width", value=int(region["width"]), step=10, min_value=50))
with col4:
    region["height"] = int(st.number_input("Height", value=int(region["height"]), step=10, min_value=50))

st.session_state["event_screen_region"] = region

if st.button("Preview region"):
    frame = capture_screen_region(region)
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Selected region preview", use_container_width=False)

# ============================================================
# Settings
# ============================================================

st.subheader("2. Event-based sentence settings")

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    run_seconds = st.slider("Run seconds", 10, 120, 45, 5)
    predict_every = st.slider("Predict every N frames", 3, 15, 5, 1)

with col_s2:
    sign_end_pause = st.slider("Sign-end pause seconds", 0.3, 1.2, 0.6, 0.1)
    sentence_end_pause = st.slider("Sentence-end pause seconds", 1.0, 3.5, 1.8, 0.1)

with col_s3:
    min_event_duration = st.slider("Min event duration", 0.2, 1.0, 0.35, 0.05)
    max_event_duration = st.slider("Max event duration", 1.0, 4.0, 2.5, 0.25)

fast_windows = (30, 45, 60)
top_k = 5
min_event_predictions = 2

show_live = st.checkbox("Show live captured region", value=True)
show_debug = st.checkbox("Show debug tables", value=True)

# ============================================================
# Run demo
# ============================================================

if st.button("Run event-based sentence demo", type="primary"):
    runtime = {
        "state": "WAITING",
        "current_event_predictions": [],
        "current_event_start_time": None,
        "last_active_time": None,
        "sentence_events": [],
        "completed_sentences": [],
        "event_id": 0,
    }

    keypoint_buffer = deque(maxlen=engine.sequence_length)
    frame_counter = 0
    start_time = time.time()

    frame_box = st.empty()
    status_box = st.empty()
    current_sentence_box = st.empty()
    event_table_box = st.empty()
    completed_box = st.empty()

    def commit_current_event(now):
        if runtime["current_event_start_time"] is None:
            runtime["current_event_predictions"] = []
            return None

        duration = now - runtime["current_event_start_time"]

        if duration < min_event_duration:
            runtime["current_event_predictions"] = []
            return None

        if len(runtime["current_event_predictions"]) < min_event_predictions:
            runtime["current_event_predictions"] = []
            return None

        merged_top_k = merge_event_predictions(runtime["current_event_predictions"], top_k=top_k)

        if len(merged_top_k) == 0:
            runtime["current_event_predictions"] = []
            return None

        runtime["event_id"] += 1

        event = {
            "event_id": runtime["event_id"],
            "time": round(runtime["current_event_start_time"] - start_time, 2),
            "duration": round(duration, 2),
            "status": "candidate",
            "top_k": merged_top_k
        }

        runtime["sentence_events"].append(event)
        runtime["current_event_predictions"] = []

        return event

    def commit_sentence_if_ready():
        if len(runtime["sentence_events"]) < 2:
            return None

        result = candidate_sentence_mode(runtime["sentence_events"][-8:], max_candidates_per_event=3)

        sentence_record = {
            "sentence_id": len(runtime["completed_sentences"]) + 1,
            "sentence": result["sentence"],
            "confidence": result["confidence"],
            "selected_glosses": result["selected_glosses"],
            "events": runtime["sentence_events"].copy(),
            "details": result
        }

        runtime["completed_sentences"].append(sentence_record)
        runtime["sentence_events"] = []

        return sentence_record

    with engine.mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while time.time() - start_time < run_seconds:
            now = time.time()

            frame = capture_screen_region(region)
            keypoints = engine.process_frame_to_keypoints(frame, holistic)
            keypoint_buffer.append(keypoints)
            frame_counter += 1

            if len(keypoint_buffer) >= min(fast_windows) and frame_counter % predict_every == 0:
                activity_sequence = np.array(
                    list(keypoint_buffer)[-min(len(keypoint_buffer), engine.sequence_length):],
                    dtype=np.float32
                )
                activity = engine.hand_activity_score(activity_sequence)

                if activity["active"]:
                    runtime["last_active_time"] = now

                    if runtime["state"] == "WAITING":
                        runtime["state"] = "SIGNING"
                        runtime["current_event_start_time"] = now
                        runtime["current_event_predictions"] = []

                    multi_preds = multi_window_predict(
                        engine,
                        keypoint_buffer,
                        fast_windows=fast_windows,
                        top_k=top_k
                    )

                    runtime["current_event_predictions"].extend(multi_preds)

                    if (
                        runtime["current_event_start_time"] is not None
                        and now - runtime["current_event_start_time"] >= max_event_duration
                    ):
                        commit_current_event(now)
                        runtime["current_event_start_time"] = now
                        runtime["current_event_predictions"] = []

                else:
                    if runtime["state"] == "SIGNING" and runtime["last_active_time"] is not None:
                        inactive_time = now - runtime["last_active_time"]

                        if inactive_time >= sign_end_pause:
                            commit_current_event(now)
                            runtime["state"] = "WAITING"
                            runtime["current_event_start_time"] = None

                    if (
                        runtime["last_active_time"] is not None
                        and now - runtime["last_active_time"] >= sentence_end_pause
                    ):
                        if len(runtime["sentence_events"]) >= 2:
                            commit_sentence_if_ready()

                status_box.info(
                    f"State: {runtime['state']} | "
                    f"Event buffer: {len(runtime['sentence_events'])} | "
                    f"Completed sentences: {len(runtime['completed_sentences'])}"
                )

            if len(runtime["sentence_events"]) >= 2:
                live_result = candidate_sentence_mode(runtime["sentence_events"][-8:], max_candidates_per_event=3)
                current_sentence_box.markdown(
                    f"## Current possible sentence: **{live_result['sentence']}**  \n"
                    f"Confidence: **{live_result['confidence']}**  \n"
                    f"Selected signs: `{ ' → '.join(live_result['selected_glosses']) }`"
                )
            else:
                current_sentence_box.markdown("## Current possible sentence: _waiting for more sign events..._")

            if show_live:
                frame_box.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption="Live captured region",
                    use_container_width=False
                )

            if show_debug and len(runtime["sentence_events"]) > 0:
                event_table_box.dataframe(summarise_events(runtime["sentence_events"]), use_container_width=True)

            if runtime["completed_sentences"]:
                completed_box.dataframe(
                    summarise_completed_sentences(runtime["completed_sentences"]),
                    use_container_width=True
                )

    # Final commit at end
    now = time.time()

    if len(runtime["current_event_predictions"]) >= min_event_predictions:
        commit_current_event(now)

    if len(runtime["sentence_events"]) >= 2:
        commit_sentence_if_ready()

    st.success("Event-based sentence demo finished.")

    if runtime["completed_sentences"]:
        st.subheader("Completed sentence transcript")
        st.dataframe(
            summarise_completed_sentences(runtime["completed_sentences"]),
            use_container_width=True
        )

        with st.expander("Full sentence details"):
            st.json(runtime["completed_sentences"])
    else:
        st.warning(
            "No completed sentence detected. Try a tighter crop, longer run time, clearer hand visibility, "
            "or reduce Sentence-end pause seconds."
        )
