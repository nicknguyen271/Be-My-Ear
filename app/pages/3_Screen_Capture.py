
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
from src.nlp.candidate_sentence_mode import candidate_sentence_mode

st.set_page_config(page_title="Screen Capture", page_icon="🖥️", layout="wide")
st.title("🖥️ Screen Capture Inference — Sentence Loop Mode")

if not MSS_AVAILABLE:
    st.warning("mss is not installed.")
    st.code("pip install mss", language="powershell")
    st.stop()

@st.cache_resource
def load_engine():
    return WLASL2000Engine(project_root="E:/Be_My_Ear")

engine = load_engine()

st.markdown("""
This version keeps a **sentence buffer** while screen inference is running.

Instead of only producing one short phrase at the end, it repeatedly collects useful sign events and updates a possible sentence during the loop.
""")

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
        shot = np.array(sct.grab(region))

    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


# ============================================================
# Sentence/event buffer helpers
# ============================================================

def should_add_candidate_event(prediction, decision, candidate_events, current_time, min_gap_seconds=0.8):
    """
    Adds a prediction into the sentence buffer only when it is useful.

    This avoids:
    - adding the same word again and again
    - adding very weak random predictions
    - filling the sentence buffer with noise
    """

    if prediction is None or decision is None:
        return False, "No prediction"

    if decision["status"] not in ["accepted", "uncertain"]:
        return False, "Decision status not useful"

    top1 = prediction["top1_gloss"]
    confidence = float(prediction["top1_confidence"])

    # Avoid extremely weak candidate events
    if confidence < 0.08:
        return False, "Confidence too low"

    if len(candidate_events) == 0:
        return True, "First event"

    last_event = candidate_events[-1]
    last_top1 = last_event["top_k"][0]["gloss"]
    last_time = float(last_event["time"])

    # Avoid duplicate same sign too close together
    if top1 == last_top1 and current_time - last_time < min_gap_seconds:
        return False, "Duplicate too close"

    return True, "Useful event"


def make_event_from_prediction(prediction, decision, event_id, current_time):
    return {
        "event_id": int(event_id),
        "time": float(round(current_time, 2)),
        "status": "locked" if decision["status"] == "accepted" else "uncertain",
        "top_k": prediction["top_k"]
    }


def summarise_events_for_table(events):
    rows = []

    for event in events:
        top5 = ", ".join([item["gloss"] for item in event["top_k"]])
        rows.append({
            "event_id": event["event_id"],
            "time": event["time"],
            "status": event["status"],
            "top1": event["top_k"][0]["gloss"],
            "top1_prob": round(float(event["top_k"][0]["probability"]), 3),
            "top5": top5
        })

    return pd.DataFrame(rows)


# ============================================================
# Region selection UI
# ============================================================

with mss.mss() as sct:
    monitors = sct.monitors

st.subheader("1. Choose monitor")

monitor_options = []
for i, monitor in enumerate(monitors):
    if i == 0:
        label = f"{i} - All monitors combined: {monitor}"
    else:
        label = f"{i} - Monitor {i}: {monitor}"
    monitor_options.append(label)

selected_monitor_label = st.selectbox(
    "Select monitor to capture from",
    monitor_options,
    index=1 if len(monitor_options) > 1 else 0
)

MONITOR_INDEX = int(selected_monitor_label.split(" - ")[0])

st.subheader("2. Select screen region")

col_a, col_b = st.columns([1, 1])

with col_a:
    if st.button("Select region with mouse", type="primary"):
        st.info("An OpenCV window will open. Drag around the signer/video tile, then press ENTER or SPACE.")

        selected_region = select_screen_region_with_mouse(monitor_index=MONITOR_INDEX)

        if selected_region is None:
            st.error("No region selected. Click the button again and drag a box around the signer/video.")
        else:
            st.session_state["screen_region"] = selected_region
            st.success("Region selected successfully.")
            st.json(selected_region)

with col_b:
    if st.button("Use whole selected monitor"):
        with mss.mss() as sct:
            st.session_state["screen_region"] = dict(sct.monitors[MONITOR_INDEX])
        st.success("Whole monitor selected.")
        st.json(st.session_state["screen_region"])

st.subheader("3. Review or manually adjust region")

if "screen_region" not in st.session_state:
    st.session_state["screen_region"] = {
        "left": 120,
        "top": 120,
        "width": 640,
        "height": 480
    }

current = st.session_state["screen_region"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    left = st.number_input("Left", value=int(current["left"]), step=10)
with col2:
    top = st.number_input("Top", value=int(current["top"]), step=10)
with col3:
    width = st.number_input("Width", value=int(current["width"]), step=10, min_value=50)
with col4:
    height = st.number_input("Height", value=int(current["height"]), step=10, min_value=50)

region = {
    "left": int(left),
    "top": int(top),
    "width": int(width),
    "height": int(height)
}

st.session_state["screen_region"] = region

st.write("Current selected region:")
st.json(region)

if st.button("Preview selected region"):
    frame = capture_screen_region(region)
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Selected region preview", use_container_width=False)


# ============================================================
# Inference settings
# ============================================================

st.subheader("4. Sentence loop settings")

run_seconds = st.slider("Run duration in seconds", 5, 120, 30, 5)
predict_every = st.slider("Predict every N frames", 5, 20, 8, 1)
sentence_window_size = st.slider("How many recent sign events to use for sentence", 2, 10, 6, 1)
min_gap_seconds = st.slider("Minimum seconds between repeated same event", 0.3, 2.0, 0.8, 0.1)
show_live_preview = st.checkbox("Show live captured region while running", value=True)
show_debug_table = st.checkbox("Show prediction debug table", value=True)

st.caption("""
For a sentence, run longer than a single sign. The app needs several detected sign events before the NLP output becomes meaningful.
""")


# ============================================================
# Run inference
# ============================================================

if st.button("Run sentence loop inference", type="primary"):
    keypoint_buffer = deque(maxlen=engine.sequence_length)
    recent = deque(maxlen=int(engine.rules.get("stability_window_count", 3)))

    candidate_events = []
    prediction_rows = []

    start = time.time()
    frame_counter = 0

    frame_box = st.empty()
    status_box = st.empty()
    sentence_box = st.empty()
    event_table_box = st.empty()
    prediction_table_box = st.empty()
    final_box = st.empty()

    current_sentence_result = None

    with engine.mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while time.time() - start < run_seconds:
            elapsed = time.time() - start

            frame = capture_screen_region(region)

            kp = engine.process_frame_to_keypoints(frame, holistic)
            keypoint_buffer.append(kp)
            frame_counter += 1

            if len(keypoint_buffer) == engine.sequence_length and frame_counter % predict_every == 0:
                seq = np.array(keypoint_buffer, dtype=np.float32)
                activity = engine.hand_activity_score(seq)

                if activity["active"]:
                    pred = engine.predict_keypoint_sequence(seq, top_k=5)
                    recent.append(pred)
                    decision = engine.decision_from_prediction(pred, recent)

                    top5_text = ", ".join([x["gloss"] for x in pred["top_k"]])

                    prediction_rows.append({
                        "time": round(elapsed, 2),
                        "status": decision["status"],
                        "top1": pred["top1_gloss"],
                        "confidence": round(pred["top1_confidence"], 3),
                        "margin": round(pred["top1_top2_margin"], 3),
                        "top5": top5_text
                    })

                    add_event, add_reason = should_add_candidate_event(
                        pred,
                        decision,
                        candidate_events,
                        current_time=elapsed,
                        min_gap_seconds=min_gap_seconds
                    )

                    if add_event:
                        event = make_event_from_prediction(
                            pred,
                            decision,
                            event_id=len(candidate_events) + 1,
                            current_time=elapsed
                        )
                        candidate_events.append(event)

                    status_box.info(
                        f"Latest: {decision['status'].upper()} | "
                        f"Top-1: {pred['top1_gloss']} ({pred['top1_confidence']:.2f}) | "
                        f"Event add: {add_reason}"
                    )

                else:
                    prediction_rows.append({
                        "time": round(elapsed, 2),
                        "status": "waiting",
                        "top1": "",
                        "confidence": 0.0,
                        "margin": 0.0,
                        "top5": ""
                    })

                    status_box.warning(
                        f"Waiting for sign... Buffer {len(keypoint_buffer)}/{engine.sequence_length}"
                    )

                # Update sentence continuously when enough events exist
                if len(candidate_events) >= 2:
                    recent_events_for_sentence = candidate_events[-sentence_window_size:]
                    current_sentence_result = candidate_sentence_mode(
                        recent_events_for_sentence,
                        max_candidates_per_event=3
                    )

                    sentence_box.markdown(
                        f"## 🧠 Current possible sentence: **{current_sentence_result['sentence']}**  \n"
                        f"Confidence: **{current_sentence_result['confidence']}**  \n"
                        f"Selected signs: `{ ' → '.join(current_sentence_result['selected_glosses']) }`"
                    )
                else:
                    sentence_box.markdown(
                        "## 🧠 Current possible sentence: _waiting for more sign events..._"
                    )

            if show_live_preview:
                frame_box.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption="Live captured region",
                    use_container_width=False
                )

            if len(candidate_events) > 0:
                event_table_box.subheader("Sentence event buffer")
                event_table_box.dataframe(
                    summarise_events_for_table(candidate_events[-sentence_window_size:]),
                    use_container_width=True
                )

            if show_debug_table and len(prediction_rows) > 0:
                prediction_table_box.subheader("Recent prediction windows")
                prediction_table_box.dataframe(
                    pd.DataFrame(prediction_rows).tail(10),
                    use_container_width=True
                )

    st.success("Sentence loop inference finished.")

    if len(candidate_events) >= 2:
        final_result = candidate_sentence_mode(
            candidate_events[-sentence_window_size:],
            max_candidates_per_event=3
        )

        final_box.subheader("Final Candidate Sentence Mode Output")
        final_box.markdown(
            f"## Final sentence: **{final_result['sentence']}**  \n"
            f"Confidence: **{final_result['confidence']}**"
        )

        with st.expander("Final details"):
            st.json(final_result)

        with st.expander("All sentence events"):
            st.dataframe(summarise_events_for_table(candidate_events), use_container_width=True)
    else:
        st.warning(
            "Not enough sign events for a sentence. Try running longer, selecting a tighter signer crop, "
            "or using a clearer video with multiple signs."
        )
