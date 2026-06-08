
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.wlasl2000_engine import WLASL2000Engine
from src.nlp.candidate_sentence_mode import candidate_sentence_mode

st.set_page_config(page_title="Upload Video", page_icon="🎥", layout="wide")
st.title("🎥 Upload Video Inference")

@st.cache_resource
def load_engine():
    return WLASL2000Engine(project_root="E:/Be_My_Ear")

with st.spinner("Loading WLASL2000 model..."):
    engine = load_engine()

st.success(f"Model loaded on: {engine.device}")

uploaded_file = st.file_uploader("Upload a sign video", type=["mp4", "mov", "avi"])
stride = st.slider("Sliding window stride", 5, 30, 15, 5)
top_k = st.slider("Top-K candidates", 3, 10, 5, 1)

if uploaded_file is not None:
    st.video(uploaded_file)

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        video_path = Path(tmp.name)

    if st.button("Run inference", type="primary"):
        with st.spinner("Running video inference..."):
            result = engine.run_video_inference(video_path, stride=stride, top_k=top_k)

        rows = result["rows"]
        events = result["candidate_events"]

        st.subheader("Prediction windows")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Candidate Sentence Mode")
        if events:
            sentence_result = candidate_sentence_mode(events[-6:], max_candidates_per_event=3)
            st.markdown(f"### 🧠 Possible meaning: **{sentence_result['sentence']}**")
            st.write("Confidence:", sentence_result["confidence"])
            st.write("Selected glosses:", " → ".join(sentence_result["selected_glosses"]))

            with st.expander("Sentence alternatives"):
                st.json(sentence_result["alternatives"])

            with st.expander("Raw candidate events"):
                st.json(events)
        else:
            st.warning("No useful sign events detected. Try a clearer video with both hands visible.")
