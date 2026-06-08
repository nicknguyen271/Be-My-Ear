
import streamlit as st

st.set_page_config(page_title="Live Camera", page_icon="📷", layout="wide")
st.title("📷 Live Camera Inference")

st.markdown("""
Live camera in Streamlit needs `streamlit-webrtc`, which can be more sensitive than the notebook OpenCV camera.

For now, use your working notebook for highest reliability:

```text
11_wlasl2000_webcam_realtime_inference.ipynb
```

To enable browser camera mode, install:

```powershell
pip install streamlit-webrtc av
```

The app already supports:
- Upload Video
- Screen Capture

The next version can add full WebRTC camera processing once the upload/screen modes are stable.
""")
