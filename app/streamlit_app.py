
import streamlit as st

st.set_page_config(page_title="Be My Ear", page_icon="🧏", layout="wide")

st.title("🧏 Be My Ear / Be My Voice")
st.caption("Phase 1 app prototype: WLASL2000 sign recognition + Candidate Sentence Mode")

st.markdown("""
## App modes

Use the pages on the left sidebar:

1. **Upload Video**  
   Upload an `.mp4` file and run sign recognition.

2. **Live Camera**  
   Use your webcam for real-time sign recognition.

3. **Screen Capture**  
   Capture a selected screen area such as YouTube, Zoom, Teams, or browser video.

## Required model package

This app expects your deployed WLASL2000 model here:

```text
E:/Be_My_Ear/app/models/ASL/WLASL2000
```

Recommended testing order:

```text
Upload Video → Screen Capture → Live Camera
```
""")
