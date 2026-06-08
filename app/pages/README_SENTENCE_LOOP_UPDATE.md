# Screen Capture Sentence Loop Update

Replace this file in your project:

`E:\Be_My_Ear\app\pages\3_Screen_Capture.py`

with the new file from this zip:

`app/pages/3_Screen_Capture.py`

## What changed

The screen capture page now uses a sentence loop:

- Keeps a candidate event buffer while inference is running
- Adds useful sign events over time
- Avoids repeated duplicate events too close together
- Continuously updates Candidate Sentence Mode
- Shows current possible sentence during the loop
- Shows final sentence at the end

## Why this helps

Before, the app gave one short phrase from only one or two predictions.

Now, it can collect several detected sign events and build a sentence from the recent event sequence.

## How to use

1. Run Streamlit.
2. Go to Screen Capture.
3. Select the signer/video region.
4. Set run duration to 30–60 seconds for sentence testing.
5. Click Run sentence loop inference.
6. Watch the Current possible sentence update.