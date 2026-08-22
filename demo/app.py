"""
demo/app.py — Streamlit demo: upload a noisy .wav, hear the enhanced result.

Run with:
    streamlit run demo/app.py
"""

import os
import sys
import tempfile

import streamlit as st

# allow `from src...` imports when running via `streamlit run demo/app.py`
# from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import enhance_file

st.set_page_config(page_title="AI Speech Enhancement Demo")
st.title("AI Speech Enhancement Demo")
st.caption("U-Net mask predictor trained on VoiceBank+DEMAND. Upload a noisy .wav file.")

checkpoint_path = "checkpoints/unet_se.pt"
if not os.path.exists(checkpoint_path):
    st.error(
        f"No trained model found at `{checkpoint_path}`. Run `python -m src.train` "
        f"first, then restart this demo."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload a noisy audio file (.wav)", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(uploaded_file.read())
        tmp_in_path = tmp_in.name

    st.subheader("Noisy Input")
    st.audio(tmp_in_path)

    output_path = os.path.join(tempfile.gettempdir(), "enhanced_output.wav")

    with st.spinner("Enhancing..."):
        try:
            enhance_file(tmp_in_path, output_path, checkpoint=checkpoint_path)
        except Exception as e:
            st.error(f"Enhancement failed: {e}")
            st.stop()

    st.subheader("Enhanced Output")
    st.audio(output_path)

    with open(output_path, "rb") as f:
        st.download_button("Download enhanced audio", f, file_name="enhanced.wav")