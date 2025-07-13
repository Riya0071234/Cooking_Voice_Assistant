# pages/2_Vision_Analyzer.py
"""
A UI page for real-time visual analysis using the device's camera.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/vision/analyze"

st.set_page_config(page_title="Vision Analyzer", page_icon="👁️")
st.title("👁️ Live Vision Analyzer")
st.markdown("Show me an ingredient or your cooking progress!")

# --- Camera Input ---
img_file_buffer = st.camera_input("Point your camera at an ingredient or your pan")

if img_file_buffer is not None:
    # To send the image to the API, we need to get its bytes.
    bytes_data = img_file_buffer.getvalue()

    st.info("Analyzing image...")

    try:
        # The file needs to be sent as multipart/form-data
        files = {'file': (img_file_buffer.name, bytes_data, img_file_buffer.type)}
        response = requests.post(API_URL, files=files, timeout=60)
        response.raise_for_status()

        results = response.json()
        detections = results.get("detections", [])

        if not detections:
            st.warning("I couldn't identify any specific items in the image.")
        else:
            st.success("Here's what I see:")
            for item in detections:
                label = item.get('label').replace("_", " ").title()
                confidence = item.get('confidence', 0)
                st.markdown(f"- **{label}** (Confidence: {confidence:.2f})")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the vision analysis service. Please ensure the backend is running. Error: {e}")