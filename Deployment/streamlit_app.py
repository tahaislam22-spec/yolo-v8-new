
import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import tempfile

# --- Page Configuration ---
st.set_page_config(page_title="YOLOv8 Vehicle Detection", layout="centered")
st.title("🚗 YOLOv8 Vehicle Detection")
st.write("Upload a video to detect vehicles using a pre-trained YOLOv8 model.")

# --- Model Loading ---
@st.cache_resource
def load_yolo_model():
    model_path = 'best.pt'
    if os.path.exists(model_path):
        model = YOLO(model_path)
        st.success("Loaded custom trained YOLOv8 model: `best.pt`")
    else:
        model = YOLO('yolov8n.pt') # Load default nano model
        st.warning("Custom model `best.pt` not found. Loading `yolov8n.pt` instead.")
    return model

model = load_yolo_model()

# Class names for COCO dataset (relevant for vehicles)
CLASS_NAMES = model.names
VEHICLE_CLASSES = [2, 3, 5, 7] # 2: car, 3: motorcycle, 5: bus, 7: truck

# --- Streamlit UI ---

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file, format="video/mp4", start_time=0)

    with st.spinner("Processing video... This might take a moment."):
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Process video with YOLOv8
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            tfile.close()
            os.unlink(tfile.name)
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use a temporary file for the output video as well
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.write("Starting object detection...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 detection with confidence threshold and class filtering
            results = model(frame, conf=0.50, classes=VEHICLE_CLASSES, verbose=False)

            # Visualize results on frame
            annotated_frame = results[0].plot() # YOLO's built-in plot function
            out.write(annotated_frame)

            frame_count += 1
            progress_bar.progress(min(100, int((frame_count / total_frames) * 100)))

        cap.release()
        out.release()
        tfile.close()
        os.unlink(tfile.name) # Clean up temporary input file

        st.success("Video processing complete!")
        st.write("Detected vehicles in the video:")

        # Re-encode to H.264 for broader compatibility (Streamlit's video player prefers it)
        reencoded_output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        cmd = f"ffmpeg -y -i {output_video_path} -vcodec libx264 -preset medium -crf 23 {reencoded_output_path}"
        os.system(cmd)
        os.unlink(output_video_path) # Clean up intermediate output file

        st.video(reencoded_output_path, format="video/mp4", start_time=0)
        os.unlink(reencoded_output_path) # Clean up final output file after display

else:
    st.info("Please upload a video file to get started.")
