import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 AI Vehicle Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. CUSTOM CSS STYLING (DARK NEON GLOW THEME)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Neon Glow Title Header */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Metric Cards Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00f2fe;
    }
    .metric-label {
        color: #cbd5e0;
        font-size: 0.9rem;
    }

    /* Customizing Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")  # Replace with your custom trained model path if needed

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Vehicle Class IDs for standard YOLOv8 COCO (2: car, 3: motorcycle, 5: bus, 7: truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

# ---------------------------------------------------------
# 4. SIDEBAR CONTROL PANEL
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.markdown("---")

input_type = st.sidebar.radio("📥 *Select Input Source*", ["Image", "Video"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Detection Parameters")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", min_value=0.05, max_value=1.0, value=0.45, step=0.05)

st.sidebar.markdown("---")
show_labels = st.sidebar.checkbox("Show Bounding Box Labels", value=True)
st.sidebar.caption("🚀 Powered by *Ultralytics YOLOv8* & *Streamlit*")

# ---------------------------------------------------------
# 5. MAIN HEADER SECTION
# ---------------------------------------------------------
st.markdown('<p class="main-title">🚗 YOLOv8 Vehicle Detection & Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced real-time computer vision system for automated vehicle identification and analytics.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. INFERENCE & VISUALIZATION LOGIC
# ---------------------------------------------------------

# --- IMAGE MODE ---
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image file", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Run YOLO Inference
        results = model.predict(img_array, conf=conf_threshold, iou=iou_threshold, classes=VEHICLE_CLASSES)
        res = results[0]

        vehicle_count = len(res.boxes)
        boxes = res.boxes.data.cpu().numpy()

        # Display Live Metric Cards
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{vehicle_count}</div><div class="metric-label">Total Vehicles Detected</div></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{conf_threshold:.2f}</div><div class="metric-label">Min Confidence Applied</div></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">Active</div><div class="metric-label">Model Status</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-Side Comparison
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original Input Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("⚡ Detection Output")
            annotated_frame = res.plot(labels=show_labels)
            st.image(annotated_frame, channels="BGR", use_container_width=True)

# --- VIDEO MODE ---
else:
    uploaded_file = st.file_uploader("Upload a Video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.markdown("---")
        st.subheader("🎥 Processing Video Stream...")

        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection frame by frame
            results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, classes=VEHICLE_CLASSES)
            annotated_frame = results[0].plot(show_labels=show_labels)

            # Display processed video live
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Video processing completed!")
