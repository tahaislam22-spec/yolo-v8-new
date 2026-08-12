
# YOLOv8 Vehicle Detection Streamlit App

This is a simple Streamlit application that uses the Ultralytics YOLOv8 model to perform real-time object detection on uploaded video files, specifically identifying vehicles (cars, motorcycles, buses, trucks).

## Setup and Deployment

To deploy this application to Streamlit Cloud (or run locally):

1.  **Create a GitHub Repository**: Make sure your code is in a GitHub repository.
2.  **Create `Deployment` Folder**: Inside your repository, create a folder named `Deployment`.
3.  **Place Files**: Put `streamlit_app.py`, `requirements.txt`, `README.md`, your chosen YOLOv8 weights (`yolov8n.pt` or `best.pt`), and the `sample_video.mp4` into this `Deployment` folder.
    *   **Model Weights**: You need to include the `yolov8n.pt` file in the `Deployment` folder. If you have a custom trained model (`best.pt`), place that file in the `Deployment` folder as well. The `streamlit_app.py` is configured to prefer `best.pt` if it exists, otherwise, it will use `yolov8n.pt`.
    *   **Sample Video**: Include `sample_video.mp4` for demonstration purposes.
4.  **Streamlit Cloud**: Connect your GitHub repository to Streamlit Cloud, pointing it to the `Deployment` folder as the root directory for the app.

## Running Locally

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>/Deployment
    ```
2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Model Weights**: Download `yolov8n.pt` from the Ultralytics GitHub releases and place it in the `Deployment` folder. If you have a custom `best.pt` model, place it here instead.
5.  **Run the Streamlit App**:
    ```bash
    streamlit run streamlit_app.py
    ```
    This will open the application in your web browser.

## How to Use

1.  Upload a video file (MP4, AVI, MOV, MKV).
2.  The application will process the video, running YOLOv8 object detection on each frame.
3.  The processed video, with bounding boxes around detected vehicles, will be displayed in the app.

Enjoy detecting vehicles with YOLOv8!