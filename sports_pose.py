import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Sports Pose Analysis",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 Sports Pose Analysis")

st.write(
    "Upload a sports image or video to detect human body keypoints "
    "using YOLO11-Pose."
)

# --------------------------------------------------
# Load YOLO11 Pose model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolo11n-pose.pt")


model = load_model()

# --------------------------------------------------
# Select input type
# --------------------------------------------------
option = st.radio(
    "Choose Input Type",
    ["🖼️ Image", "🎥 Video"],
    horizontal=True
)


# ==================================================
# IMAGE POSE ANALYSIS
# ==================================================

if option == "🖼️ Image":

    st.subheader("🖼️ Image Pose Analysis")

    # Upload image
    uploaded_image = st.file_uploader(
        "Upload Sports Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        # Read uploaded image
        file_bytes = uploaded_image.read()

        image = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            st.error("Could not read the image.")
            st.stop()

        # Display original image
        st.subheader("Original Image")

        original_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            original_rgb,
            caption="Uploaded Sports Image",
            width=500
        )

        # YOLO11-Pose inference
        with st.spinner("Detecting sports pose..."):

            results = model(
                image,
                conf=0.5,
                verbose=False
            )

        # Draw skeleton and keypoints
        annotated_image = results[0].plot()

        # Convert BGR to RGB
        annotated_rgb = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB
        )

        # Display pose result
        st.subheader("Pose Detection Result")

        st.image(
            annotated_rgb,
            caption="Detected Body Keypoints and Skeleton",
            width=500
        )

        st.success("Pose detection completed!")


# ==================================================
# VIDEO POSE ANALYSIS
# ==================================================

elif option == "🎥 Video":

    st.subheader("🎥 Video Pose Analysis")

    # Upload video
    uploaded_file = st.file_uploader(
        "Upload Sports Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:

        # Save uploaded video temporarily
        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(
            uploaded_file.read()
        )

        temp_input.close()

        # Open video
        cap = cv2.VideoCapture(
            temp_input.name
        )

        if not cap.isOpened():
            st.error("Could not open the video.")
            os.remove(temp_input.name)
            st.stop()

        # Get video information
        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Prevent invalid FPS
        if fps <= 0:
            fps = 30

        # Create output video
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name
        output_file.close()

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        # Progress UI
        st.subheader("Pose Detection")

        video_placeholder = st.empty()

        progress_bar = st.progress(0)

        frame_number = 0

        # Process video frame by frame
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # YOLO11-Pose inference
            results = model(
                frame,
                conf=0.5,
                verbose=False
            )

            # Draw skeleton and keypoints
            annotated_frame = results[0].plot()

            # Write processed frame
            out.write(
                annotated_frame
            )

            # Convert BGR to RGB
            annotated_rgb = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            # Display processed frame
            video_placeholder.image(
                annotated_rgb,
                channels="RGB",
                width=700
            )

            # Update progress
            if total_frames > 0:

                progress = (
                    frame_number / total_frames
                )

                progress_bar.progress(
                    min(progress, 1.0)
                )

        # Release resources
        cap.release()
        out.release()

        progress_bar.progress(1.0)

        st.success(
            "Pose detection completed!"
        )

        # Display processed video
        st.subheader("Processed Video")

        with open(
            output_path,
            "rb"
        ) as video:

            video_bytes = video.read()

        st.video(
            video_bytes
        )

        # Download processed video
        st.download_button(
            label="⬇️ Download Processed Video",
            data=video_bytes,
            file_name="sports_pose_output.mp4",
            mime="video/mp4"
        )

        # Cleanup temporary files
        os.remove(temp_input.name)
        os.remove(output_path)
