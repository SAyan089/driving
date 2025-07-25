import streamlit as st
import cv2
import dlib
import tempfile
from ultralytics import YOLO
import os

# Streamlit page setup
st.set_page_config(page_title="AI Driving Test", layout="centered")
st.title("🚗 AI Driving Test – Upload Video")
st.write("This app evaluates driving test videos using AI. It detects cone collisions and head movement (face detection).")

# Load YOLOv5s (auto-downloads model from Ultralytics)
try:
    model = YOLO('yolov5s')
except Exception as e:
    st.error(f"Error loading YOLOv5s model: {e}")
    st.stop()

# Load face detector
face_detector = dlib.get_frontal_face_detector()

# Upload driving test video
video_file = st.file_uploader("📤 Upload your test video (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

if video_file:
    # Save video to temp file
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(video_file.read())
    video_path = temp_video.name

    # Display uploaded video
    st.video(video_file)

    if st.button("▶️ Run AI Evaluation"):
        st.info("⏳ Processing video... please wait.")
        cap = cv2.VideoCapture(video_path)

        cone_detected = False
        face_missing = False
        frame_count = 0
        last_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 10 != 0:
                continue  # Check every 10th frame

            display_frame = frame.copy()

            # YOLOv5s cone detection
            try:
                results = model(frame, verbose=False)
                boxes = results[0].boxes
                names = results[0].names
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        label = names[cls_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if label in ["cone", "traffic cone", "person"]:
                            cone_detected = True
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(display_frame, "CONE", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except Exception as e:
                st.warning(f"YOLO detection failed: {e}")
                break

            # Face detection using dlib
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)
            if len(faces) == 0:
                face_missing = True
            else:
                for f in faces:
                    x, y, w, h = f.left(), f.top(), f.width(), f.height()
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "FACE", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            last_frame = display_frame

            if cone_detected or face_missing:
                break

        cap.release()

        # Show preview frame with boxes
        if last_frame is not None:
            st.image(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB),
                     caption="📸 Detection Preview", use_column_width=True)

        # Result
        st.subheader("📊 Final Result")
        if not cone_detected and not face_missing:
            st.success("✅ PASS: No cones touched, and head stayed inside vehicle.")
        else:
            st.error("❌ FAIL")
            if cone_detected:
                st.warning("🚧 Cone touched or crossed.")
            if face_missing:
                st.warning("🚫 Head moved out of vehicle (face missing).")

        st.info(f"✅ Frames analyzed: {frame_count}")
