import streamlit as st
import cv2
import dlib
import tempfile
from ultralytics import YOLO

st.title("🚗 AI Driving Test (Video Analyzer)")
st.write("Upload your driving test video. The system will detect cone collisions and head movement using AI.")

# Load models
model = YOLO('yolov5s.pt')
face_detector = dlib.get_frontal_face_detector()

video_file = st.file_uploader("📤 Upload your test video", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_file)

    if st.button("▶️ Run AI Evaluation"):
        cap = cv2.VideoCapture(video_path)
        cone_detected = False
        face_missing = False
        total_frames = 0

        st.write("🧠 Analyzing...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            if total_frames % 10 != 0:
                continue  # Skip frames for speed

            # --- Cone Detection ---
            results = model(frame, verbose=False)
            boxes = results[0].boxes
            names = results[0].names
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    if label in ["cone", "traffic cone", "person"]:
                        cone_detected = True
                        st.warning(f"🚧 Cone detected at frame {total_frames}")
                        break

            # --- Face Detection ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)
            if len(faces) == 0:
                face_missing = True
                st.warning(f"🚫 Face not detected at frame {total_frames}")
                break

            if cone_detected or face_missing:
                break

        cap.release()

        st.subheader("📊 Final Result")
        if not cone_detected and not face_missing:
            st.success("✅ PASS: No cones touched, head stayed inside vehicle")
        else:
            st.error("❌ FAIL")
            if cone_detected:
                st.warning("⚠️ Cone touched or crossed.")
            if face_missing:
                st.warning("⚠️ Head moved out of vehicle (face not visible).")
