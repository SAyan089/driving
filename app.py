import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="AI Driving Test", layout="centered")
st.title("🚗 AI Driving Test – Cone Detection Only")
st.write("This app detects cones in a driving test video using AI (YOLOv5s).")

# Load YOLOv5s model
model = YOLO('yolov5s')

# Upload video
video_file = st.file_uploader("📤 Upload driving test video", type=["mp4", "mov", "avi"])

if video_file:
    # Save video to temp file
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(video_file.read())
    video_path = temp_video.name

    # Show uploaded video
    st.video(video_file)

    if st.button("▶️ Run AI Test"):
        cap = cv2.VideoCapture(video_path)
        cone_detected = False
        last_frame = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 != 0:
                continue  # Every 10th frame

            display_frame = frame.copy()

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
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            last_frame = display_frame

            if cone_detected:
                break

        cap.release()

        # Show preview
        if last_frame is not None:
            st.image(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB),
                     caption="📸 Detection Preview", use_column_width=True)

        # Final Result
        st.subheader("📊 Final Result")
        if not cone_detected:
            st.success("✅ PASS: No cone or person detected.")
        else:
            st.error("❌ FAIL: Cone or person detected in path.")
