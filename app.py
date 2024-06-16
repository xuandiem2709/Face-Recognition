import streamlit as st
import cv2
import numpy as np
from detect import FaceDetector
from recognizer import Recognizer
from face_alignment import frontalize_face
from datetime import datetime
from call_api import post_attendance, handle_data_synced


detect = FaceDetector()
recognizer = Recognizer()
def capture_frame(cap):   
    name = "Unknown"
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        st.error("Failed to capture image")
        return None
    faces = detect(image=frame)
    if faces:
        box, landmarks, det_score = faces[0]
        x, y, w, h = map(int, box)
        facial_landmarks = landmarks.astype(np.int32)
        face_img, landmarks5, trans = frontalize_face(frame, facial_landmarks)
        face_array = cv2.resize(face_img, (112, 112))
        face_array = np.array(face_array, dtype=np.float32)

        input_embs = recognizer.vectorize(face_array)[0]
        input_emb = input_embs[0]            
        recognized = recognizer.compare(embedding=input_emb)
        if recognized[0] is not None:
            name = recognized[1]

    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (x, h+35), font, 1.0, (255, 255, 255), 1)
    return frame

def handle_recognition_detection(frame):   
    name = "Unknown"
    
    faces = detect(image=frame)
    if faces:
        box, landmarks, det_score = faces[0]
        x, y, w, h = map(int, box)
        facial_landmarks = landmarks.astype(np.int32)
        face_img, landmarks5, trans = frontalize_face(frame, facial_landmarks)
        face_array = cv2.resize(face_img, (112, 112))
        face_array = np.array(face_array, dtype=np.float32)

        input_embs = recognizer.vectorize(face_array)[0]
        input_emb = input_embs[0]            
        recognized = recognizer.compare(embedding=input_emb)
        if recognized[0] is not None:
            name = recognized[1]

    return name, x, y, w, h

def main():
    st.set_page_config(page_title="Attendance App")
    st.markdown("<h1 style='font-size: 36px;'>Face Recognition Attendance System</h1>", unsafe_allow_html=True)

   # Initialize session state variables if they don't exist
    if 'stop' not in st.session_state:
        st.session_state.stop = True  # Default to stop to show the home page
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None
    if 'action' not in st.session_state:
        st.session_state.action = "home"  # Default action
        

    # Sidebar menu
    st.sidebar.title("Menu")
    if st.sidebar.button("Home"):
        st.session_state.stop = True
        st.session_state.action = "home"    

    # Sync Data
    if st.sidebar.button("Synchronize data from server", key="sidebar_sync"):
        st.session_state.stop = True
        st.session_state.action = "sync_data"

    # Check-in
    if st.sidebar.button("Start Doing Check In", key="sidebar_check_in", ):
        st.session_state.stop = False
        st.session_state.action = "check-in"

    # Check-out
    if st.sidebar.button("Start Doing Check Out", key="sidebar_check_out"):
        st.session_state.stop = False
        st.session_state.action = "check-out"

    if st.session_state.action == "home":
        st.image("images/face-recognition-attendance-system.jpg")

    # Render content based on the action
    elif st.session_state.action == "sync_data" and st.session_state.stop == True:
        st.write("Click the button below to sync data with the human resources management system.")
        if st.button("Sync Data", key="sync_data"):
            handle_data_synced()
            st.success("Data synced successfully!")

    elif st.session_state.stop == False and st.session_state.action in ["check-in", "check-out"]:
        frame_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        frame_count = 0
        recognized_email = ""
        
        while (True):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_count += 1
            if frame_count >= 35:
                email, x, y, w, h = handle_recognition_detection(frame=frame)
                name = email.split("@")[0]
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x, h+35), font, 1.0, (255, 255, 255), 1)
                recognized_name = name
                # final_frame = frame
                recognized_email = email
            frame_placeholder.image(frame)
            if frame_count == 40 and recognized_name != "Unknown":
                break
            if frame_count >= 50 and recognized_name == "Unknown":
                st.error(f"Opps, {st.session_state.action} failed!")
                break
                
        cap.release()
        cv2.destroyAllWindows()
        if recognized_name != "Unknown":
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            payload = {
                "email": recognized_email,
                "type": st.session_state.action,
                "timezone": "Asia/Ho_Chi_Minh",
                "timestamp": current_time
            }
            response = post_attendance(payload)
            response_json = response.json()
            if response_json['status'] == 200:
                st.success(f"{recognized_name} {st.session_state.action} successfully at {current_time}")
            else:
                st.error("Odoo server is having an unexpected error.")

if __name__ == "__main__":
    main()

