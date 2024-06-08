import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from recognizer import Recognizer
import numpy as np
import time
from detect import FaceDetector
from face_alignment import frontalize_face
import requests
from datetime import datetime
import json
import threading

# Create the main application window
app = tk.Tk()
app.title("Attendance Device")
app.geometry("700x500")

# Create a canvas to hold the video capture box
canvas = tk.Canvas(app, width=630, height=400, bd=2, relief=tk.SOLID)
canvas.pack(pady=10)

recognizer = Recognizer()
detector = FaceDetector()

cap = None

def post_attendance(image_id, type, timestamp, timezone):
    url = "http://127.0.0.1:8069/api/employee/attendance"
    headers = {
        "api-key": "@HKo#@eud&oDl^I9Drmp",
        "Content-Type": "application/json",
    }
    payload = {
        "image_id": image_id,
        "type": type,
        "timestamp": timestamp,
        "timezone": timezone
    }
    print("payload: ", payload)
    response = requests.post(url=url, data=json.dumps(payload), headers=headers)
    print("response: ", response.content)
    if response.status_code == 200:
        print("Successfully")
    else:
        print("Failed")

def process_frame(frame):
    name = "Unknown"  # Initialize name as "Unknown"
    type = "check_in"
    timezone = "Asia/Ho_Chi_Minh"
    
    faces = detector(image=frame)

    if faces:  # Check if any faces are detected
        box, landmarks, det_score = faces[0]
        x, y, w, h = map(int, box)
        facial_landmarks = landmarks.astype(np.int32)
        face_img, landmarks5, trans = frontalize_face(frame, facial_landmarks)
        face_array = cv2.resize(face_img, (112, 112))
        face_array = np.array(face_array, dtype=np.float32)

        input_embs = recognizer.vectorize(face_array)[0]
        input_emb = input_embs[0]            
        recognized = recognizer.compare(embedding=input_emb)
        print("recognized: ", recognized)
        
        if recognized[0] is not None:
            name = recognized[1]
            # time.sleep(3)
            # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # post_attendance(image_id=name, type=type, timestamp=timestamp, timezone=timezone)
            # cap.release()

        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x, h+35), font, 1.0, (255, 255, 255), 1)

    return frame

def update_video():
    global cap
    if cap is not None and cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        if ret:
            # Process the frame in a separate thread to avoid blocking the main loop
            threading.Thread(target=process_and_display, args=(frame,)).start()

        # Call this function again as soon as possible
        app.after(1, update_video)

def process_and_display(frame):
    processed_frame = process_frame(frame)
    
    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PIL image and then to an ImageTk object
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the canvas with the new frame
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk

def open_camera():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
    else:
        update_video()

# Create the button to open the camera
button = tk.Button(app, text="Open Camera", command=update_video)
button.pack(side=tk.BOTTOM, pady=20)

# Start the Tkinter event loop
app.mainloop()

cap.release()