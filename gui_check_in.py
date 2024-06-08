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
from call_api import post_attendance

# Create the main application window
app = tk.Tk()
app.title("Attendance Device")
app.geometry("700x500")

# Create a canvas to hold the video capture box
canvas = tk.Canvas(app, width=630, height=400, bd=2, relief=tk.SOLID)
canvas.pack(pady=10)

recognizer = Recognizer()
detector = FaceDetector()

cap = cv2.VideoCapture(0)


def update_video():
    name = "Unknown"  # Initialize name as "Unknown"

    # Read a frame from the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    type = "check_in"
    timezone = "Asia/Ho_Chi_Minh"
    
    if ret:
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
             
            if recognized[0] is not None:
                name = recognized[1]
                # time.sleep(3)
                # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # post_attendance(image_id=name, type=type, timestamp=timestamp, timezone=timezone)
                # cap.release()

            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x, h+35), font, 1.0, (255, 255, 255), 1)

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PIL image and then to an ImageTk object
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new frame
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk

    # Call this function again after 10 milliseconds
    app.after(10, update_video)

# Create the button to open the camera
button = tk.Button(app, text="Open Camera", command=update_video)
button.pack(side=tk.BOTTOM, pady=20)


# Start the Tkinter event loop
app.mainloop()

cap.release()