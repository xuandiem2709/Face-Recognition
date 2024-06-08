import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from recognizer import Recognizer
import numpy as np
import time
from detect import FaceDetector
from face_alignment import frontalize_face

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

name = "Unknown"  # Initialize name as "Unknown"
input_emb = None  # Initialize input_emb as None
last_frame = None  # To store the last captured frame
capture_enabled = True  # Flag to control frame capture
wait_period = False  # Flag to control the 4-second wait period

def update_video():
    global input_emb, last_frame, capture_enabled

    if capture_enabled:
        # Read a frame from the camera
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
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

            # Store the last frame
            last_frame = frame_rgb

    else:
        # Display the last frame
        if last_frame is not None:
            img = Image.fromarray(last_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.image = img_tk

    # Call this function again after 10 milliseconds
    app.after(10, update_video)

def periodic_recognition():
    global name, input_emb, capture_enabled, wait_period

    if not wait_period:
        if input_emb is not None:
            recognize = recognizer.compare(embedding=input_emb)
            if recognize[0] is not None:
                name = recognize[1]
                print("recognize: ", recognize)
                # Turn off camera
                capture_enabled = False
                wait_period = True
                # Schedule to turn on the camera after 4 seconds
                app.after(4000, turn_on_camera)
            else:
                name = "Unknown"

        # Schedule the next call to this function after 3000 milliseconds (3 seconds)
        app.after(3000, periodic_recognition)

def turn_on_camera():
    global capture_enabled, wait_period
    capture_enabled = True
    wait_period = False
    # Restart periodic recognition
    periodic_recognition()

button = tk.Button(app, text="Open Camera", command=update_video)
button.pack(side=tk.BOTTOM, pady=20)

# Schedule the first call to periodic_recognition
app.after(3000, periodic_recognition)

# Start the Tkinter event loop
app.mainloop()

cap.release()