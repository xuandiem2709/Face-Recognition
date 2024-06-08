import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from recognizer import Recognizer
import numpy as np
import time
from datetime import datetime
from call_api import call_api_sync_data, post_attendance
from create_embedding_to_db import save_embeddings
from detect import FaceDetector
from face_alignment import frontalize_face


root = tk.Tk()
root.title("Attendance Device")
root.geometry("700x500")

# Create a canvas to hold the video capture box
canvas = tk.Canvas(root, width=630, height=400, bd=2, relief=tk.SOLID)
canvas.pack(pady=10)

recognizer = Recognizer()
detector = FaceDetector()
cap = cv2.VideoCapture(0)

def _update_video(type):
    name = "Unknown"

    # Read a frame from the cametypera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    timezone = "Asia/Ho_Chi_Minh"
    
    if ret:
        faces = detector(image=frame)

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
    root.after(10, _update_video)


def _sync_data():
    url = "http://localhost:8069/api/sync/employee"
    headers = {
        "api-key": "@HKo#@eud&oDl^I9Drmp"
    }
    call_api_sync_data(url=url, headers=headers)
    save_embeddings()
    messagebox.showinfo("Sync Data", "Sync Data Successfully")


# Create a notebook widget
notebook = ttk.Notebook(root)

# Create the Check-in Device tab
check_in_frame = ttk.Frame(notebook)
# check_in_button = ttk.Button(check_in_frame, text="Check-in", command=_update_video(type='check_in'))
check_in_button = ttk.Button(check_in_frame, text="Check-in")
check_in_button.pack(padx=10, pady=10)
notebook.add(check_in_frame, text="Check-in Device")

# Create the Check-out Device tab
check_out_frame = ttk.Frame(notebook)
# check_out_button = ttk.Button(check_out_frame, text="Check-out", command=_update_video(type='check_out'))
check_out_button = ttk.Button(check_out_frame, text="Check-out")
check_out_button.pack(padx=10, pady=10)
notebook.add(check_out_frame, text="Check-out Device")

# Create the Sync Data tab
sync_data_frame = ttk.Frame(notebook)
sync_label = ttk.Label(sync_data_frame, text="Sync Data From Server")
sync_label.pack(pady=10)

button = ttk.Button(sync_data_frame, text="Sync Data", command=_sync_data)
button.pack()

notebook.add(sync_data_frame, text="Sync Data")

notebook.pack(expand=True, fill=tk.BOTH)

# Start the main event loop
root.mainloop()

cap.release()