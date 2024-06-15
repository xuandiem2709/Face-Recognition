import requests
import os
import base64
from PIL import Image
from io import BytesIO
import json

import cv2
import numpy as np
from recognizer import Recognizer
from db.managers import DBManager
from db.common import dbSession
from detect import FaceDetector
from face_alignment import frontalize_face


SQLALCHEMY_DATABASE_URI = "postgresql://diemxuan:1@localhost:5432/face_test"
db_session = dbSession.create_database_session(SQLALCHEMY_DATABASE_URI)

URL = "http://127.0.0.1:8069/api/sync/employee"
HEADERS = {
    "api-key": "@HKo#@eud&oDl^I9Drmp",
}

def save_embeddings(image, email):
    dbManager = DBManager(db_session)
    recognizer = Recognizer()
    detect = FaceDetector()
    
    faces = detect(image=image)
    if faces:
        box, landmarks, det_score = faces[0]
        facial_landmarks = landmarks.astype(np.int32)
        face_img, landmarks5, trans = frontalize_face(image, facial_landmarks)
        face_array = cv2.resize(face_img, (112, 112))
        face_array = np.array(face_array, dtype=np.float32)

        embeddings = recognizer.vectorize(face_array)[0]
        embedding = embeddings[0]

        # name = email.split("@")[0]

        dbManager.create_embeddings(embedding=embedding, username=email)


def base64_to_image(base64_string):
    # Remove the data URI prefix if present
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(base64_string)
    return image_bytes

def create_image_from_bytes(image_bytes):
    # Create a BytesIO object to handle the image data
    image_stream = BytesIO(image_bytes)

    # Open the image using Pillow (PIL)
    image = Image.open(image_stream)
    return image

def handle_data_synced():
    response = requests.get(url=URL, headers=HEADERS)
    employees = list(response.json()['employees'])
    if len(employees):
        dbManager = DBManager(db_session)
        dbManager.clear_users_embeddings()

        for emp in employees:
            if emp['image'] and emp['email']:
                email = emp["email"]
                # name = email.split("@")
                # image_name = name[0] + ".jpg"
                # file_path = os.path.join(folder, image_name)
                image_bytes = base64_to_image(emp['image'])
                img_byte = create_image_from_bytes(image_bytes)
                img = np.array(img_byte)
                save_embeddings(image=img, email=email)
                # img.save(file_path)

def post_attendance(payload):
    url = "http://127.0.0.1:8069/api/employee/attendance"
    headers = {
        "api-key": "@HKo#@eud&oDl^I9Drmp",
        "Content-Type": "application/json",
    }
    response = requests.post(url=url, data=json.dumps(payload), headers=headers)
    return response