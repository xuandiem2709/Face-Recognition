import numpy as np
import face_recognition
import os
import pandas as pd
import cv2
from detect import FaceDetector
from recognizer import Recognizer
from face_alignment import frontalize_face


face_detector = FaceDetector()
recognize = Recognizer()
# align = FaceAlignment()

folder_path = "examples/datas"
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith((".jpg", ".jpeg", ".png"))]
known_face_embeddings = []
known_face_names = []

# Iterate over each image file
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)

    image = cv2.imread(image_path)
    faces = face_detector(image=image)

    box, landmarks, det_score = faces[0]
    x, y, w, h = map(int, box)
    # face_crop = image[y:y+h, x:x+w]
    facial_landmarks = landmarks.astype(np.int32)
    face_img, landmarks5, trans = frontalize_face(image, facial_landmarks)
    face_array = cv2.resize(face_img, (112, 112))
    face_array = np.array(face_array, dtype=np.float32)

    embeddings = recognize.vectorize(face_array)[0]
    embedding = embeddings[0]
    # print("embedding: ", embedding.shape)
    known_face_embeddings.append(embedding)
    
    # Extract the name from the file name
    name = os.path.splitext(image_file)[0]
    known_face_names.append(name)

folder_path = "examples/verifies"
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith((".jpg", ".jpeg", ".png"))]

df1 = pd.DataFrame(columns=known_face_names)
df2 = pd.DataFrame(columns=known_face_names)
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    faces = face_recognition.load_image_file(image_path)

    image = cv2.imread(image_path)
    faces = face_detector(image=image)

    box, landmarks, det_score = faces[0]
    x, y, w, h = map(int, box)
    # face_crop = image[y:y+h, x:x+w]
    facial_landmarks = landmarks.astype(np.int32)
    face_img, landmarks5, trans = frontalize_face(image, facial_landmarks)
    face_array = cv2.resize(face_img, (112, 112))
    face_array = np.array(face_array, dtype=np.float32)

    embeddings = recognize.vectorize(face_array)[0]
    embedding = embeddings[0]
    # print("embedding_shape: ", embedding.shape)
    
    similarities = [recognize.cosine_similarity(embedding, i) for i in known_face_embeddings]
    updated_similarities = [1-similarity for similarity in similarities]
    # print("similarities: ", updated_similarities)
    

    df_xlsx = pd.DataFrame([similarities], index=[os.path.splitext(image_file)[0]], columns=known_face_names)
    df_csv = pd.DataFrame([updated_similarities], index=[os.path.splitext(image_file)[0]], columns=known_face_names)
    df1 = pd.concat([df1, df_xlsx ], axis=0)
    df2 = pd.concat([df2, df_csv ], axis=0)
df1.to_excel('data.xlsx', sheet_name='total_distance')
df2.to_csv("data.csv", index=False, header=False)
