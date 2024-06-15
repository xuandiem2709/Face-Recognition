import cv2
import numpy as np
import os
import psycopg2
from config import load_config
from recognizer import Recognizer
from db.managers import DBManager
from db.common import dbSession
from detect import FaceDetector
from face_alignment import frontalize_face

FOLDER_PATH = "examples/emp_images"


SQLALCHEMY_DATABASE_URI = "postgresql://diemxuan:1@localhost:5432/face_test"
db_session = dbSession.create_database_session(SQLALCHEMY_DATABASE_URI)
def insert_user_embedding(known_face_name, known_face_encoding):
    """ Insert multiple vendors into the vendors table  """

    sql = "INSERT INTO users(username, embedding) VALUES (%s, %s) RETURNING id"
    config = load_config()
    user_id = None

    try:
        with  psycopg2.connect(**config) as conn:
            with  conn.cursor() as cur:
                del_record = "DELETE FROM users"
                cur.execute(del_record)
                # execute the INSERT statement
                for name, embedding in zip(known_face_name, known_face_encoding):
                    cur.execute(sql, (name, str(embedding),))

                # commit the changes to the database
                    rows = cur.fetchone()

                    if rows:
                        user_id = rows[0]
                    conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)    
    finally:
        return user_id

def save_embeddings():
    image_files = [f for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f)) and f.endswith((".jpg", ".jpeg", ".png"))]

    dbManager = DBManager(db_session)
    recognizer = Recognizer()
    detect = FaceDetector()

    dbManager.clear_users_embeddings()
    # Iterate over each image filed
    for image_file in image_files:
        image_path = os.path.join(FOLDER_PATH, image_file)

        image = cv2.imread(image_path)
        faces = detect(image=image)
        if faces:
            box, landmarks, det_score = faces[0]
            facial_landmarks = landmarks.astype(np.int32)
            face_img, landmarks5, trans = frontalize_face(image, facial_landmarks)
            face_array = cv2.resize(face_img, (112, 112))
            face_array = np.array(face_array, dtype=np.float32)

            embeddings = recognizer.vectorize(face_array)[0]
            embedding = embeddings[0]
            # print("embedding:", embedding.shape)

            # Extract the name from the file name
            name = os.path.splitext(image_file)[0]
            # print('name: ', name)
            # print("="*20)

            dbManager.create_embeddings(embedding=embedding, username=name)         