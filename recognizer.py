import cv2
import numpy as np
import logging
from db.managers import DBManager
from db.common import dbSession


THRESHOLD = 0.45
MIN_SIMILARITY_DIFFERENCE = 0.1

logger = logging.getLogger(__name__)

import onnxruntime as ort

EP_list = ['CPUExecutionProvider']
session = ort.InferenceSession('w600k_r50.onnx', providers=EP_list)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

SQLALCHEMY_DATABASE_URI = "postgresql://diemxuan:1@localhost:5432/face_test"
db_session = dbSession.create_database_session(SQLALCHEMY_DATABASE_URI)
class Recognizer:
    def __init__(self, **kwargs):  
        self._dbmanager = DBManager(db_session)       
    
    def vectorize(self, img, normalize=True):
        img = img.astype(np.float32)
        img = cv2.resize(img, (112, 112))
        if normalize: img = (img - 127.5) / 127.5

        # 2. inferencing
        img = img.transpose((2, 0, 1)) # HWC->CHW transformation
        img = np.expand_dims(img, axis=0) 

        embedding = session.run([output_name], {input_name: img})

        return embedding
    
    @staticmethod
    def cosine_similarity(x, y):
        x = x / np.linalg.norm(x) # L2 norm
        y = y / np.linalg.norm(y) # L2 norm
        return np.dot(x, y)

    def compare(self, embedding):
        """Perform Face Recognition

        Args:
            embedding (numpy.array) : user's embedding
            embeddings (numpy.array) : array of database's embeddings
            embedding_dict (numpy.array) : list of tuple (user_id, username)

        Returns:
            List[str, str] : list of recognized user ID and user name
                in string format
        """

        embeddings, embedding_dict = self._dbmanager.fetch_embeddings_data()

        similarities = [Recognizer.cosine_similarity(embedding, i) for i in embeddings]
        # print("similarities: ", similarities)
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        second_max_similarity = 0
        if len(similarities) >= 2:
            second_max_similarity = sorted(similarities)[-2]

        if max_similarity > THRESHOLD\
           and (max_similarity - second_max_similarity) > MIN_SIMILARITY_DIFFERENCE:
            embedding_dict[max_idx].append(1)
            return embedding_dict[max_idx]
        else:
            return [None, 'Unknown']