import logging
import numpy as np
import sqlalchemy
from ..models import Users
from coolname import generate_slug
import time
logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self, session):
        self.session = session
        self.embedding_dims = 512
        # self.embedding_dims = 128
    
    def fetch_embeddings_data(self):
        """Get embedding and user data from database
        Returns:
            user_dict (dict) : map user ids to user names
            embeddings (numpy.array) : array of embedding
            embedding_dict (dict) : map array row id to user id
        """
        users = self.session.query(Users).all()
        embeddings = np.empty((0, self.embedding_dims))
        embedding_dict = []
        for user in users:
            user_data = user.to_dict()
            
            if user_data['embedding'] is not None and len(user_data['embedding']) > 0:

                embedding = np.frombuffer(bytes(user_data['embedding']), dtype=np.float32)
                embedding = np.expand_dims(embedding, axis=0)
                embeddings = np.append(embeddings, embedding, axis=0)
                embedding_dict.append([user_data['id'], user_data['username']])
        self.session.close()
        return embeddings, embedding_dict
    
    def create_embeddings(self, embedding, username=None):
        """Create a new embedding from dict for an employee."""
        user = Users()
        # embedding = np.empty((0, self.embedding_dims))
        user.from_dict(dict(
            username=generate_slug(2) if not username else username,
            embedding=embedding,
            created_at=None
        ))
        try:
            self.session.add(user)
            self.session.flush()
            data = user.to_dict()
            self.session.commit()
            self.session.close()
        except Exception as e:
            self.session.rollback()
            print(f"Rollbackk ... {e}")
        return data
    
    def clear_users_embeddings (self):
        query = self.session.query(Users)
        query.delete()
        self.session.commit()


