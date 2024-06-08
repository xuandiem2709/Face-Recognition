"""Database model for loggingface_db"""
__author__ = "TuyenQV"
__version__ = "0.1.0"
__date__ = "2021-Jan-18"
__maintainer__ = "TuyenQV"
__email__ = "tuyenqv@d-soft.com.vn"
__status__ = "Development"

import cv2
import logging
import numpy as np
import base64
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, LargeBinary
from sqlalchemy import BLOB, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql.expression import false
from werkzeug.security import generate_password_hash, check_password_hash

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
logger = logging.getLogger(__name__)


class Users(Base):
    """Embedding database class."""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(36), nullable=False)
    embedding = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)

    @staticmethod
    def create(data):
        """Create a new embedding from dict."""
        embedding = Users()
        embedding.from_dict(data, is_sync=False)
        return embedding

    def from_dict(self, data):
        """Add new embedding from dict."""
        for field in ['username', 'embedding', 'created_at']:
            try:
                if field == 'embedding':
                    # self.embedding = base64.b64decode(data[field])e
                    self.embedding = data[field].tobytes()
                elif field == 'created_at':
                    if data[field] is not None:
                        self.created_at = strtime2utc(data[field])
                    # else:
                    #     self.created_at = None
                else:
                    setattr(self, field, data[field])
            except Exception as e:
                logger.warn(e, exc_info=True)

    def to_dict(self):
        """Export Embedding to dict.
        Ref: https://stackabuse.com/encoding-and-decoding-base64-strings-in-python
        """
        return {
            'id': self.id,
            'embedding':  np.frombuffer(self.embedding, dtype=np.float32),
            'created_at': self.created_at.isoformat() + 'Z',
            'username': self.username
        }

    def __repr__(self):
        return f'User(id={self.id}, username={self.username}, embedding={self.embedding}, created_at={self.created_at})'

