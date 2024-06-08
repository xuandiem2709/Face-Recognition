from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def create_database_session(database_uri):
    # Create the engine
    engine = create_engine(database_uri)
    
    # Create a session factory
    Session = sessionmaker(bind=engine)
    
    # Create a session
    session = Session()
    
    return session