from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config.manager import get_config


DATABASE_URL = get_config().db.db_url

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread":False})
SessionLocal = sessionmaker(engine, autoflush=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db

    finally:
        db.close()