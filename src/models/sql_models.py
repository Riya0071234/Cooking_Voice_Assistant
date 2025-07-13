# src/models/sql_models.py
"""
Defines the SQLAlchemy ORM models for the project's database schema.

This centralizes all table definitions, ensuring consistency across the application.
"""

from sqlalchemy import (
    Column, Integer, String, Text, Float, JSON, DateTime, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class ContextualEntry(Base):
    """Maps to the 'contextual_entries' table."""
    __tablename__ = 'contextual_entries'

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False, unique=True) # Ensure questions are unique
    answer = Column(Text, nullable=False)
    intent = Column(String(100), default='troubleshooting')
    source_platform = Column(String(50))
    source_url = Column(String(512), unique=True)
    score = Column(Integer, default=0)
    language = Column(String(10), default='en')
    tags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ContextualEntry(id={self.id}, question='{self.question[:40]}...')>"

class Recipe(Base):
    """Maps to the 'recipes' table."""
    __tablename__ = 'recipes'
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    source_url = Column(String(512), unique=True)
    ingredients = Column(JSON, nullable=False)
    instructions = Column(JSON, nullable=False)
    cuisine = Column(String(100))
    tags = Column(JSON, nullable=True)

# You would add other models like YouTubeVideo here as well.

def get_db_session(db_url: str):
    """Creates a database engine and returns a session."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine) # Creates tables if they don't exist
    Session = sessionmaker(bind=engine)
    return Session()