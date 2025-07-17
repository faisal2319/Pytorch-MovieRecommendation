import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint, PrimaryKeyConstraint
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import inspect

# Database configuration from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password123@localhost:5432/imdb_recommender")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Reusable helper to get PKs ---
def get_primary_key_columns(model_class):
    """Returns a list of primary key column names for a SQLAlchemy model."""
    return [key.name for key in inspect(model_class).primary_key]


# --- Base Model ---
class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True, index=True)  # MovieLens userId
    username = Column(String, unique=True, nullable=True)
    ratings          = relationship("Rating", back_populates="user", cascade="all, delete-orphan")
    recommendations  = relationship("BatchRecommendation", back_populates="user", cascade="all, delete-orphan")

class Movie(Base):
    __tablename__ = "movies"
    id        = Column(Integer, primary_key=True, index=True) # MovieLens movieId
    imdb_id   = Column(String, unique=True, index=True)      # IMDb tconst
    title     = Column(String, index=True)
    genres    = Column(String)                                # pipe-separated
    start_year = Column(Integer, nullable=True)               # Release year
    runtime_minutes = Column(Integer, nullable=True)         # Runtime in minutes
    ratings          = relationship("Rating", back_populates="movie", cascade="all, delete-orphan")
    recommendations  = relationship("BatchRecommendation", back_populates="movie", cascade="all, delete-orphan")
    principals       = relationship("MoviePerson", back_populates="movie", cascade="all, delete-orphan")

class Rating(Base):
    __tablename__ = "ratings"
    
    user_id   = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False)
    movie_id  = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True, nullable=False)
    rating    = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    
    user  = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")

class Person(Base):
    """
    Represents a person (actor/director/etc.) from IMDb.
    """
    __tablename__ = 'persons'
    id = Column(String, primary_key=True) # nconst from IMDb
    name = Column(String)
    birth_year = Column(Integer, nullable=True)
    death_year = Column(Integer, nullable=True)
    primary_profession = Column(String, nullable=True)
    principals = relationship("MoviePerson", back_populates="person")

class MoviePerson(Base):
    """
    Represents the association between a movie and a person (principal).
    Uses a composite primary key as a person can have multiple roles in a movie.
    """
    __tablename__ = 'movie_persons'

    movie_id = Column(Integer, ForeignKey('movies.id'), primary_key=True)
    person_id = Column(String, ForeignKey('persons.id'), primary_key=True)
    category = Column(String, primary_key=True)
    ordering = Column(Integer, primary_key=True)

    job = Column(String, nullable=True)
    characters = Column(String, nullable=True)
    
    movie = relationship("Movie", back_populates="principals")
    person = relationship("Person", back_populates="principals")
    
    __table_args__ = (
        PrimaryKeyConstraint('movie_id', 'person_id', 'category', 'ordering'),
    )

class BatchRecommendation(Base):
    __tablename__ = "batch_recommendations"
    __table_args__ = (
        UniqueConstraint('user_id', 'rank', name='uix_user_rank'),
    )
    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    movie_id         = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), index=True, nullable=False)
    predicted_rating = Column(Float, nullable=False)
    rank             = Column(Integer, nullable=False) # position: 1,2,...
    generated_at     = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    user  = relationship("User", back_populates="recommendations")
    movie = relationship("Movie", back_populates="recommendations") 