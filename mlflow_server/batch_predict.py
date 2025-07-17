import os
import sys
import torch
import numpy as np
import datetime
from tqdm import tqdm
import mlflow
from mlflow import pytorch
from mlflow.tracking import set_tracking_uri

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.database import SessionLocal, User, Movie, Rating, BatchRecommendation
from database.model import HybridRecommender
import pickle
from typing import List

# --- CONFIG ---
TOP_N = 10  # Number of recommendations per user
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), 'database', 'prediction_artifacts.pkl')
MODEL_NAME = "hybrid-recommender"  # Name in MLflow model registry
MODEL_ALIAS = "staging"  # Or "production"

# --- MLflow tracking URI setup ---
is_container = os.path.exists('/.dockerenv')
if is_container:
    tracking_uri = "http://mlflow:5000"
else:
    tracking_uri = "http://localhost:5001"
set_tracking_uri(tracking_uri)

# --- Load model from MLflow ---
try:
    # Since Version 1 has the staging alias, load it directly
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model from: {model_uri}")
    model = pytorch.load_model(model_uri)
    print("✅ Model loaded successfully from MLflow (Version 1 with staging alias)")
        
except Exception as e:
    print(f"❌ Failed to load model version 1: {e}")
    print("Trying to load latest version...")
    try:
        model_uri = f"models:/{MODEL_NAME}/latest"
        model = pytorch.load_model(model_uri)
        print("✅ Model loaded successfully (latest version)")
    except Exception as e2:
        print(f"❌ Failed to load latest version: {e2}")
        print("Available options:")
        print("- Make sure the model is registered in MLflow")
        print("- Check if MLflow server is running")
        print("- Try using a different alias or version")
        raise e2

model.eval()

# --- Load artifacts ---
with open(ARTIFACTS_PATH, 'rb') as f:
    artifacts = pickle.load(f)

user_to_idx = artifacts['user_to_idx']
movie_to_idx = artifacts['movie_to_idx']
movie_features = artifacts['movie_features']

num_users = len(user_to_idx)
num_movies = len(movie_to_idx)

# --- DB session ---
session = SessionLocal()

# --- Get all users and movies ---
users = session.query(User).all()
movies: List[Movie] = session.query(Movie).all()

# --- Build user-movie rating lookup ---
user_rated_movies = {}
for r in session.query(Rating.user_id, Rating.movie_id):
    user_rated_movies.setdefault(r.user_id, set()).add(r.movie_id)

# --- Batch recommend ---
for user in tqdm(users, desc='Users'):
    user_id = user.id
    if user_id not in user_to_idx:
        continue  # Skip users not in training
    rated = user_rated_movies.get(user_id, set())
    candidate_movies = [m for m in movies if m.id not in rated and m.id in movie_to_idx]
    if not candidate_movies:
        continue
    # Prepare features for all candidate movies
    user_idx = user_to_idx[user_id]
    user_ids = np.full(len(candidate_movies), user_idx, dtype=np.int64)
    movie_idxs = np.array([movie_to_idx[m.id] for m in candidate_movies], dtype=np.int64)
    genres = np.stack([movie_features[m.id]['genres_encoded'] for m in candidate_movies])
    titles_st = np.stack([movie_features[m.id]['title_st_embedding'] for m in candidate_movies])
    directors = np.stack([movie_features[m.id]['director_ids_encoded'] for m in candidate_movies])
    actors = np.stack([movie_features[m.id]['actor_ids_encoded'] for m in candidate_movies])
    numerical = np.stack([movie_features[m.id]['numerical_scaled'] for m in candidate_movies])
    # Predict
    with torch.no_grad():
        preds = model.predict({
            'user_ids': user_ids,
            'movie_ids_collaborative': movie_idxs,
            'genres_input': genres,
            'title_st_embedding_input': titles_st,
            'director_ids': directors,
            'actor_ids': actors,
            'numerical_inputs': numerical
        })
    # Top N
    top_idx = np.argsort(-preds)[:min(TOP_N, len(candidate_movies))]
    top_movies = [candidate_movies[i] for i in top_idx]
    top_scores = preds[top_idx].tolist()  # Ensure this is a list
    # Clear old recs
    session.query(BatchRecommendation).filter_by(user_id=user_id).delete()
    # Insert new recs
    now = datetime.datetime.utcnow()
    for rank, (movie, score) in enumerate(zip(top_movies, top_scores), 1):
        rec = BatchRecommendation(
            user_id=user_id,
            movie_id=movie.id,
            predicted_rating=float(score),
            rank=rank,
            generated_at=now
        )
        session.add(rec)
    session.commit()

print('Batch recommendations generated and stored in DB.')
session.close() 