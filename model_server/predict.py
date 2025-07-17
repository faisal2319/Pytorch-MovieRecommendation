import torch
import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import sys
import mlflow
import mlflow.pytorch
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy.orm import Session

# Add parent directory to sys.path to allow importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.model import HybridRecommender
from database.database import SessionLocal, Movie, Rating, MoviePerson, Person


class MovieRecommenderPredictor:
    """
    A comprehensive prediction class that handles:
    1. Loading preprocessing artifacts
    2. Loading models from MLflow or direct file
    3. Making predictions for existing movies in the database
    4. Making predictions for new/unseen movies with raw content features
    """
    
    def __init__(self):
        self.model: Optional[HybridRecommender] = None
        self.device: Optional[torch.device] = None
        self.st_model: Optional[SentenceTransformer] = None
        self.artifacts: Optional[Dict[str, Any]] = None
        
    def load_artifacts(self, artifacts_path: str = 'database/prediction_artifacts.pkl'):
        """Load preprocessing artifacts needed for feature engineering."""
        print(f"📦 Loading preprocessing artifacts from {artifacts_path}...")
        
        if not os.path.exists(artifacts_path):
            # Try alternative paths
            alt_paths = [
                'database/processed/all_processed_data.pkl',
                '../database/prediction_artifacts.pkl',
                '../database/processed/all_processed_data.pkl'
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    artifacts_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Could not find artifacts file. Tried: {artifacts_path}, {alt_paths}")
        
        with open(artifacts_path, 'rb') as f:
            self.artifacts = pickle.load(f)
        
        # Type guard to ensure artifacts is loaded
        if self.artifacts is None:
            raise ValueError("Failed to load artifacts")
        
        # Derive missing metadata from what's available (for lightweight prediction_artifacts.pkl)
        if 'num_users' not in self.artifacts:
            self.artifacts['num_users'] = len(self.artifacts['user_to_idx'])
            
        if 'num_movies_ml' not in self.artifacts:
            self.artifacts['num_movies_ml'] = len(self.artifacts['movie_to_idx'])
            
        if 'num_genres' not in self.artifacts:
            self.artifacts['num_genres'] = len(self.artifacts['genre_labels'])
            
        if 'num_directors' not in self.artifacts:
            self.artifacts['num_directors'] = len(self.artifacts['director_to_idx'])
            
        if 'num_actors' not in self.artifacts:
            self.artifacts['num_actors'] = len(self.artifacts['actor_to_idx'])
            
        if 'num_numerical_features' not in self.artifacts:
            self.artifacts['num_numerical_features'] = 2  # runtime and year (standard)
        
        print("✅ Artifacts loaded successfully!")
        print(f"   - Users: {self.artifacts['num_users']}")
        print(f"   - Movies: {self.artifacts['num_movies_ml']}")
        print(f"   - Genres: {self.artifacts['num_genres']}")
        print(f"   - ST Embedding Dim: {self.artifacts['st_embedding_dim']}")
        print(f"   - Directors: {self.artifacts['num_directors']}")
        print(f"   - Actors: {self.artifacts['num_actors']}")
        print(f"   - Numerical Features: {self.artifacts['num_numerical_features']}")
        
        # Check if this is the complete training artifact or lightweight prediction artifact
        if 'movie_features' in self.artifacts:
            print("   - Using prediction-optimized artifact with pre-computed features")
        else:
            print("   - Using complete training artifact")
        
    def load_sentence_transformer(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Load the sentence transformer model for title embeddings."""
        print(f"🤖 Loading Sentence Transformer: {model_name}...")
        
        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("✅ Using Apple MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("✅ Using CUDA GPU")
        else:
            self.device = torch.device('cpu')
            print("✅ Using CPU")
            
        self.st_model = SentenceTransformer(model_name)
        self.st_model.to(self.device)
        print("✅ Sentence Transformer loaded!")
        
    def load_model_from_mlflow(self, 
                              tracking_uri: Optional[str] = None,
                              model_name: str = "hybrid-recommender",
                              model_alias: str = "staging") -> bool:
        """Load model from MLflow Model Registry."""
        try:
            # Auto-detect appropriate tracking URI if not provided
            if tracking_uri is None:
                # Check if we're running inside a container
                is_container = os.path.exists('/.dockerenv')
                if is_container:
                    tracking_uri = "http://mlflow:5000"  # Internal service name
                    print("🐳 Detected container environment")
                else:
                    tracking_uri = "http://localhost:5001"  # Host-mapped port
                    print("💻 Detected local environment")
            
            print(f"🔄 Loading model from MLflow...")
            print(f"   - Tracking URI: {tracking_uri}")
            print(f"   - Model: {model_name}@{model_alias}")
            
            mlflow.set_tracking_uri(tracking_uri)
            model_uri = f"models:/{model_name}@{model_alias}"
            self.model = mlflow.pytorch.load_model(model_uri)
            
            if self.device:
                self.model.to(self.device)
            
            self.model.eval()
            print("✅ Model loaded from MLflow successfully!")
            return True
            
        except Exception as e:
            print(f"⚠️  MLflow model loading failed: {e}")
            return False
    
    def load_model_from_file(self, model_path: str):
        """Load model directly from a .pth file."""
        print(f"📁 Loading model from file: {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not self.artifacts:
            raise ValueError("Artifacts must be loaded before loading model")
            
        # Recreate model architecture
        self.model = HybridRecommender(
            num_users=self.artifacts['num_users'],
            num_movies=self.artifacts['num_movies_ml'],
            num_genres=self.artifacts['num_genres'],
            st_embedding_dim=self.artifacts['st_embedding_dim'],
            num_directors=self.artifacts['num_directors'],
            num_actors=self.artifacts['num_actors'],
            person_embedding_dim=16,  # Must match training
            num_numerical_features=self.artifacts['num_numerical_features'],
            content_mlp_hidden_dims=[128, 64],  # Must match training
            collaborative_embedding_dim=64,  # Must match training
            main_mlp_hidden_dims=[256, 128, 64],  # Must match training
            dropout_rate=0.0  # Disable dropout for inference
        )
        
        # Load state dict
        if not self.device:
            self.device = torch.device('cpu')
            
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded from file successfully!")
    
    def _encode_genres(self, genres_str: str) -> np.ndarray:
        """Encode genres using the trained binarizer."""
        if not self.artifacts:
            raise ValueError("Artifacts not loaded")
            
        # Handle pipe-separated or comma-separated genres
        if '|' in genres_str:
            genres_list = genres_str.split('|')
        else:
            genres_list = genres_str.split(',')
            
        # Clean up genres
        genres_list = [g.strip() for g in genres_list if g.strip()]
        
        # Filter out unknown genres to prevent warnings
        known_genres = set(self.artifacts['genre_labels'])
        filtered_genres = [g for g in genres_list if g in known_genres]
        
        # Use the same binarizer from training
        mlb = self.artifacts['genre_binarizer']
        encoded = mlb.transform([filtered_genres])
        return encoded[0].astype(np.float32)
    
    def _encode_persons(self, person_nconsts: List[str], person_type: str) -> np.ndarray:
        """Encode director or actor nconsts to indices."""
        if not self.artifacts:
            raise ValueError("Artifacts not loaded")
            
        if person_type == 'director':
            person_to_idx = self.artifacts['director_to_idx']
            max_per_movie = self.artifacts['max_directors_per_movie']
        else:  # actor
            person_to_idx = self.artifacts['actor_to_idx']
            max_per_movie = self.artifacts['max_actors_per_movie']
            
        pad_token_id = self.artifacts['PAD_TOKEN_ID']
        
        # Map nconsts to indices
        mapped_ids = [person_to_idx.get(nconst, pad_token_id) for nconst in person_nconsts]
        
        # Truncate or pad
        if len(mapped_ids) > max_per_movie:
            mapped_ids = mapped_ids[:max_per_movie]
        else:
            mapped_ids += [pad_token_id] * (max_per_movie - len(mapped_ids))
            
        return np.array(mapped_ids, dtype=np.int64)
    
    def _normalize_numerical(self, runtime: float, year: float) -> np.ndarray:
        """Normalize numerical features using the trained scaler."""
        if not self.artifacts:
            raise ValueError("Artifacts not loaded")
            
        scaler = self.artifacts['numerical_scaler']
        # Create a DataFrame with proper feature names to avoid sklearn warnings
        numerical_df = pd.DataFrame([[runtime, year]], columns=['runtimeMinutes', 'startYear'])
        scaled = scaler.transform(numerical_df)
        return scaled[0].astype(np.float32)
    
    def predict_for_existing_movies(self, user_id: int, movie_ids: List[int]) -> List[Dict]:
        """
        Make predictions for existing movies in the database.
        """
        if not self.model or not self.artifacts:
            raise ValueError("Model and artifacts must be loaded")
            
        # Check if user exists in training data
        if user_id not in self.artifacts['user_to_idx']:
            raise ValueError(f"User ID {user_id} not found in training data")
            
        user_idx = self.artifacts['user_to_idx'][user_id]
        
        # Get database session
        db = SessionLocal()
        try:
            predictions = []
            
            for movie_id in movie_ids:
                # Check if movie exists in training data
                if movie_id not in self.artifacts['movie_to_idx']:
                    print(f"⚠️  Movie ID {movie_id} not in training data, skipping...")
                    continue
                    
                movie_idx = self.artifacts['movie_to_idx'][movie_id]
                
                # Get movie features from database
                movie = db.query(Movie).filter(Movie.id == movie_id).first()
                if not movie:
                    print(f"⚠️  Movie ID {movie_id} not found in database, skipping...")
                    continue
                
                # Use pre-computed features if available
                if hasattr(self.artifacts, 'movie_features') and movie_id in self.artifacts['movie_features']:
                    features = self.artifacts['movie_features'][movie_id]
                    genres_encoded = features['genres_encoded']
                    title_embedding = features['title_st_embedding'] 
                    director_ids = features['director_ids_encoded']
                    actor_ids = features['actor_ids_encoded']
                    numerical_scaled = features['numerical_scaled']
                else:
                    # Compute features on the fly
                    genres_encoded = self._encode_genres(movie.genres or "")
                    title_embedding = self.st_model.encode(movie.title, convert_to_numpy=True)
                    
                    # Get directors and actors from database
                    directors = db.query(Person.id).join(MoviePerson).filter(
                        MoviePerson.movie_id == movie_id,
                        MoviePerson.category == 'director'
                    ).all()
                    director_nconsts = [d.id for d in directors]
                    director_ids = self._encode_persons(director_nconsts, 'director')
                    
                    actors = db.query(Person.id).join(MoviePerson).filter(
                        MoviePerson.movie_id == movie_id,
                        MoviePerson.category.in_(['actor', 'actress'])
                    ).all()
                    actor_nconsts = [a.id for a in actors]
                    actor_ids = self._encode_persons(actor_nconsts, 'actor')
                    
                    numerical_scaled = self._normalize_numerical(
                        movie.runtime_minutes or 0.0,
                        movie.start_year or 0.0
                    )
                
                # Prepare tensors
                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                movie_tensor = torch.LongTensor([movie_idx]).to(self.device)
                genres_tensor = torch.FloatTensor([genres_encoded]).to(self.device)
                title_tensor = torch.FloatTensor([title_embedding]).to(self.device)
                director_tensor = torch.LongTensor([director_ids]).to(self.device)
                actor_tensor = torch.LongTensor([actor_ids]).to(self.device)
                numerical_tensor = torch.FloatTensor([numerical_scaled]).to(self.device)
                
                # Make prediction
                with torch.no_grad():
                    prediction = self.model(
                        user_ids=user_tensor,
                        movie_ids_collaborative=movie_tensor,
                        genres_input=genres_tensor,
                        title_st_embedding_input=title_tensor,
                        director_ids=director_tensor,
                        actor_ids=actor_tensor,
                        numerical_inputs=numerical_tensor
                    )
                    
                    predicted_rating = torch.clamp(prediction, 0.5, 5.0).item()
                    
                predictions.append({
                    'movie_id': movie_id,
                    'title': movie.title,
                    'predicted_rating': round(predicted_rating, 2)
                })
                    
            return predictions
            
        finally:
            db.close()
    
    def predict_for_unseen_movie(self, 
                                user_id: int,
                                movie_title: str,
                                movie_genres: str,
                                directors: List[str],
                                actors: List[str],
                                runtime: float,
                                year: float) -> float:
        """
        Make prediction for a new/unseen movie using raw content features.
        """
        if not self.model or not self.artifacts or not self.st_model:
            raise ValueError("Model, artifacts, and sentence transformer must be loaded")
            
        # Check if user exists in training data
        if user_id not in self.artifacts['user_to_idx']:
            raise ValueError(f"User ID {user_id} not found in training data")
            
        user_idx = self.artifacts['user_to_idx'][user_id]
        dummy_movie_idx = 0  # Use dummy for collaborative filtering part
        
        # Encode features
        genres_encoded = self._encode_genres(movie_genres)
        title_embedding = self.st_model.encode(movie_title, convert_to_numpy=True)
        director_ids = self._encode_persons(directors, 'director')
        actor_ids = self._encode_persons(actors, 'actor')
        numerical_scaled = self._normalize_numerical(runtime, year)
        
        # Prepare tensors
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        movie_tensor = torch.LongTensor([dummy_movie_idx]).to(self.device)
        genres_tensor = torch.FloatTensor([genres_encoded]).to(self.device)
        title_tensor = torch.FloatTensor([title_embedding]).to(self.device)
        director_tensor = torch.LongTensor([director_ids]).to(self.device)
        actor_tensor = torch.LongTensor([actor_ids]).to(self.device)
        numerical_tensor = torch.FloatTensor([numerical_scaled]).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(
                user_ids=user_tensor,
                movie_ids_collaborative=movie_tensor,
                genres_input=genres_tensor,
                title_st_embedding_input=title_tensor,
                director_ids=director_tensor,
                actor_ids=actor_tensor,
                numerical_inputs=numerical_tensor
            )
            
            predicted_rating = torch.clamp(prediction, 0.5, 5.0).item()
            
        return round(predicted_rating, 2)


# Global predictor instance for the FastAPI server
predictor = None

def load_prediction_artifacts():
    """Load prediction artifacts for the FastAPI server."""
    global predictor
    try:
        predictor = MovieRecommenderPredictor()
        predictor.load_artifacts()
        predictor.load_sentence_transformer()
        
        # Try MLflow first, then fallback to direct file
        mlflow_success = predictor.load_model_from_mlflow()
        if not mlflow_success:
            # Try local model paths
            model_paths = [
                'models/trained_model.pth',
                '../models/trained_model.pth',
                'mlflow_server/models/trained_model.pth'
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    predictor.load_model_from_file(model_path)
                    break
            else:
                raise FileNotFoundError("No trained model found")
                
        print("✅ Prediction system ready!")
        
    except Exception as e:
        print(f"❌ Failed to load prediction system: {e}")
        predictor = None


def prepare_prediction_data(db: Session, user_id: int, movie_ids: List[int]) -> pd.DataFrame:
    """
    Prepare prediction data in the format expected by the model's predict method.
    This is used by the FastAPI server.
    """
    if not predictor:
        raise RuntimeError("Prediction system not loaded")
        
    predictions = predictor.predict_for_existing_movies(user_id, movie_ids)
    
    # Convert to DataFrame format expected by model.predict()
    data = []
    for pred in predictions:
        # This would need to be implemented based on the exact format
        # expected by the model's predict method
        pass
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    # Demo usage
    print("🎬 Movie Recommender Prediction Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = MovieRecommenderPredictor()
    
    # Load artifacts
    predictor.load_artifacts()
    
    # Load sentence transformer
    predictor.load_sentence_transformer()
    
    # Try to load model
    print("\n📡 Attempting to load model from MLflow...")
    mlflow_success = predictor.load_model_from_mlflow()
    
    if not mlflow_success:
        print("\n📁 Attempting to load model from file...")
        model_paths = [
            'models/trained_model.pth',
            '../models/trained_model.pth', 
            'mlflow_server/models/trained_model.pth'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                predictor.load_model_from_file(model_path)
                break
        else:
            print("❌ No trained model found. Please train a model first.")
            exit(1)
    
    # Demo 1: Predict for unseen movie
    print("\n--- DEMO 1: Predicting rating for a NEW/UNSEEN movie ---")
    try:
        predicted_rating = predictor.predict_for_unseen_movie(
            user_id=1,
            movie_title="Echoes of the Quantum Realm",
            movie_genres="Sci-Fi|Action|Adventure", 
            directors=["nm0000229"],  # Christopher Nolan
            actors=["nm3918035", "nm3501799"],  # Zendaya, Timothée Chalamet
            runtime=150.0,
            year=2024.0
        )
        
        print(f"✅ Predicted rating: {predicted_rating}")
        
    except Exception as e:
        print(f"❌ Demo 1 failed: {e}")
    
    # Demo 2: Predict for existing movies
    print("\n--- DEMO 2: Predicting ratings for existing movies ---")
    try:
        predictions = predictor.predict_for_existing_movies(
            user_id=1,
            movie_ids=[1, 2, 3, 4, 5]  # Sample movie IDs
        )
        
        print("✅ Predictions for existing movies:")
        for pred in predictions[:5]:  # Show top 5
            print(f"   - {pred['title']}: {pred['predicted_rating']}")
            
    except Exception as e:
        print(f"❌ Demo 2 failed: {e}")
    
    print("\n�� Demo completed!")
