# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Movie Content Encoder ---
# This class takes all the raw content features and encodes them into a single vector.
class MovieContentEncoder(nn.Module):
    def __init__(self,
                 num_genres: int,
                 st_embedding_dim: int, # Dimension of Sentence Transformer embeddings
                 num_directors: int,
                 num_actors: int,
                 person_embedding_dim: int, # Embedding dim for directors/actors
                 num_numerical_features: int, # e.g., 2 for runtime, year
                 content_mlp_hidden_dims: list, # Hidden dims for combining content features
                 dropout_rate: float):
        super(MovieContentEncoder, self).__init__()

        # A. Genre Processing
        self.genre_mlp = nn.Sequential(
            nn.Linear(num_genres, content_mlp_hidden_dims[0] // 4),
            nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.genre_output_dim = content_mlp_hidden_dims[0] // 4

        # B. Text (Sentence Transformer Embedding) Processing
        # A simple MLP to learn a task-specific transformation of the general-purpose ST embeddings
        self.text_mlp = nn.Sequential(
            nn.Linear(st_embedding_dim, content_mlp_hidden_dims[0] // 2),
            nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.text_output_dim = content_mlp_hidden_dims[0] // 2

        # C. Person (Director/Actor) Embeddings
        self.director_embedding = nn.Embedding(num_directors, person_embedding_dim)
        self.actor_embedding = nn.Embedding(num_actors, person_embedding_dim)
        self.person_output_dim = person_embedding_dim

        # D. Numerical Features Processing
        self.numerical_linear = nn.Sequential(
            nn.Linear(num_numerical_features, content_mlp_hidden_dims[0] // 8),
            nn.ReLU()
        )
        self.numerical_output_dim = content_mlp_hidden_dims[0] // 8

        # E. Final Content Combination MLP
        total_content_input_dim = (
            self.genre_output_dim + self.text_output_dim +
            self.person_output_dim * 2 + # Multiply by 2 for director + actor
            self.numerical_output_dim
        )
        self.final_content_mlp = nn.Sequential(
            nn.Linear(total_content_input_dim, content_mlp_hidden_dims[0]),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(content_mlp_hidden_dims[0], content_mlp_hidden_dims[1]),
            nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.output_dim = content_mlp_hidden_dims[-1]

    def forward(self, genres_input, title_st_embedding_input, director_ids, actor_ids, numerical_inputs):
        encoded_genres = self.genre_mlp(genres_input)
        encoded_title = self.text_mlp(title_st_embedding_input)
        director_embs = self.director_embedding(director_ids)
        encoded_directors = director_embs.sum(dim=1)
        actor_embs = self.actor_embedding(actor_ids)
        encoded_actors = actor_embs.sum(dim=1)
        encoded_numerical = self.numerical_linear(numerical_inputs)

        combined_content_features = torch.cat([
            encoded_genres, encoded_title, encoded_directors, encoded_actors, encoded_numerical
        ], dim=-1)
        
        return self.final_content_mlp(combined_content_features)


# --- 2. Main Hybrid Recommender Model ---
# This class takes all parameters, uses some for itself, and passes the rest to MovieContentEncoder.
class HybridRecommender(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_movies: int, # Total unique MovieLens IDs seen in training
                 # --- Parameters to be passed down to MovieContentEncoder ---
                 num_genres: int,
                 st_embedding_dim: int,
                 num_directors: int,
                 num_actors: int,
                 person_embedding_dim: int,
                 num_numerical_features: int,
                 content_mlp_hidden_dims: list,
                 # --- General parameters for the main model ---
                 collaborative_embedding_dim: int = 64,
                 main_mlp_hidden_dims: list = [256, 128, 64],
                 dropout_rate: float = 0.2):
        super(HybridRecommender, self).__init__()

        self.user_embedding = nn.Embedding(num_users, collaborative_embedding_dim)
        self.movie_embedding_collaborative = nn.Embedding(num_movies, collaborative_embedding_dim)

        self.movie_content_encoder = MovieContentEncoder(
            num_genres=num_genres,
            st_embedding_dim=st_embedding_dim,
            num_directors=num_directors,
            num_actors=num_actors,
            person_embedding_dim=person_embedding_dim,
            num_numerical_features=num_numerical_features,
            content_mlp_hidden_dims=content_mlp_hidden_dims,
            dropout_rate=dropout_rate
        )

        main_mlp_input_dim = (
            collaborative_embedding_dim +
            collaborative_embedding_dim +
            self.movie_content_encoder.output_dim
        )
        self.main_mlp_layers = nn.ModuleList()
        in_dim = main_mlp_input_dim
        for h_dim in main_mlp_hidden_dims:
            self.main_mlp_layers.append(nn.Linear(in_dim, h_dim))
            self.main_mlp_layers.append(nn.ReLU())
            self.main_mlp_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        self.output_layer = nn.Linear(main_mlp_hidden_dims[-1], 1)

    def forward(self,
                user_ids,
                movie_ids_collaborative,
                genres_input,
                title_st_embedding_input,
                director_ids,
                actor_ids,
                numerical_inputs):
        
        user_emb = self.user_embedding(user_ids)
        movie_emb_collaborative = self.movie_embedding_collaborative(movie_ids_collaborative)

        movie_content_emb = self.movie_content_encoder(
            genres_input, title_st_embedding_input, director_ids, actor_ids, numerical_inputs
        )

        combined_features = torch.cat([
            user_emb,
            movie_emb_collaborative,
            movie_content_emb
        ], dim=-1)

        mlp_output = combined_features
        for layer in self.main_mlp_layers:
            mlp_output = layer(mlp_output)
        
        prediction = self.output_layer(mlp_output)
        return prediction.squeeze(1)
    
    def predict(self, data):
        """
        Prediction method for MLflow compatibility.
        Handles both pandas DataFrames and numpy arrays.
        """
        import pandas as pd
        import numpy as np
        
        self.eval()
        
        # Convert data to tensors
        with torch.no_grad():
            # Handle both pandas DataFrame and numpy array inputs
            if isinstance(data['user_ids'], pd.Series):
                # Pandas DataFrame case
                user_ids = torch.tensor(data['user_ids'].values, dtype=torch.long)
                movie_ids_collaborative = torch.tensor(data['movie_ids_collaborative'].values, dtype=torch.long)
                
                # Handle genres (convert from numpy arrays to tensor)
                genres_list = [np.array(g) for g in data['genres_input'].values]
                genres_input = torch.tensor(np.vstack(genres_list), dtype=torch.float32)
                
                # Handle title embeddings
                title_embeddings_list = [np.array(emb) for emb in data['title_st_embedding_input'].values]
                title_st_embedding_input = torch.tensor(np.vstack(title_embeddings_list), dtype=torch.float32)
                
                # Handle director and actor IDs
                director_ids_list = [np.array(d) for d in data['director_ids'].values]
                director_ids = torch.tensor(np.vstack(director_ids_list), dtype=torch.long)
                
                actor_ids_list = [np.array(a) for a in data['actor_ids'].values]
                actor_ids = torch.tensor(np.vstack(actor_ids_list), dtype=torch.long)
                
                # Handle numerical features
                numerical_list = [np.array(n) for n in data['numerical_inputs'].values]
                numerical_inputs = torch.tensor(np.vstack(numerical_list), dtype=torch.float32)
            else:
                # Numpy array case (direct input)
                user_ids = torch.tensor(data['user_ids'], dtype=torch.long)
                movie_ids_collaborative = torch.tensor(data['movie_ids_collaborative'], dtype=torch.long)
                genres_input = torch.tensor(data['genres_input'], dtype=torch.float32)
                title_st_embedding_input = torch.tensor(data['title_st_embedding_input'], dtype=torch.float32)
                director_ids = torch.tensor(data['director_ids'], dtype=torch.long)
                actor_ids = torch.tensor(data['actor_ids'], dtype=torch.long)
                numerical_inputs = torch.tensor(data['numerical_inputs'], dtype=torch.float32)
            
            # Make predictions using the forward method
            predictions = self.forward(
                user_ids=user_ids,
                movie_ids_collaborative=movie_ids_collaborative,
                genres_input=genres_input,
                title_st_embedding_input=title_st_embedding_input,
                director_ids=director_ids,
                actor_ids=actor_ids,
                numerical_inputs=numerical_inputs
            )
            
            # Clamp predictions to valid rating range and return as numpy array
            predictions_clamped = torch.clamp(predictions, 0.5, 5.0)
            return predictions_clamped.cpu().numpy()