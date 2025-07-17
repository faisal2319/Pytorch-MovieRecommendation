# src/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np

class HybridMovieDataset(Dataset):
    def __init__(self, users, movies_ml, ratings, genres, titles_st, directors, actors, numerical):
        # Convert numpy arrays from pickle to PyTorch Long/Float Tensors
        self.users = torch.LongTensor(users)
        self.movies_ml = torch.LongTensor(movies_ml)
        self.ratings = torch.FloatTensor(ratings)
        self.genres = torch.FloatTensor(genres)
        self.titles_st = torch.FloatTensor(titles_st)
        self.directors = torch.LongTensor(directors)
        self.actors = torch.LongTensor(actors)
        self.numerical = torch.FloatTensor(numerical)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # Return all features needed by the model's forward method
        return (self.users[idx], self.movies_ml[idx], self.genres[idx], 
                self.titles_st[idx], self.directors[idx], self.actors[idx], 
                self.numerical[idx], self.ratings[idx])

def load_processed_data_and_create_dataloaders(processed_data_dir='database/processed', batch_size=256):
    print("📦 Loading processed data for PyTorch Dataset...")
    
    # Load all data from the single pickle file
    input_filepath = os.path.join(processed_data_dir, 'all_processed_data.pkl')
    print(f"📁 Looking for data file: {input_filepath}")
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Processed data file not found: {input_filepath}. Please run data_preprocessing.py first.")
    
    print("📂 Loading pickle file...")
    with open(input_filepath, 'rb') as f:
        data = pickle.load(f)
    print("✅ Pickle file loaded successfully!")

    print("🔨 Creating training dataset...")
    train_dataset = HybridMovieDataset(
        data['train_users'], data['train_movies_ml'], data['train_ratings'], data['train_genres'],
        data['train_titles_st'], data['train_directors'], data['train_actors'], data['train_numerical']
    )
    print(f"✅ Training dataset created: {len(train_dataset):,} samples")
    
    print("🔨 Creating test dataset...")
    test_dataset = HybridMovieDataset(
        data['test_users'], data['test_movies_ml'], data['test_ratings'], data['test_genres'],
        data['test_titles_st'], data['test_directors'], data['test_actors'], data['test_numerical']
    )
    print(f"✅ Test dataset created: {len(test_dataset):,} samples")

    print("⚡ Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("✅ DataLoaders created successfully!")
    # Return train/test loaders and the full data dictionary for metadata/mappings
    return train_loader, test_loader, data

if __name__ == '__main__':
    # Example usage:
    train_loader, test_loader, data = load_processed_data_and_create_dataloaders()
    
    print(f"Num users: {data['num_users']}, Num ML movies: {data['num_movies_ml']}, Num genres: {data['num_genres']}")
    print(f"ST embedding dim: {data['st_embedding_dim']}")
    print(f"Num directors: {data['num_directors']}, Max directors per movie: {data['max_directors_per_movie']}")
    print(f"Num actors: {data['num_actors']}, Max actors per movie: {data['max_actors_per_movie']}")
    print(f"Num numerical features: {data['num_numerical_features']}")
    print(f"PAD_TOKEN_ID: {data['PAD_TOKEN_ID']}")

    for i, (users, movies_ml, genres, titles_st, directors, actors, numerical, ratings) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  users shape: {users.shape}")
        print(f"  movies_ml shape: {movies_ml.shape}")
        print(f"  genres shape: {genres.shape}")
        print(f"  titles_st shape: {titles_st.shape}")
        print(f"  directors shape: {directors.shape}")
        print(f"  actors shape: {actors.shape}")
        print(f"  numerical shape: {numerical.shape}")
        print(f"  ratings shape: {ratings.shape}")
        if i == 0:
            break