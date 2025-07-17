# src/data_preprocessing.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.preprocessing import (
    create_person_mappings_and_features,
    prepare_genre_features,
    prepare_title_embeddings,
    prepare_numerical_features,
    PAD_TOKEN_ID
)

def preprocess_all_data(
    movielens_dir='data/movielens/ml-1m',
    imdb_dir='data/imdb',
    processed_data_dir='data/processed',
    max_directors_per_movie=3,
    max_actors_per_movie=5,
    st_model_name='all-MiniLM-L6-v2',
    test_size=0.2,
    random_state=42
):
    os.makedirs(processed_data_dir, exist_ok=True)

    # 1. Load raw data
    print("Loading raw data...")
    ratings_df = pd.read_csv(
        os.path.join(movielens_dir, 'ratings.dat'),
        sep='::', header=None, names=['userId', 'movieId', 'rating', 'timestamp'],
        engine='python', encoding='latin-1'
    )
    movies_df = pd.read_csv(
        os.path.join(movielens_dir, 'movies.dat'),
        sep='::', header=None, names=['movieId', 'title', 'genres'],
        engine='python', encoding='latin-1'
    )
    # The original script didn't use user data, but loading it as per your instruction.
    users_df = pd.read_csv(
        os.path.join(movielens_dir, 'users.dat'),
        sep='::', header=None, names=['userId', 'gender', 'age', 'occupation', 'zipcode'],
        engine='python', encoding='latin-1'
    )
    imdb_basics = pd.read_csv(os.path.join(imdb_dir, 'title.basics.tsv'), sep='\t', low_memory=False)
    imdb_principals = pd.read_csv(os.path.join(imdb_dir, 'title.principals.tsv'), sep='\t', low_memory=False)

    # 2. Clean IMDb basics
    print("Cleaning IMDb basics...")
    imdb_basics = imdb_basics.replace({'\\N': np.nan})
    imdb_basics = imdb_basics[
        (imdb_basics['titleType'] == 'movie') &
        imdb_basics['genres'].notna() &
        (imdb_basics['genres'] != '\\N') &
        imdb_basics['runtimeMinutes'].notna() &
        imdb_basics['startYear'].notna()
    ].copy()
    imdb_basics['runtimeMinutes'] = pd.to_numeric(imdb_basics['runtimeMinutes'], errors='coerce')
    imdb_basics['startYear'] = pd.to_numeric(imdb_basics['startYear'], errors='coerce').astype('Int64')
    
    # 3. Map MovieLens ID -> IMDb tconst
    print("Mapping MovieLens IDs to IMDb IDs...")
    # New mapping: match titles and years from MovieLens with IMDb data
    movies_df['year_from_title'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies_df['title_clean'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()

    imdb_to_match = imdb_basics[['tconst', 'primaryTitle', 'startYear']].copy()
    imdb_to_match['startYear'] = pd.to_numeric(imdb_to_match['startYear'], errors='coerce')

    # Normalize titles to lowercase for better matching
    movies_df['title_lower'] = movies_df['title_clean'].str.lower()
    imdb_to_match['title_lower'] = imdb_to_match['primaryTitle'].str.lower()

    merged = pd.merge(
        movies_df,
        imdb_to_match,
        left_on=['title_lower', 'year_from_title'],
        right_on=['title_lower', 'startYear'],
        how='inner'
    )

    ml_to_imdb = merged.set_index('movieId')['tconst'].to_dict()
    num_movielens_movies = movies_df['movieId'].nunique()
    print(f"Successfully mapped {len(ml_to_imdb)} of {num_movielens_movies} MovieLens movies to IMDb IDs.")


    # 4. Merge ratings with IMDb basics
    print("Merging ratings with IMDb metadata...")
    df = ratings_df.copy()
    df['imdb_id'] = df['movieId'].map(ml_to_imdb)
    df.dropna(subset=['imdb_id'], inplace=True)
    df = df.merge(
        imdb_basics[['tconst','primaryTitle','genres','runtimeMinutes','startYear']],
        left_on='imdb_id', right_on='tconst', how='inner'
    )
    df.drop(columns=['tconst'], inplace=True)
    print(f"Linked {len(df)} ratings to IMDb movie data.")

    # 5. Build user/movie mappings
    print("Building user/movie mappings...")
    users = df['userId'].unique()
    movies_ml = df['movieId'].unique()
    user_to_idx = {u: i for i, u in enumerate(sorted(users))}
    movie_to_idx_ml = {m: i for i, m in enumerate(sorted(movies_ml))}
    num_users = len(users)
    num_movies_ml = len(movies_ml)

    # 6A. Process genres (multi-hot)
    print("Processing genres...")
    mlb, df['genres_encoded'] = prepare_genre_features(df.rename(columns={'genres_x': 'genres'}))
    num_genres = len(mlb.classes_)

    # 6B. Sentence Transformer for titles
    title_map, st_dim = prepare_title_embeddings(
        df.rename(columns={'primaryTitle': 'primaryTitle'}), st_model_name
    )
    df['title_st_embedding'] = df['imdb_id'].map(title_map).apply(
        lambda x: x if x is not None else np.zeros(st_dim, dtype=np.float32)
    )

    # 6C. Director/Actor features
    print("Processing directors...")
    director_to_idx, num_directors, ml_to_dirs = \
        create_person_mappings_and_features(imdb_principals, 'director',
                                            max_directors_per_movie, ml_to_imdb)
    df['director_ids_encoded'] = df['movieId'].map(ml_to_dirs)

    print("Processing actors...")
    actor_to_idx, num_actors, ml_to_actors = \
        create_person_mappings_and_features(imdb_principals, 'actor_or_actress',
                                            max_actors_per_movie, ml_to_imdb)
    df['actor_ids_encoded'] = df['movieId'].map(ml_to_actors)

    # 6D. Numerical features
    print("Scaling numerical features...")
    scaler, scaled_features = prepare_numerical_features(
        df.rename(columns={'runtimeMinutes_x': 'runtimeMinutes', 'startYear_x': 'startYear'})
    )
    df['numerical_scaled'] = list(scaled_features)
    num_numerical = scaled_features.shape[1]

    # 7. Train-test split
    print("Splitting train and test sets...")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # 8. Save everything
    print("Saving processed data...")
    data = {
        # Training tensors
        'train_users': train_df['userId'].map(user_to_idx).values,
        'train_movies_ml': train_df['movieId'].map(movie_to_idx_ml).values,
        'train_ratings': train_df['rating'].values,
        'train_genres': np.vstack(train_df['genres_encoded'].values),
        'train_titles_st': np.vstack(train_df['title_st_embedding'].values),
        'train_directors': np.vstack(train_df['director_ids_encoded'].tolist()),
        'train_actors': np.vstack(train_df['actor_ids_encoded'].tolist()),
        'train_numerical': np.vstack(train_df['numerical_scaled'].tolist()),

        # Testing tensors
        'test_users': test_df['userId'].map(user_to_idx).values,
        'test_movies_ml': test_df['movieId'].map(movie_to_idx_ml).values,
        'test_ratings': test_df['rating'].values,
        'test_genres': np.vstack(test_df['genres_encoded'].values),
        'test_titles_st': np.vstack(test_df['title_st_embedding'].values),
        'test_directors': np.vstack(test_df['director_ids_encoded'].tolist()),
        'test_actors': np.vstack(test_df['actor_ids_encoded'].tolist()),
        'test_numerical': np.vstack(test_df['numerical_scaled'].tolist()),

        # Metadata and mappings
        'num_users': num_users,
        'num_movies_ml': num_movies_ml,
        'num_genres': num_genres,
        'st_embedding_dim': st_dim,
        'num_directors': num_directors,
        'num_actors': num_actors,
        'num_numerical_features': num_numerical,
        'max_directors_per_movie': max_directors_per_movie,
        'max_actors_per_movie': max_actors_per_movie,
        'PAD_TOKEN_ID': PAD_TOKEN_ID,

        'user_to_idx': user_to_idx,
        'movie_to_idx': movie_to_idx_ml,
        'genre_binarizer': mlb,
        'genre_labels': mlb.classes_.tolist(),
        'director_to_idx': director_to_idx,
        'actor_to_idx': actor_to_idx,
        'numerical_scaler': scaler,
        'ml_id_to_imdb_tconst_map': ml_to_imdb
    }

    with open(os.path.join(processed_data_dir, 'all_processed_data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    print("Data preprocessing complete. Saved to data/processed/all_processed_data.pkl")
    
    # 9. Create and save a smaller data file specifically for prediction
    print("Creating and saving prediction-optimized data file...")
    
    # Consolidate all movie features from train/test splits into one DataFrame
    all_movies_df = pd.concat([train_df, test_df]).drop_duplicates(subset=['movieId'])
    
    prediction_data = {
        'movie_features': all_movies_df[[
            'movieId', 'genres_encoded', 'title_st_embedding', 
            'director_ids_encoded', 'actor_ids_encoded', 'numerical_scaled'
        ]].set_index('movieId').to_dict('index'),
        
        'user_to_idx': user_to_idx,
        'movie_to_idx': movie_to_idx_ml,
        'genre_binarizer': mlb,
        'genre_labels': mlb.classes_.tolist(),
        'director_to_idx': director_to_idx,
        'actor_to_idx': actor_to_idx,
        'numerical_scaler': scaler,
        'st_embedding_dim': st_dim,
        'PAD_TOKEN_ID': PAD_TOKEN_ID,
        'max_directors_per_movie': max_directors_per_movie,
        'max_actors_per_movie': max_actors_per_movie,
    }
    
    artifacts_output_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'prediction_artifacts.pkl')
    with open(artifacts_output_path, 'wb') as f:
        pickle.dump(prediction_data, f)
        
    print(f"✅ Prediction-optimized data file saved to: {artifacts_output_path}")


if __name__ == '__main__':
    # Adjust paths to be relative to the project root, not the script's location
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    preprocess_all_data(
        movielens_dir=os.path.join(project_root, 'mlflow_server', 'data', 'movielens', 'ml-1m'),
        imdb_dir=os.path.join(project_root, 'mlflow_server', 'data', 'imdb'),
        processed_data_dir=os.path.join(project_root, 'mlflow_server', 'data', 'processed')
    )