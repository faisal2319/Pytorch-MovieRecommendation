
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tqdm import tqdm
import os
import pickle
from typing import Tuple, Dict, List

# --- Constants ---
PAD_TOKEN_ID = 0

# --- Feature Engineering Functions ---

def create_person_mappings_and_features(
    df_principals: DataFrame, 
    role: str, 
    max_per_movie: int, 
    ml_imdb_id_map: dict
) -> Tuple[Dict, int, Dict]:
    """
    Maps IMDb nconsts to contiguous integers for a given role and creates padded ID lists.
    """
    if role == 'actor_or_actress':
        role_principals = df_principals[df_principals['category'].isin(['actor', 'actress'])].copy()
    else:
        role_principals = df_principals[df_principals['category'] == role].copy()
    
    relevant_tconsts = set(ml_imdb_id_map.values())
    role_principals = role_principals[role_principals['tconst'].isin(list(relevant_tconsts))]
    
    all_nconsts = role_principals['nconst'].dropna().unique()
    person_to_idx = {nconst: idx + 1 for idx, nconst in enumerate(sorted(list(all_nconsts)))}
    num_persons = len(all_nconsts) + 1

    grouped: Series = role_principals.groupby('tconst')['nconst'].apply(list)
    grouped_dict = grouped.to_dict()

    ml_to_person_ids = {}
    for ml_id, imdb_id in tqdm(ml_imdb_id_map.items(), desc=f"Mapping {role}s"):
        nconsts = grouped_dict.get(imdb_id, [])
        mapped = [person_to_idx.get(n, PAD_TOKEN_ID) for n in nconsts][:max_per_movie]
        mapped += [PAD_TOKEN_ID] * (max_per_movie - len(mapped))
        ml_to_person_ids[ml_id] = mapped

    return person_to_idx, num_persons, ml_to_person_ids

def prepare_genre_features(df: DataFrame) -> Tuple[MultiLabelBinarizer, Series]:
    """Creates a multi-hot encoding for genres."""
    # The genres in the database are comma-separated, not pipe-separated.
    all_genres: set[str] = set(g for genres in df['genres'].dropna() for g in genres.split(','))
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_genres)])
    
    encoded_genres: Series = df['genres'].apply(
        lambda g: mlb.transform([g.split(',') if g else []])[0].astype(np.float32)
    )
    return mlb, encoded_genres

def prepare_title_embeddings(df: DataFrame, model_name: str) -> Tuple[Dict[str, np.ndarray], int]:
    """Generates sentence embeddings for movie titles."""
    print(f"Encoding titles with Sentence Transformer ({model_name})...")
    st_model = SentenceTransformer(model_name)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    st_model.to(device)

    unique_titles = df[['imdb_id', 'primaryTitle']].drop_duplicates()
    title_list: List[str] = unique_titles['primaryTitle'].fillna("").tolist()
    imdb_ids: List[str] = unique_titles['imdb_id'].tolist()
    
    embeddings: np.ndarray = st_model.encode(title_list, show_progress_bar=True, convert_to_numpy=True)
    st_dim = embeddings.shape[1]
    
    title_map: Dict[str, np.ndarray] = {imdb_ids[i]: embeddings[i] for i in range(len(imdb_ids))}
    return title_map, st_dim

def prepare_numerical_features(df: DataFrame) -> Tuple[StandardScaler, np.ndarray]:
    """Scales numerical features using StandardScaler."""
    num_feats: DataFrame = df[['runtimeMinutes', 'startYear']].astype(float)
    num_feats.fillna(num_feats.mean(), inplace=True)
    scaler = StandardScaler()
    scaled_features: np.ndarray = scaler.fit_transform(num_feats)
    return scaler, scaled_features 