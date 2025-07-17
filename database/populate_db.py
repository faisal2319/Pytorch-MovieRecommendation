#!/usr/bin/env python3
"""
Populates the PostgreSQL database with movie and user data from the
MovieLens 1M and IMDb datasets.
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from sqlalchemy import inspect

# Add src to path to import database settings
from database import SessionLocal, User, Movie, Rating, Person, MoviePerson, get_primary_key_columns

def clean_df(df):
    """Replace all '\\N' strings with numpy NaN for consistency."""
    return df.replace({'\\N': np.nan})

def populate_table(session, df, model_class, chunk_size=10000):
    """
    Generic function to bulk insert a DataFrame into a table, avoiding duplicates.
    It reads existing data from the table and only inserts new records.
    """
    print(f"Populating {model_class.__tablename__}...")
    
    pk_cols = get_primary_key_columns(model_class)
    merge_on_cols = []  # Initialize to empty list

    if not pk_cols:
        # No primary key, cannot check for duplicates.
        # This path is not expected for this script.
        records_to_add = [model_class(**record) for record in df.to_dict(orient='records')]
    else:
        # Check for duplicates against the DB
        try:
            existing_df = pd.read_sql(session.query(model_class).statement, session.bind)
        except Exception as e:
            # This can happen if the table doesn't exist yet on the first run
            if "does not exist" in str(e):
                existing_df = pd.DataFrame(columns=df.columns)
            else:
                raise e

        # Determine the columns to merge and deduplicate on.
        merge_on_cols = [col for col in pk_cols if col in df.columns]

        if existing_df.empty:
            df_to_insert = df
        else:
            # We must use only the PKs that are in both the df and the table
            if not merge_on_cols:
                raise ValueError("No common primary key columns found to merge on.")
            
            # Ensure the columns also exist in the table from the DB, which they should.
            if not all(col in existing_df.columns for col in merge_on_cols):
                 raise ValueError(f"Primary key columns {merge_on_cols} not found in database table.")

            merged = df.merge(existing_df[merge_on_cols].copy(), on=merge_on_cols, how='left', indicator=True)
            df_to_insert = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

        # Final safety check for duplicates within the batch to be inserted.
        if merge_on_cols:
            df_to_insert.drop_duplicates(subset=merge_on_cols, inplace=True)
        
        records_to_add = [model_class(**record) for record in df_to_insert.to_dict(orient='records')]

    if not records_to_add:
        print(f"Table {model_class.__tablename__} is already up-to-date.")
        return

    # Insert in chunks to manage memory
    for i in tqdm(range(0, len(records_to_add), chunk_size), desc=f"Inserting into {model_class.__tablename__}"):
        chunk = records_to_add[i:i + chunk_size]
        try:
            session.bulk_save_objects(chunk)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error on chunk {i}: {e}. Trying records one-by-one.")
            for record in chunk:
                try:
                    session.add(record)
                    session.commit()
                except Exception as inner_e:
                    session.rollback() # Skip individual records that fail
    print(f"Finished populating {model_class.__tablename__}.")


def populate_all_data(base_data_dir):
    """Main function to orchestrate the population of all tables."""
    session = SessionLocal()
    
    movielens_dir = os.path.join(base_data_dir, 'movielens/ml-1m')
    imdb_dir = os.path.join(base_data_dir, 'imdb')

    try:
        # --- 1. Populate Users ---
        users_df = pd.read_csv(os.path.join(movielens_dir, 'users.dat'), sep='::', header=None, names=['id', 'gender', 'age', 'occupation', 'zipcode'], engine='python')
        populate_table(session, users_df[['id']], User)

        # --- 2. Populate Movies (with IMDb ID mapping) ---
        ml_movies_df = pd.read_csv(os.path.join(movielens_dir, 'movies.dat'), sep='::', header=None, names=['id', 'title', 'genres'], engine='python', encoding='latin-1')
        
        # Limit IMDB movies to 100k for performance
        imdb_basics_df = pd.read_csv(os.path.join(imdb_dir, 'title.basics.tsv'), sep='\t', nrows=200000) # Read more to ensure matches
        imdb_basics_df = clean_df(imdb_basics_df)
        imdb_basics_df = imdb_basics_df[imdb_basics_df['titleType'] == 'movie'].head(100000) # Limit to 100k movies

        ml_movies_df['year_from_title'] = ml_movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        ml_movies_df['title_clean'] = ml_movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip().str.lower()
        imdb_basics_df['title_lower'] = imdb_basics_df['primaryTitle'].str.lower()
        imdb_basics_df['startYear'] = pd.to_numeric(imdb_basics_df['startYear'], errors='coerce')
        imdb_basics_df['runtimeMinutes'] = pd.to_numeric(imdb_basics_df['runtimeMinutes'], errors='coerce')
        movies_merged_df = pd.merge(
            ml_movies_df,
            imdb_basics_df[['tconst', 'title_lower', 'startYear', 'runtimeMinutes']],
            left_on=['title_clean', 'year_from_title'],
            right_on=['title_lower', 'startYear'],
            how='left'
        )
        movies_to_populate = movies_merged_df[['id', 'title', 'genres', 'tconst', 'startYear', 'runtimeMinutes']].rename(columns={'tconst': 'imdb_id', 'startYear': 'start_year', 'runtimeMinutes': 'runtime_minutes'})
        movies_to_populate = movies_to_populate.dropna(subset=['imdb_id'])
        movies_to_populate = movies_to_populate.drop_duplicates(subset=['id'])
        populate_table(session, movies_to_populate, Movie)

        # --- 3. Populate Ratings ---
        ratings_df = pd.read_csv(os.path.join(movielens_dir, 'ratings.dat'), sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp_unix'], engine='python')
        ratings_df['timestamp'] = ratings_df['timestamp_unix'].apply(datetime.datetime.fromtimestamp)
        
        # Filter ratings to only include movies that exist in our database
        existing_movie_ids = {row[0] for row in session.query(Movie.id).all()}
        ratings_df = ratings_df[ratings_df['movie_id'].isin(existing_movie_ids)]
        
        # Sort by timestamp to ensure we keep the LATEST rating for any duplicates
        ratings_df.sort_values('timestamp_unix', ascending=True, inplace=True)
        # Drop duplicates from the source data, keeping the last (most recent) entry
        ratings_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last', inplace=True)
        populate_table(session, ratings_df[['user_id', 'movie_id', 'rating', 'timestamp']], Rating)

        # --- 4. Identify and Populate Persons and MoviePersons ---
        # First, get the movies we have in DB to find principals
        movie_id_map = {row.id: row.imdb_id for row in session.query(Movie.id, Movie.imdb_id).all() if row.imdb_id}
        valid_movie_tconsts = set(movie_id_map.values())

        print(f"Processing principals for {len(valid_movie_tconsts)} movies...")
        
        # Load principals with a reasonable limit and filter by our movies
        principals_df = pd.read_csv(os.path.join(imdb_dir, 'title.principals.tsv'), sep='\t', nrows=500000)  # Limit rows
        principals_df = clean_df(principals_df)
        principals_df = principals_df[principals_df['tconst'].isin(list(valid_movie_tconsts))]
        
        if len(principals_df) == 0:
            print("No principals found for our movies. Skipping persons and movie_persons population.")
            print("\n✅ Core tables populated successfully!")
            return
        
        # Now we know which persons we need
        required_person_ids = set(principals_df['nconst'].unique())
        print(f"Found {len(required_person_ids)} unique persons to process...")

        # --- Populate Persons (filtered) ---
        # Process persons file in chunks to avoid memory issues
        chunk_size = 100000
        persons_list = []
        
        print("Processing persons file in chunks...")
        for chunk in pd.read_csv(os.path.join(imdb_dir, 'name.basics.tsv'), sep='\t', chunksize=chunk_size):
            chunk = clean_df(chunk)
            filtered_chunk = chunk[chunk['nconst'].isin(list(required_person_ids))]
            if len(filtered_chunk) > 0:
                persons_list.append(filtered_chunk)
            
            # Break early if we've found all required persons
            if len(persons_list) > 0:
                found_ids = set()
                for df in persons_list:
                    found_ids.update(df['nconst'].tolist())
                if len(found_ids) >= len(required_person_ids):
                    print(f"Found all required persons ({len(found_ids)})! Stopping chunk processing.")
                    break
        
        if not persons_list:
            print("No matching persons found. Skipping persons population.")
            print("\n✅ Core tables populated successfully!")
            return
            
        persons_df = pd.concat(persons_list, ignore_index=True)
        persons_df = persons_df[['nconst', 'primaryName', 'birthYear', 'deathYear', 'primaryProfession']]
        persons_df = persons_df.rename(columns={'nconst': 'id', 'primaryName': 'name', 'birthYear': 'birth_year', 'deathYear': 'death_year', 'primaryProfession': 'primary_profession'})
        persons_df['birth_year'] = pd.to_numeric(persons_df['birth_year'], errors='coerce').astype('Int64')
        persons_df['death_year'] = pd.to_numeric(persons_df['death_year'], errors='coerce').astype('Int64')
        populate_table(session, persons_df.dropna(subset=['id']), Person)

        # --- Populate MoviePersons ---
        valid_person_ids = {row[0] for row in session.query(Person.id).all()}
        principals_df = principals_df[principals_df['nconst'].isin(list(valid_person_ids))]
        
        # Create a reverse map from tconst -> movieId for insertion
        tconst_to_movie_id = {v: k for k, v in movie_id_map.items()}
        principals_df['movie_id'] = principals_df['tconst'].map(tconst_to_movie_id)
        
        principals_to_populate = principals_df[['movie_id', 'nconst', 'category', 'ordering', 'job', 'characters']]
        principals_to_populate = principals_to_populate.rename(columns={'nconst': 'person_id'})
        principals_to_populate['ordering'] = pd.to_numeric(principals_to_populate['ordering'], errors='coerce').astype('Int64')
        
        populate_table(session, principals_to_populate.dropna(subset=['movie_id', 'person_id']), MoviePerson)

        print("\n✅ All tables populated successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred during population: {e}")
        session.rollback()
    finally:
        session.close()


if __name__ == '__main__':
    # Adjust the path to be relative to the script's location in `src`
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # Optional: A simple check to ensure the db is ready, useful when run with docker-compose
    from setup_database import wait_for_db
    if wait_for_db():
        populate_all_data(data_dir)
    else:
        sys.exit("Database was not ready, aborting population.") 