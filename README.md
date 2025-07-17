# 🎬 Hybrid Movie Recommendation System

A sophisticated movie recommendation system that combines collaborative filtering with content-based features using PyTorch. This system leverages both IMDB and MovieLens datasets to provide accurate movie recommendations through a hybrid neural network architecture.

## 🏗️ Architecture Overview

### Model Architecture

The system uses a **Hybrid Neural Network** that combines:

1. **Collaborative Filtering**: User and movie embeddings for collaborative patterns
2. **Content-Based Features**:
   - Movie genres (multi-label encoding)
   - Movie titles (Sentence Transformer embeddings)
   - Directors and actors (learned embeddings)
   - Numerical features (runtime, year)

### Key Components

- **MovieContentEncoder**: Processes raw movie features into a unified representation
- **HybridRecommender**: Main model that combines collaborative and content-based signals
- **MLflow Integration**: Model versioning and experiment tracking
- **FastAPI Server**: RESTful API for real-time predictions
- **PostgreSQL Database**: Stores users, movies, ratings, and batch recommendations

## 📊 Data Sources

### MovieLens Dataset

- **Ratings**: 1M+ user-movie ratings (1-5 scale)
- **Users**: User demographics and preferences
- **Movies**: Basic movie metadata

### IMDB Dataset

- **Movies**: Comprehensive movie information
- **Directors**: Movie director information
- **Actors**: Cast information with roles
- **Genres**: Detailed genre classifications

## 🔄 Batch Processing & Automation

The system is designed for **weekly batch processing** to handle new data:

### Current Batch Processing

- **Batch Recommendations**: Pre-computed top-N recommendations for all users
- **SQL-based**: Uses `batch_recommendations.sql` for efficient processing
- **Database Storage**: Results stored in `batch_recommendations` table

### Future Airflow Integration

Planned features for production deployment:

- **Weekly Data Ingestion**: Automated collection of new ratings
- **Model Retraining**: Scheduled model updates with new data
- **Recommendation Updates**: Automated batch recommendation generation
- **Performance Monitoring**: Track recommendation quality over time

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Task (for task automation)

### Option 1: Full Setup (with batch recommendations)

```bash
# Build and start all services
task build
task up
```

### Option 2: Setup without batch recommendations

```bash
# Start services without pre-computed recommendations
task up --NO_PRETRAINED=true
```

### Option 3: Stop all services

```bash
task down
```

## 🏃‍♂️ Running the Project

### 1. Start All Services

```bash
task up
```

This will:

- Start PostgreSQL database
- Start MLflow server (port 5001)
- Start FastAPI model server (port 8000)
- Initialize database with seed data
- Populate batch recommendations (unless `--no-pretrained` is used)

### 2. Access Services

- **FastAPI Server**: http://localhost:8000
- **MLflow UI**: http://localhost:5001
- **API Documentation**: http://localhost:8000/docs

### 3. Make Predictions

#### Real-time API Endpoints

```bash
# Get recommendations for existing movies
curl -X POST "http://localhost:8000/recommendations/existing" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_ids": [1, 2, 3]}'

# Predict rating for new movie
curl -X POST "http://localhost:8000/predict/new-movie" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "movie_title": "The Matrix",
    "genres": "Action,Sci-Fi",
    "directors": ["Lana Wachowski"],
    "actors": ["Keanu Reeves", "Laurence Fishburne"],
    "runtime": 136,
    "year": 1999
  }'
```

## 🧠 Model Training

### Training Configuration

```bash
cd mlflow_server
python train.py
```

### Key Training Parameters

- **Epochs**: 10 (configurable)
- **Batch Size**: 256
- **Learning Rate**: 0.001
- **Embedding Dimensions**:
  - Collaborative: 64
  - Person (directors/actors): 16
- **Architecture**:
  - Main MLP: [256, 128, 64]
  - Content MLP: [128, 64]

### Model Performance

- **RMSE**: ~0.8-0.9 on test set
- **Training Time**: ~10-15 minutes on CPU
- **Model Size**: ~50MB

## 📈 Batch Processing Details

### Current Implementation

The system includes batch processing capabilities:

1. **Pre-computed Recommendations**: Stored in PostgreSQL
2. **Efficient SQL Queries**: Fast retrieval of top-N recommendations
3. **User-based Filtering**: Excludes already-rated movies

### Batch Processing Script

```bash
# Run batch predictions
cd mlflow_server
python batch_predict.py
```

### Database Schema

- `users`: User information
- `movies`: Movie metadata
- `ratings`: User-movie ratings
- `batch_recommendations`: Pre-computed recommendations
- `movie_persons`: Director/actor relationships

## 🔧 Development Setup

### Manual Setup (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set up database
python database/setup_database.py

# Preprocess data
cd database
python preprocessing.py
python populate_db.py

# Train model
cd mlflow_server
python train.py
```

### Environment Variables

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=imdb_recommender
DB_USER=postgres
DB_PASSWORD=password123
MLFLOW_TRACKING_URI=http://localhost:5001
```

## 📁 Project Structure

```
pytorch/
├── database/                 # Database setup and preprocessing
│   ├── setup_database.py    # Database initialization
│   ├── preprocessing.py     # Data preprocessing
│   ├── populate_db.py       # Seed data population
│   └── model.py            # Neural network architecture
├── mlflow_server/           # Training and MLflow
│   ├── train.py            # Model training script
│   ├── batch_predict.py    # Batch prediction script
│   └── data_preprocessing.py # Data preprocessing
├── model_server/            # FastAPI prediction server
│   ├── main.py             # FastAPI application
│   ├── predict.py          # Prediction logic
│   └── routes.py           # API endpoints
├── docker-compose.yml       # Container orchestration
├── Taskfile.yml            # Task automation
└── init_db.sh              # Database initialization script
```

## 🔮 Future Enhancements

### Planned Airflow Integration

- **DAGs**: Weekly data processing workflows
- **Sensors**: Monitor for new data availability
- **Operators**: Automated model retraining
- **Monitoring**: Track recommendation quality metrics

