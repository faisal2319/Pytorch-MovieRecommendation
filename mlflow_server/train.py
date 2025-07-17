import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import sys

# Add the project root to the Python path to allow imports from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import load_processed_data_and_create_dataloaders
from database.model import HybridRecommender
import mlflow
import mlflow.pytorch


def train_model(num_epochs=10,             
                batch_size=256,
                learning_rate=0.001,
                collaborative_embedding_dim=64,
                person_embedding_dim=16,
                main_mlp_hidden_dims=[256, 128, 64],
                content_mlp_hidden_dims=[128, 64],
                dropout_rate=0.2,
                processed_data_dir='database/processed'
                ):
    
    print("🚀 Starting training process...")
    print(f"📊 Training Configuration:")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Learning Rate: {learning_rate}")
    print(f"   - Collaborative Embedding Dim: {collaborative_embedding_dim}")
    print(f"   - Person Embedding Dim: {person_embedding_dim}")
    
    # Set environment variable to use proxy for artifacts
    print("🔧 Configuring MLflow...")
    
    # Check if we're running inside the container
    is_container = os.path.exists('/.dockerenv')
    
    if is_container:
        print("🐳 Running inside container - using internal MLflow server")
        # Inside container, use the service name and internal port
        os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
        tracking_uri = "http://mlflow:5000"
        # Don't set artifacts URI - let MLflow server handle it
    else:
        print("💻 Running locally - using host-mapped MLflow server")
        # Outside container, use localhost with mapped port
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
        tracking_uri = "http://localhost:5001"
        # Set local artifact root for local runs
        local_artifacts_dir = os.path.join(os.getcwd(), "mlflow", "artifacts")
        os.makedirs(local_artifacts_dir, exist_ok=True)
        os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = f"file://{local_artifacts_dir}"
    
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✅ MLflow tracking URI set to: {tracking_uri}")
    
    print("🖥️  Setting up device...")
    # Proper device detection for macOS (MPS) and other platforms
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("✅ Using CPU")
    print(f"Device: {device}")

    print("📂 Loading processed data and creating data loaders...")
    print(f"   - Data directory: {processed_data_dir}")
    train_loader, test_loader, data = load_processed_data_and_create_dataloaders(processed_data_dir=processed_data_dir, batch_size=batch_size)
    print(f"✅ Data loaded successfully!")
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    print(f"   - Users: {data['num_users']}")
    print(f"   - Movies: {data['num_movies_ml']}")
    print(f"   - Genres: {data['num_genres']}")
    print(f"   - Directors: {data['num_directors']}")
    print(f"   - Actors: {data['num_actors']}")
    
    print("🔬 Starting MLflow experiment...")
    with mlflow.start_run():
        print("📝 Logging parameters to MLflow...")
        mlflow.log_params({
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'collaborative_embedding_dim': collaborative_embedding_dim,
            'person_embedding_dim': person_embedding_dim,
            'device': str(device),
            'is_container': is_container
        })
        
        print("🏗️  Building model architecture...")
        model = HybridRecommender(
            num_users=data['num_users'],
            num_movies=data['num_movies_ml'],
            num_genres=data['num_genres'],
            st_embedding_dim=data['st_embedding_dim'],
            num_directors=data['num_directors'],
            num_actors=data['num_actors'],
            person_embedding_dim=person_embedding_dim,
            num_numerical_features=data['num_numerical_features'],
            content_mlp_hidden_dims=content_mlp_hidden_dims,
            collaborative_embedding_dim=collaborative_embedding_dim,
            main_mlp_hidden_dims=main_mlp_hidden_dims,
            dropout_rate=dropout_rate
        ).to(device)
        print(f"✅ Model created and moved to {device}")
        print(f"📈 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        print("⚙️  Setting up training components...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print("✅ Loss function and optimizer ready")

        print(f"🏃 Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (users, movies_ml, genres, titles_st, directors, actors, numerical, ratings) in enumerate(train_loader):
                # Move all batch tensors to the appropriate device (CPU/GPU/MPS)
                users, movies_ml, genres, titles_st, directors, actors, numerical, ratings = users.to(device), movies_ml.to(device), genres.to(device), titles_st.to(device), directors.to(device), actors.to(device), numerical.to(device), ratings.to(device)
                
                predictions = model(
                    user_ids=users,
                    movie_ids_collaborative=movies_ml,
                    genres_input=genres,
                    title_st_embedding_input=titles_st,
                    director_ids=directors,
                    actor_ids=actors,
                    numerical_inputs=numerical
                )

                loss = criterion(predictions, ratings)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Print loss every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            avg_train_loss = train_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            print(f"📊 Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
            
            # Eval
            print(f"🔍 Running evaluation for epoch {epoch+1}...")
            model.eval()
            all_predictions = []
            all_ratings = []
            with torch.no_grad():
                for users, movies_ml, genres, titles_st, directors, actors, numerical, ratings in test_loader:
                    users, movies_ml, genres, titles_st, directors, actors, numerical, ratings = users.to(device), movies_ml.to(device), genres.to(device), titles_st.to(device), directors.to(device), actors.to(device), numerical.to(device), ratings.to(device)

                    predictions = model(
                    user_ids=users,
                    movie_ids_collaborative=movies_ml,
                    genres_input=genres,
                    title_st_embedding_input=titles_st,
                    director_ids=directors,
                    actor_ids=actors,
                    numerical_inputs=numerical
                )

                predictions_clamped = torch.clamp(predictions, 0.5, 5)

                all_predictions.extend(predictions_clamped.cpu().numpy().tolist())
                all_ratings.extend(ratings.cpu().numpy().tolist())

            all_predictions = np.array(all_predictions)
            all_ratings = np.array(all_ratings)

            rmse = np.sqrt(mean_squared_error(all_predictions, all_ratings))
            mlflow.log_metric("test_rmse", rmse, step=epoch)
            print(f"🎯 Epoch {epoch+1}/{num_epochs}, Test RMSE: {rmse:.4f}")

        print("💾 Saving model...")
        try:
            # Log the model with proper PyTorch format
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                code_paths=["database/model.py", "dataset.py"],  # Include dependencies
                pip_requirements=["torch", "numpy", "scikit-learn"],
                registered_model_name="hybrid-recommender"
            )
            print("✅ Model saved to MLflow successfully!")
        except Exception as e:
            print(f"⚠️  MLflow model saving failed: {e}")
            print(f"Error details: {str(e)}")
            # Fallback: save model locally
            local_model_path = "models/trained_model.pth"
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), local_model_path)
            print(f"✅ Model saved locally to: {local_model_path}")
            print("📊 Training metrics are still logged to MLflow")


if __name__ == "__main__":
    train_model()




