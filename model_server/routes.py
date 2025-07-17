from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database.database import SessionLocal, Movie, Rating, BatchRecommendation, User
from model_server.limiter import limiter

router = APIRouter()

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PredictionRequest(BaseModel):
    user_id: int

# Health check
@router.get("/")
async def read_root():
    return {"status": "alive"}

# Get movies for a user
@router.get("/movies/{user_id}")
@limiter.limit("10/minute")
async def get_movies(request: Request, user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    user_movies = db.query(Rating).filter(Rating.user_id == user_id).all()
    movie_ids = [rating.movie_id for rating in user_movies]
    movies = db.query(Movie).filter(Movie.id.in_(movie_ids)).all()
    movie_dict = {movie.id: movie for movie in movies}
    rating_dict = {rating.movie_id: rating.rating for rating in user_movies}

    movie_list = []
    for movie_id in movie_ids:
        movie = movie_dict.get(movie_id)
        if movie:
            movie_list.append({
                "id": movie.id,
                "title": movie.title,
                "genres": movie.genres,
                "start_year": movie.start_year,
                "runtime_minutes": movie.runtime_minutes,
                "user_rating": rating_dict.get(movie_id)
            })
    return {"movies": movie_list}

# Predict endpoint with rate limiting
@router.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, req: PredictionRequest, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.id == req.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User {req.user_id} not found")

        batch_recs = db.query(BatchRecommendation).filter(
            BatchRecommendation.user_id == req.user_id
        ).order_by(BatchRecommendation.rank).all()

        if not batch_recs:
            return {
                "message": f"No batch recommendations found for user {req.user_id}",
                "predictions": [],
                "total_recommendations": 0
            }

        movie_ids = [rec.movie_id for rec in batch_recs]
        movies = db.query(Movie).filter(Movie.id.in_(movie_ids)).all()
        movie_dict = {movie.id: movie for movie in movies}

        predictions = []
        for rec in batch_recs:
            movie = movie_dict.get(rec.movie_id)
            if movie:
                predictions.append({
                    "movie_id": rec.movie_id,
                    "movie_title": movie.title,
                    "movie_genres": movie.genres,
                    "predicted_rating": rec.predicted_rating,
                    "rank": rec.rank,
                    "generated_at": rec.generated_at.isoformat()
                })

        return {
            "predictions": predictions,
            "total_recommendations": len(predictions),
            "user_id": req.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 