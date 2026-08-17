import logging
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.data_preprocessing import DataPreprocessing, DataPreprocessingConfig
from database import engine, get_db, Base
from models import PredictionLog

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Mute the noisy logs from the data_preprocessing module
# It will now only log WARNINGs or ERRORs, skipping the INFO level spam.
logging.getLogger("src.data_preprocessing").setLevel(logging.WARNING)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment from customer reviews.",
    version="1.0.0"
)

# Mount static files and templates directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Path to the trained model pipeline
MODEL_PATH = Path("models/best_model.joblib")
model = None
preprocessor = None

@app.on_event("startup")
def load_resources():
    """Load the ML model, database, and perform a warm-up request."""
    global model, preprocessor
    
    # 1. Create database tables if they don't exist
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database connected and tables verified/created successfully.")
    except Exception as e:
        logger.exception("Failed to connect to the database or create tables.")

    # 2. Load ML model and preprocessor
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info("Trained pipeline loaded successfully.")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Please run main.py first.")
            
        prep_config = DataPreprocessingConfig(review_column="body", cleaned_column="clean_body")
        preprocessor = DataPreprocessing(prep_config)
        
        # 3. WARM-UP (Fixes the first-request latency)
        # We run a dummy text through the preprocessor and model right now.
        # This forces all heavy NLP libraries to load into memory BEFORE the server starts.
        if model is not None and preprocessor is not None:
            logger.info("Performing model warm-up to prevent first-request latency...")
            dummy_df = pd.DataFrame({"body": ["this is a warm up text"]})
            dummy_clean = preprocessor.preprocess_data(dummy_df)
            dummy_text = dummy_clean["clean_body"].iloc[0] if not dummy_clean.empty else "this is a warm up text"
            model.predict([dummy_text])
            logger.info("Model warm-up complete. API is ready for fast responses!")
            
    except Exception as e:
        logger.exception("Failed to load application resources.")


class ReviewRequest(BaseModel):
    review: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main frontend HTML interface."""
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.post("/predict")
async def predict_sentiment(payload: ReviewRequest, db: Session = Depends(get_db)):
    """Endpoint to clean text, process through pipeline, return predicted sentiment, and save to DB."""
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model is not loaded. Please run 'python main.py' to generate the model."
        )
    
    text = payload.review.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    try:
        # Preprocess text (Will be fast now, and no longer spam your console)
        temp_df = pd.DataFrame({"body": [text]})
        cleaned_df = preprocessor.preprocess_data(temp_df)
        
        if cleaned_df.empty:
            processed_text = text.lower()  
        else:
            processed_text = cleaned_df["clean_body"].iloc[0]

        # Make prediction
        prediction = model.predict([processed_text])[0]
        
        # Calculate confidence/probability
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([processed_text])[0]
            confidence = float(max(probs))

        # Save to Neon DB
        db_log = PredictionLog(
            user_query=text,
            sentiment=str(prediction),
            confidence=confidence
        )
        db.add(db_log)
        db.commit()
        db.refresh(db_log)

        logger.info(f"Prediction successful & saved to DB (ID: {db_log.id}): {prediction} for text: '{processed_text}'")
        
        return {
            "review": text,
            "cleaned_review": processed_text,
            "sentiment": prediction,
            "confidence": round(confidence * 100, 2) if confidence else None
        }

    except Exception as e:
        logger.exception("Error during prediction inference or database storage.")
        raise HTTPException(status_code=500, detail=str(e))