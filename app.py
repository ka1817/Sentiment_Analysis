import logging
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.data_preprocessing import DataPreprocessing, DataPreprocessingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment from customer reviews using a trained Pipeline.",
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
    """Load the trained scikit-learn pipeline and preprocessor during startup."""
    global model, preprocessor
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info("Trained pipeline loaded successfully.")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Please run main.py first.")
            
        # Initialize text preprocessor configuration matching training
        prep_config = DataPreprocessingConfig(review_column="body", cleaned_column="clean_body")
        preprocessor = DataPreprocessing(prep_config)
    except Exception as e:
        logger.exception("Failed to load application resources.")


class ReviewRequest(BaseModel):
    review: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main frontend HTML interface."""
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.post("/predict")
async def predict_sentiment(payload: ReviewRequest):
    """Endpoint to clean text, process through pipeline, and return predicted sentiment."""
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model is not loaded. Please run 'python main.py' to generate the model."
        )
    
    text = payload.review.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    try:
        # Step 1: Clean raw input text using preprocessing logic
        temp_df = pd.DataFrame({"body": [text]})
        cleaned_df = preprocessor.preprocess_data(temp_df)
        
        if cleaned_df.empty:
            processed_text = text.lower()  # Fallback if text is completely stripped
        else:
            processed_text = cleaned_df["clean_body"].iloc[0]

        # Step 2: Make prediction directly using the loaded pipeline (TfidfVectorizer + Classifier)
        prediction = model.predict([processed_text])[0]
        
        # Step 3: Extract prediction probabilities if available
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([processed_text])[0]
            confidence = float(max(probs))

        logger.info(f"Prediction successful: {prediction} for text: '{processed_text}'")
        return {
            "review": text,
            "cleaned_review": processed_text,
            "sentiment": prediction,
            "confidence": round(confidence * 100, 2) if confidence else None
        }

    except Exception as e:
        logger.exception("Error during prediction inference.")
        raise HTTPException(status_code=500, detail=str(e))