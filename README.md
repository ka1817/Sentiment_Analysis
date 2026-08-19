### Sentiment-Analysis# 🎭 Sentiment Analysis API

An end-to-end Machine Learning web application that classifies text reviews as **Positive, Negative, or Neutral**. This project features a robust FastAPI backend, a responsive HTML/CSS frontend, and a fully automated CI/CD pipeline using Docker, GitHub Actions, and Render. 

---

## 🚀 Overview

This application serves a trained Machine Learning model via a RESTful API. User interactions and analysis results are securely stored in a cloud-hosted PostgreSQL database (Neon DB). The repository is configured with a continuous deployment pipeline that automatically builds and pushes a Docker image to Docker Hub, subsequently triggering a live deployment on Render.

## 🛠️ Tech Stack

*   **Machine Learning:** Scikit-learn, TensorFlow, NLTK, Imbalanced-learn, Pandas, NumPy
*   **Backend & API:** FastAPI, Uvicorn, SQLAlchemy, Psycopg2-binary
*   **Frontend:** HTML, CSS, Jinja2 Templates
*   **Database:** Neon DB (PostgreSQL)
*   **DevOps & Deployment:** Docker, GitHub Actions, Render

## 📂 Project Structure

Based on the repository layout, here is the core structure:

*   **`.github/workflows/`**: Contains `cd.yml` for CI/CD automation.
*   **`notebook/`**: Jupyter notebooks (`experiments.ipynb`) for EDA, model training, and evaluation.
*   **`models/` & `data/`**: Stored datasets and serialized ML models (e.g., joblib/h5 files).
*   **`templates/` & `static/`**: Frontend assets (`index.html`) and styling.
*   **`app.py` / `main.py`**: FastAPI application entry points and route definitions.
*   **`database.py` & `models.py`**: SQLAlchemy database connection and schema definitions.

## ⚙️ Local Setup & Installation

**1. Clone and Configure Environment**
Clone the repository and create a `.env` file in the root directory to store your Neon DB credentials:
`DATABASE_URL=postgresql://user:password@endpoint.neon.tech/dbname`

**2. Install Dependencies**
Install the required Python packages and download the necessary NLTK corpora:
`pip install -r requirements.txt`
`python -m nltk.downloader stopwords punkt wordnet omw-1.4`

**3. Run the Application locally**
Start the FastAPI server:
`uvicorn app:app --host 0.0.0.0 --port 8000 --reload`
Navigate to `http://localhost:8000` to access the web interface.

**4. Run with Docker**
Alternatively, build and run the container locally:
`docker build -t sentiment-analysis .`
`docker run -p 8000:8000 --env-file .env sentiment-analysis`