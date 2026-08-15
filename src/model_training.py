import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    feature_column: str = "clean_body"
    target_column: str = "sentiment"
    rating_column: str = "rating"
    model_dir: str = "models"
    model_file_name: str = "best_model.joblib"


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def add_sentiment_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment column based on rating."""
        logger.info("Generating sentiment labels from %s", self.config.rating_column)
        
        if self.config.rating_column not in df.columns:
            raise ValueError(
                f"Rating column '{self.config.rating_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        def sentiment_label(x):
            if x >= 4:
                return "Positive"
            elif x == 3:
                return "Neutral"
            else:
                return "Negative"

        df[self.config.target_column] = df[self.config.rating_column].apply(sentiment_label)
        return df

    def split_data(self, df: pd.DataFrame):
        """Preprocess labels and split dataset into training and testing sets."""
        try:
            df = self.add_sentiment_label(df)
            
            logger.info("Splitting dataset into train and test sets")
            
            if self.config.feature_column not in df.columns or self.config.target_column not in df.columns:
                raise ValueError(
                    f"Columns '{self.config.feature_column}' or '{self.config.target_column}' "
                    f"not found in dataframe columns: {list(df.columns)}"
                )

            X = df[self.config.feature_column]
            y = df[self.config.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )

            logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.exception("Error occurred during data splitting")
            raise e

    def get_pipelines_and_grids(self):
        """Define explicit Pipelines and Hyperparameter Grids for multiple models."""
        
        # 1. Random Forest Pipeline & Grid
        rf_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True)),
            ("classifier", RandomForestClassifier(random_state=self.config.random_state, n_jobs=-1, class_weight="balanced"))
        ])
        rf_params = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [None, 10, 20]
        }

        # 2. Support Vector Classifier (SVC) Pipeline & Grid
        svc_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True)),
            ("classifier", SVC(random_state=self.config.random_state, probability=True, class_weight="balanced"))
        ])
        svc_params = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"]
        }

        # 3. Decision Tree Pipeline & Grid
        dt_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True)),
            ("classifier", DecisionTreeClassifier(random_state=self.config.random_state, class_weight="balanced"))
        ])
        dt_params = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "classifier__max_depth": [None, 10, 20],
            "classifier__criterion": ["gini", "entropy"]
        }

        models = {
            "Random Forest": (rf_pipeline, rf_params),
            "SVC": (svc_pipeline, svc_params),
            "Decision Tree": (dt_pipeline, dt_params)
        }

        return models

    def train_and_tune_pipelines(self, X_train, y_train, X_test, y_test):
        """Iterate through models, tune them via GridSearchCV, evaluate, and save the best overall pipeline."""
        try:
            models = self.get_pipelines_and_grids()
            best_overall_pipeline = None
            best_score = 0.0
            best_model_name = ""

            for name, (pipeline, params) in models.items():
                logger.info(f"Starting hyperparameter tuning for: {name}")
                
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=params,
                    cv=3,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)

                # Evaluate on test set
                y_pred = grid_search.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                logger.info(f"--- {name} Results ---")
                logger.info(f"Best Parameters: {grid_search.best_params_}")
                logger.info(f"Test Accuracy: {acc:.4f}")
                print(f"\nClassification Report for {name}:\n")
                print(classification_report(y_test, y_pred))

                # Track best performing model
                if acc > best_score:
                    best_score = acc
                    best_overall_pipeline = grid_search.best_estimator_
                    best_model_name = name

            logger.info(f"Best Overall Model Found: {best_model_name} with Accuracy: {best_score:.4f}")

            # Save the best fitted pipeline to disk
            model_dir_path = Path(self.config.model_dir)
            model_dir_path.mkdir(parents=True, exist_ok=True)
            file_path = model_dir_path / self.config.model_file_name

            logger.info(f"Saving best pipeline to {file_path}")
            joblib.dump(best_overall_pipeline, file_path)
            logger.info("Pipeline saved successfully")

            return best_overall_pipeline

        except Exception as e:
            logger.exception("Error occurred during multi-model training/tuning")
            raise e


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion, DataIngestionConfig
    from src.data_preprocessing import DataPreprocessing, DataPreprocessingConfig

    ingestion_config = DataIngestionConfig(raw_data_path="data/dataset.xlsx")
    data_ingestion = DataIngestion(ingestion_config)
    df = data_ingestion.load_data()

    preprocessing_config = DataPreprocessingConfig(review_column="body", cleaned_column="clean_body")
    data_preprocessing = DataPreprocessing(preprocessing_config)
    processed_df = data_preprocessing.preprocess_data(df)

    training_config = ModelTrainingConfig(
        feature_column="clean_body",
        target_column="sentiment",
        rating_column="rating",
        model_dir="models",
        model_file_name="best_model.joblib"
    )
    
    model_trainer = ModelTraining(training_config)
    X_train, X_test, y_train, y_test = model_trainer.split_data(processed_df)
    
    best_pipeline = model_trainer.train_and_tune_pipelines(X_train, y_train, X_test, y_test)