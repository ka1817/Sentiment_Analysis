import logging
from src.data_ingestion import DataIngestion, DataIngestionConfig
from src.data_preprocessing import DataPreprocessing, DataPreprocessingConfig
from src.model_training import ModelTraining, ModelTrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    try:
        logger.info("Starting End-to-End Pipeline Execution")

        # -------------------------------------------------------------
        # 1. Data Ingestion Phase
        # -------------------------------------------------------------
        logger.info("Step 1: Data Ingestion Started")
        ingestion_config = DataIngestionConfig(raw_data_path="data/dataset.xlsx")
        data_ingestion = DataIngestion(ingestion_config)
        
        raw_df = data_ingestion.load_data()
        logger.info(f"Data ingestion completed. Raw shape: {raw_df.shape}")

        # -------------------------------------------------------------
        # 2. Data Preprocessing Phase
        # -------------------------------------------------------------
        logger.info("Step 2: Data Preprocessing Started")
        preprocessing_config = DataPreprocessingConfig(
            review_column="body",
            cleaned_column="clean_body"
        )
        data_preprocessing = DataPreprocessing(preprocessing_config)
        
        processed_df = data_preprocessing.preprocess_data(raw_df)
        logger.info(f"Data preprocessing completed. Processed shape: {processed_df.shape}")

        # -------------------------------------------------------------
        # 3. Model Training, Tuning, and Pipeline Evaluation Phase
        # -------------------------------------------------------------
        logger.info("Step 3: Model Training, Tuning, and Selection Started")
        training_config = ModelTrainingConfig(
            test_size=0.2,
            random_state=42,
            feature_column="clean_body",
            target_column="sentiment",
            rating_column="rating",
            model_dir="models",
            model_file_name="best_model.joblib"
        )
        
        model_trainer = ModelTraining(training_config)
        
        # Split data (includes automatic sentiment label creation from ratings)
        X_train, X_test, y_train, y_test = model_trainer.split_data(processed_df)
        
        # Train and tune pipelines (Random Forest, SVC, Decision Tree), save best one
        best_pipeline = model_trainer.train_and_tune_pipelines(X_train, y_train, X_test, y_test)
        
        logger.info("Pipeline execution and model deployment preparation completed successfully!")

    except Exception as e:
        logger.exception("Pipeline failed due to an error.")
        raise e


if __name__ == "__main__":
    main()