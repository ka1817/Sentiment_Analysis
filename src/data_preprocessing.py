import logging
import re
import string
from dataclasses import dataclass

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


logger = logging.getLogger(__name__)


@dataclass
class DataPreprocessingConfig:
    review_column: str = "body"
    cleaned_column: str = "clean_body"


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform all text preprocessing in one function:
        1. Validate review column
        2. Remove missing values
        3. Preserve original review
        4. Handle Hindi-English mixed text
        5. Convert text to lowercase
        6. Remove numbers
        7. Remove punctuation
        8. Remove English stopwords
        9. Lemmatize English words
        10. Remove empty cleaned reviews
        """

        try:
            logger.info("Starting data preprocessing")

            review_column = self.config.review_column
            cleaned_column = self.config.cleaned_column

            # Check if review column exists
            if review_column not in df.columns:
                raise ValueError(
                    f"Column '{review_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

            # Create a copy
            df = df.copy()

            # Remove missing reviews
            df = df.dropna(subset=[review_column])

            # Convert reviews to string
            df[review_column] = df[review_column].astype(str)

            # Keep original review
            df["original_body"] = df[review_column]

            # Load stopwords
            stop_words = set(stopwords.words("english"))

            # Keep important sentiment words
            important_words = {"not", "no", "nor", "never", "very"}

            stop_words = stop_words - important_words

            # Initialize lemmatizer
            lemmatizer = WordNetLemmatizer()

            # Create cleaned reviews
            cleaned_reviews = []

            for text in df[review_column]:


                # Preserve English, Hindi, numbers, spaces
                # and common punctuation
                text = re.sub(
                    r'[^a-zA-Z0-9\u0900-\u097F\s.,!?\'"()-]',
                    " ",
                    text
                )

                # Remove extra spaces
                text = re.sub(
                    r"\s+",
                    " ",
                    text
                ).strip()

                # Convert to lowercase
                text = text.lower()

                # Remove numbers
                text = re.sub(r"\d+", "", text)

                # Remove punctuation
                text = text.translate(
                    str.maketrans(
                        "",
                        "",
                        string.punctuation
                    )
                )

                # Remove extra spaces again
                text = re.sub(
                    r"\s+",
                    " ",
                    text
                ).strip()

                # Split into words
                words = text.split()

                # Remove English stopwords
                words = [
                    word
                    for word in words
                    if word not in stop_words
                ]

                # Lemmatize words
                words = [
                    lemmatizer.lemmatize(word)
                    for word in words
                ]

                # Join words back into text
                cleaned_text = " ".join(words)

                cleaned_reviews.append(cleaned_text)

            # Add cleaned reviews to dataframe
            df[cleaned_column] = cleaned_reviews

            # Remove empty cleaned reviews
            df = df[
                df[cleaned_column].str.strip() != ""
            ].copy()

            logger.info(
                "Data preprocessing completed successfully. "
                "Final shape: %s",
                df.shape
            )

            return df

        except Exception:
            logger.exception(
                "Error occurred during data preprocessing"
            )
            raise


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion, DataIngestionConfig

    # Load data
    ingestion_config = DataIngestionConfig(
        raw_data_path="data/dataset.xlsx"
    )

    data_ingestion = DataIngestion(ingestion_config)

    df = data_ingestion.load_data()

    # Preprocess data
    preprocessing_config = DataPreprocessingConfig(
        review_column="body",
        cleaned_column="clean_body"
    )

    data_preprocessing = DataPreprocessing(
        preprocessing_config
    )

    processed_df = data_preprocessing.preprocess_data(df)

    # Display results
    print(
        processed_df[
            [
                "original_body",
                "body",
                "clean_body"
            ]
        ].head(10)
    )

    print("\nFinal Dataset Shape:", processed_df.shape)
    print("Final columns:",processed_df.columns)