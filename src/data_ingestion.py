import os
import logging
from dataclasses import dataclass

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class DataIngestionConfig:
    raw_data_path: str = "data/dataset.xlsx"


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.config.raw_data_path):
                raise FileNotFoundError(
                    f"Data file not found: {self.config.raw_data_path}"
                )

            logger.info(
                "Loading data from %s",
                self.config.raw_data_path
            )

            df = pd.read_excel(self.config.raw_data_path)

            logger.info(
                "Data loaded successfully. Shape: %s",
                df.shape
            )

            return df

        except Exception as e:
            logger.exception("Error occurred while loading data")
            raise e


if __name__ == "__main__":
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)

    df = data_ingestion.load_data()

    print(df.head())
    print(f"\nDataset shape: {df.shape}")