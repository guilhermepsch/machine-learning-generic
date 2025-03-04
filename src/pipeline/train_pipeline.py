import os
import sys
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging

def main():
    try:
        logging.info("Starting Data Ingestion")
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        logging.info(f"Train data: {train_data}, Test data: {test_data}")
    except Exception as e:
        raise CustomException(f"Error in Data Ingestion: {str(e)}", sys)

    try:
        logging.info("Starting Data Transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

        logging.info(f"Data Transformation completed. Preprocessor saved at {preprocessor_path}")
    except Exception as e:
        raise CustomException(f"Error in Data Transformation: {str(e)}", sys)

    try:
        logging.info("Starting Model Training")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Model Training completed with R2 Score: {r2_score}")
    except Exception as e:
        raise CustomException(f"Error in Model Training: {str(e)}", sys)

if __name__ == "__main__":
    main()
