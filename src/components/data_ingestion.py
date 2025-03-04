import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
  train_data_path: str=os.path.join('artifacts','train.csv')
  test_data_path: str=os.path.join('artifacts','test.csv')
  raw_data_path: str=os.path.join('artifacts','data.pkl')
  treated_data_path: str=os.path.join('artifacts','data.csv')
  
class DataIngestion:
  def __init__(self):
      self.ingestion_config=DataIngestionConfig()
      
  def flatten_data(self, data):
    records = []
    
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                records.append({
                    'syndrome_id': syndrome_id,
                    'subject_id': subject_id,
                    'image_id': image_id,
                    **{f"dim_{i}": embedding[i] for i in range(len(embedding))}
                })
    
    df = pd.DataFrame(records)
    return df      

  def initiate_data_ingestion(self):
    logging.info("Entered the data ingestion method or component")
    try:
        logging.info('Loading pickle data')
        data = load_object(self.ingestion_config.raw_data_path)
        
        logging.info("Flattening data structure")
        df = self.flatten_data(data)

        logging.info("Performing train-test split")
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        df.to_csv(self.ingestion_config.treated_data_path, index=False, header=True)
        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

        logging.info("Ingestion of the data is completed")

        return (
            df,
            train_set,
            test_set,
        )
    except Exception as e:
        raise CustomException(e, sys)
  