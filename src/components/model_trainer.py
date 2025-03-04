import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join("artifacts")
  
class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()
    
  def initiate_model_trainer(self, train_array, test_array):
    try:
      logging.info("Split training and test input data")
      X_train, y_train, X_test, y_test = (
        train_array[:,:-1],
        train_array[:,-1],
        test_array[:,:-1],
        test_array[:,-1],
      )
      
      model = KNeighborsClassifier()
      params = {
        "n_neighbors": list(range(1, 16))
      }
      
      return (
        X_train,
        y_train,
        X_test,
        y_test,
        model,
        params,
        self.model_trainer_config.trained_model_file_path
      )
      
    except Exception as e:
        raise CustomException(e,sys)