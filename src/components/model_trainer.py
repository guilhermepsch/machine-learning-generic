import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score, f1_score, roc_auc_score, top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

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
      
      model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model=model, params=params)
      
      for metric, report in model_report.items():
        logging.info(f"Best {metric} found")
        logging.info(f"{metric} parameters: {report['model'].get_params()}")
        logging.info("Metrics:")
        logging.info(f"AUC: {report['auc']:.4f}, F1-Score: {report['f1']:.4f}, Top-K Accuracy: {report['top_k_acc']:.4f}")

        save_object(
            file_path=os.path.join(self.model_trainer_config.trained_model_file_path, f"{metric}_model.pkl"),
            obj=report['model']
        )
      return report['auc'], report['f1'], report['top_k_acc']
      
    except Exception as e:
        raise CustomException(e,sys)