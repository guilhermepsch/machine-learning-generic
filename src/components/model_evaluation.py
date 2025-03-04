import os
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.utils import save_object

@dataclass
class ModelEvaluationConfig:
    metrics_table_output_path: str = os.path.join('artifacts', 'metrics_table.csv')

class ModelEvaluation:
  def __init__(self):
        self.evaluation_config = ModelEvaluationConfig()

  def evaluate_models(self, X_train, y_train, X_test, y_test, model, params, path):
    try:
            
        report = {}
        for metric in ['euclidean', 'cosine']:
            param_grid = {**params, "metric": [metric]}
            gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
            gs.fit(X_train, y_train)

            model = model.__class__(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)
            
            auc_score = roc_auc_score(y_test, y_test_prob, multi_class='ovr')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            top_k_acc = top_k_accuracy_score(y_test, y_test_prob, k=model.n_neighbors)
            
            report[metric] = {
                'model': model,
                'auc': auc_score,
                'f1': f1,
                'top_k_acc': top_k_acc,
            }
            
            save_object(
                file_path=os.path.join(path, f"{metric}_model.pkl"),
                obj=model
            )
        self.save_metrics_table(report)
        
        logging.info("Model comparison completed successfully")
        return report
        
    except Exception as e:
        raise CustomException(e, sys)

  def save_metrics_table(self, report):
    try:
        logging.info("Saving metrics table")
        metrics_data = {
            'Metric': [],
            'AUC': [],
            'F1-Score': [],
            'Top-K Accuracy': []
        }
          
        for metric, values in report.items():
            metrics_data['Metric'].append(metric)
            metrics_data['AUC'].append(values['auc'])
            metrics_data['F1-Score'].append(values['f1'])
            metrics_data['Top-K Accuracy'].append(values['top_k_acc'])

        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(self.evaluation_config.metrics_table_output_path, index=False)
            
        logging.info(f"Metrics table saved at {self.evaluation_config.metrics_table_output_path}")
    except Exception as e:
        raise CustomException(f"Error in saving metrics table: {str(e)}", sys)
