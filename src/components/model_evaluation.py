import os
import sys
import numpy as np
import pandas as pd
from sklearn.calibration import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, top_k_accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class ModelEvaluationConfig:
    metrics_table_output_path: str = os.path.join('artifacts', 'metrics_table.csv')
    roc_plot_output_path: str = os.path.join('artifacts')

class ModelEvaluation:
  def __init__(self):
    self.evaluation_config = ModelEvaluationConfig()

  def evaluate_models(self, X_train, y_train, X_test, y_test, model, params, model_path):
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
            
            class_names = model.classes_
            
            auc_score = roc_auc_score(y_test, y_test_prob, multi_class='ovo')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            top_k_acc = top_k_accuracy_score(y_test, y_test_prob, k=model.n_neighbors)
            
            report[metric] = {
                'model': model,
                'auc': auc_score,
                'f1': f1,
                'top_k_acc': top_k_acc,
            }
            
            logging.info(f"Best {metric} model found")
            logging.info(f"{metric} parameters: {model.get_params()}")
            
            save_object(
                file_path=os.path.join(model_path, f"{metric}_model.pkl"),
                obj=model
            )
            
            self.plot_roc_curve(y_test, y_test_prob, metric, class_names)

        self.save_metrics_table(report)
        
        logging.info("Model comparison completed successfully")
        return report
        
    except Exception as e:
        raise CustomException(e, sys)

  def plot_roc_curve(self, y_test, y_test_prob, metric, class_names):
    try:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        plt.figure(figsize=(10, 8))

        for i in range(y_test_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_prob[:, i])
            auc_score = roc_auc_score(y_test_bin[:, i], y_test_prob[:, i])
                
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_score:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {metric} metric')
        plt.legend(loc='lower right')

        plt.savefig(os.path.join(self.evaluation_config.roc_plot_output_path, f'{metric}_roc_curve.png'))
        plt.close()
            
        logging.info(f"ROC curve saved at {os.path.join(self.evaluation_config.roc_plot_output_path, f'{metric}_roc_curve.png')}")

    except Exception as e:
        raise CustomException(f"Error in plotting ROC curve: {str(e)}", sys)




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
