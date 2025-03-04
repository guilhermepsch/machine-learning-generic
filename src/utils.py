
import logging
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import f1_score, r2_score, roc_auc_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV
import os
import sys

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, model, params):
    try:
        report = {}
        for metric in ['euclidean', 'cosine']:
            param_grid = {**params, "metric": [metric]}
            gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
            gs.fit(X_train, y_train)

            best_model = model.__class__(**gs.best_params_)
            best_model.fit(X_train, y_train)

            y_test_pred = best_model.predict(X_test)

            auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            top_k_acc = top_k_accuracy_score(y_test, best_model.predict_proba(X_test), k=best_model.n_neighbors)

            report[metric] = {
                'model': best_model,
                'auc': auc,
                'f1': f1,
                'top_k_acc': top_k_acc,
            }
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)