import numpy as np
import pandas as pd
import sys
import os
import dill
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from src.exception import Custom_Exception
from sklearn.metrics import roc_auc_score
import pickle
from src.exception import Custom_Exception

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise Custom_Exception(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            gs = GridSearchCV(model, para, cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = roc_auc_score(y_train, y_train_pred)
            test_model_score = roc_auc_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise Custom_Exception(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise Custom_Exception(e,sys)