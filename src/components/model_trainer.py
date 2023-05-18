import sys
import os
from dataclasses import dataclass
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split Train and Test Data has started')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestClassifier(random_state=42),
                "Decision Tree":DecisionTreeClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state = 42, solver='liblinear', max_iter = 1500),
                "Gradient Boosting":GradientBoostingClassifier(random_state=42),
                "KNeighbors Classifier":KNeighborsClassifier(),
                "Catboost Classifer":CatBoostClassifier(verbose=False),
                "Adaboost Classifier":AdaBoostClassifier(random_state=42)
            }
            params={
                "Random Forest":{
                    'n_estimators':[5,10,50,100],
                    'max_features':['sqrt','log2'],
                    'max_depth':[4,5,6],
                    'criterion':['gini','entropy']
                },
                "Decision Tree":{
                    'criterion':["gini", "entropy"],
                    'max_depth' :  [1, 2, 3, 10, 20],
                    'min_samples_leaf': [2, 5, 10, 50, 100]
                },
                "Logistic Regression":{
                    'C': np.logspace(-4,4,20),
                    'penalty':['l1','l2']
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [10, 50, 100, 250]
                },
                "KNeighbors Classifier":{
                    'n_neighbors':[5,7,9,11,13,15],
                    'weights': ['uniform','distance'],
                    'metric' :  ['minkowski','euclidean','manhattan']
                },
                "Catboost Classifer":{
                    "depth":[6,8,10],
                    "iterations":[30,50,100],
                    "learning_rate":[0.01,0.5,0.1]
                },
                "Adaboost Classifier":{
                    'learning_rate':[(0.97 + x / 100) for x in range(0, 8)],
                    'n_estimators':[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                    'algorithm':['SAMME','SAMME.R']
                }
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            auc_score=roc_auc_score(y_test,predicted)
            return predicted

        except Exception as e:
            raise Custom_Exception(e,sys)

