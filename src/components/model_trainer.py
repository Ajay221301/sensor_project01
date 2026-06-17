import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from collections import Counter


from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    artifact_folder= os.path.join(artifact_folder)
    trained_model_path= os.path.join(artifact_folder,"model.pkl" )
    expected_accuracy=0.45
    model_config_file_path= os.path.join('config','model.yaml')






class ModelTrainer:
    def __init__(self):
       


        self.model_trainer_config = ModelTrainerConfig()




        self.utils = MainUtils()


        self.models = {
            'XGBClassifier': ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', XGBClassifier(random_state=42))
            ]),
            'GradientBoostingClassifier': ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', GradientBoostingClassifier(random_state=42))
            ]),
            'SVC': ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', SVC(random_state=42))
            ]),
            'RandomForestClassifier': ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', RandomForestClassifier(random_state=42))
            ]),
        }


   
    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        try:
            report = {}
 
            for name, model in models.items():
                # model.fit() runs SMOTE internally on X_train/y_train only.
                # X_test/y_test are never touched by SMOTE - they remain
                # the real, untouched, imbalanced evaluation set.
                model.fit(X_train, y_train)
 
                y_pred = model.predict(X_test)
 
                report[name] = f1_score(y_test, y_pred)
 
            return report
 
        except Exception as e:
            raise CustomException(e, sys)

       
    def finetune_best_model(self, best_model_object, best_model_name, X_train, y_train,cv_folds):
        try:
            model_param_grid = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )["model_selection"]["model"][best_model_name]["search_param_grid"]
 
            grid_search = GridSearchCV(
                best_model_object,
                param_grid=model_param_grid,
                cv= cv_folds,
                n_jobs=-1,
                verbose=1,
                scoring='f1'
            )
 
            grid_search.fit(X_train, y_train)
 
            print("best params are:", grid_search.best_params_)
 
            return best_model_object.set_params(**grid_search.best_params_)
 
        except Exception as e:
            raise CustomException(e, sys)




    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input and target feature")
 
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]
 
            # Keep SMOTE's k_neighbors safe given this dataset's tiny minority class
            minority_count = min(Counter(y_train).values())
            cv_folds = min(5, minority_count)
            safe_k = max(1, minority_count - 2)
 
            for model in self.models.values():
                model.set_params(smote__k_neighbors=safe_k)
 
            model_report = self.evaluate_models(x_train, y_train, x_test, y_test, self.models)
            best_model_name = max(model_report, key=model_report.get)
 
            best_model = self.finetune_best_model(
                best_model_object=self.models[best_model_name],
                best_model_name=best_model_name,
                X_train=x_train,
                y_train=y_train,
                cv_folds=cv_folds
            )
 
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = f1_score(y_test, y_pred)
 
            print(f"best model name {best_model_name} and score: {best_model_score}")
 
            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No best model found with f1 score greater than the threshold {self.model_trainer_config.expected_accuracy}"
                )
 
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
 
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
 
            return self.model_trainer_config.trained_model_path,best_model_score
 
        except Exception as e:
            raise CustomException(e, sys)
