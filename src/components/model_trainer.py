from collections import Counter
from dataclasses import dataclass
import os
import sys
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class ModelTrainerConfig:
    artifact_folder: str = os.path.join(artifact_folder)
    trained_model_path: str = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy: float = 0.45
    model_config_file_path: str = os.path.join("config", "model.yaml")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        # Standard SMOTE resampler instance
        self.resampler = SMOTE(random_state=42)

        self.models = {
            "XGBClassifier": ImbPipeline(
                [
                    ("smote", self.resampler),
                    (
                        "clf",
                        XGBClassifier(
                            random_state=42,
                            eval_metric="logloss",
                            scale_pos_weight=1.5,
                        ),
                    ),
                ]
            ),
            "GradientBoostingClassifier": ImbPipeline(
                [
                    ("smote", self.resampler),
                    ("clf", GradientBoostingClassifier(random_state=42)),
                ]
            ),
            "SVC": ImbPipeline(
                [
                    ("smote", self.resampler),
                    (
                        "clf",
                        SVC(
                            random_state=42,
                            probability=True,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            "RandomForestClassifier": ImbPipeline(
                [
                    ("smote", self.resampler),
                    (
                        "clf",
                        RandomForestClassifier(
                            random_state=42, class_weight="balanced"
                        ),
                    ),
                ]
            ),
        }

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        try:
            report = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Target class 0 (Bad wafers)
                score = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
                report[name] = score
                logging.info(f"Model: {name} | F1 Score (Class 0): {score}")

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(
        self, best_model_object, best_model_name, X_train, y_train, cv_folds
    ):
        try:
            model_config = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )
            model_param_grid = model_config["model_selection"]["model"][
                best_model_name
            ]["search_param_grid"]

            # Custom scorer targeting Class 0
            f1_class0_scorer = make_scorer(
                f1_score, pos_label=0, zero_division=0
            )

            # Stratified split ensures balanced class ratios in every fold
            skf = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=42
            )

            grid_search = GridSearchCV(
                best_model_object,
                param_grid=model_param_grid,
                cv=skf,
                n_jobs=-1,
                verbose=1,
                scoring=f1_class0_scorer,
            )

            grid_search.fit(X_train, y_train)
            print(
                f"Best parameters for {best_model_name}: {grid_search.best_params_}"
            )

            return best_model_object.set_params(**grid_search.best_params_)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(
                "Splitting training and testing input and target feature"
            )

            x_train, y_train = train_array[:, :-1], train_array[
                :, -1
            ].astype(int)
            x_test, y_test = test_array[:, :-1], test_array[:, -1].astype(int)

            # Normalize labels to 0 and 1 if -1 is present
            if -1 in y_train:
                y_train = np.where(y_train == -1, 0, 1)
                y_test = np.where(y_test == -1, 0, 1)

            # Calculate safe CV folds and SMOTE k_neighbors
            minority_count = min(Counter(y_train).values())
            cv_folds = min(5, max(2, minority_count))
            safe_k = min(5, max(1, minority_count - 1))

            # Set safe k_neighbors across all pipelines
            for model in self.models.values():
                model.set_params(smote__k_neighbors=safe_k)

            logging.info(
                f"Training baseline models with Simple SMOTE, cv_folds={cv_folds}, smote_k={safe_k}"
            )
            model_report = self.evaluate_models(
                x_train, y_train, x_test, y_test, self.models
            )

            best_model_name = max(model_report, key=model_report.get)
            best_initial_score = model_report[best_model_name]
            print(
                f"Initial best model: {best_model_name} with F1 (Class 0): {best_initial_score}"
            )

            best_model = self.finetune_best_model(
                best_model_object=self.models[best_model_name],
                best_model_name=best_model_name,
                X_train=x_train,
                y_train=y_train,
                cv_folds=cv_folds,
            )

            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = f1_score(
                y_test, y_pred, pos_label=0, zero_division=0
            )

            print(
                f"Final fine-tuned {best_model_name} F1 Score (Class 0): {best_model_score}"
            )

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No best model found with f1 score greater than the threshold {self.model_trainer_config.expected_accuracy}"
                )

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_path),
                exist_ok=True,
            )

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model,
            )

            return self.model_trainer_config.trained_model_path, best_model_score

        except Exception as e:
            raise CustomException(e, sys)