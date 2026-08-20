from dataclasses import dataclass
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class DataTransformationConfig:
    artifact_dir: str = os.path.join(artifact_folder)
    transformed_train_file_path: str = os.path.join(
        artifact_dir, "train.npy"
    )
    transformed_test_file_path: str = os.path.join(artifact_dir, "test.npy")
    transformed_object_file_path: str = os.path.join(
        artifact_dir, "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self, feature_store_file_path: str):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"good/bad": TARGET_COLUMN}, inplace=True)
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            imputer_step = (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0),
            )
            variance_step = (
                "variance_filter",
                VarianceThreshold(threshold=0.0),
            )
            scaler_step = ("scaler", RobustScaler())
            pca_step = ("pca", PCA(n_components=0.95, random_state=42))

            preprocessor = Pipeline(
                steps=[imputer_step, variance_step, scaler_step, pca_step]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        logging.info(
            "Entered initiate_data_transformation method of DataTransformation class"
        )
        try:
            dataframe = self.get_data(
                feature_store_file_path=self.feature_store_file_path
            )

            X = dataframe.drop(columns=TARGET_COLUMN)
            y = dataframe[TARGET_COLUMN].values

            # Standardize binary target: map -1 to 0 (Bad) and 1 to 1 (Good)
            y = np.where(y == -1, 0, y).astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            preprocessor = self.get_data_transformer_object()

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            logging.info(
                f"Features transformed: Reduced from {X_train.shape[1]} to {X_train_scaled.shape[1]} components via PCA"
            )

            preprocessor_path = (
                self.data_transformation_config.transformed_object_file_path
            )
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

            self.utils.save_object(
                file_path=preprocessor_path, obj=preprocessor
            )

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            return (train_arr, test_arr, preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys) from e