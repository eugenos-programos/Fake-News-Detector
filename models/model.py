import mlflow 
from data_preprocess import DataPreprocessor
import os


class Model():
    def __init__(self) -> None: 
        self.catboost_model = mlflow.catboost.load_model('./models/CatBoostClassifier')
        self.lightgbm_model = mlflow.lightgbm.load_model('./models/LGBMClassifier')
        self.data_preproces = DataPreprocessor()

    def predict(self, model_type : str, url : str):
        if model_type == 'LightGBM':
            vectorized_text = self.__vectorize_text__(url)
            prediction = self.lightgbm_model.predict(vectorized_text)
            pred_proba = self.lightgbm_model.predict_proba(vectorized_text)
            pred_proba = round(float(pred_proba[0][prediction]), 4)
        elif model_type == 'CatBoost':
            vectorized_text = self.__vectorize_text__(url)
            prediction = self.catboost_model.predict(vectorized_text)
            pred_proba = self.catboost_model.predict_proba(vectorized_text)
            pred_proba = round(float(pred_proba[0][prediction]), 4)
        else:
            raise ValueError(f"Unknown model type - {model_type}")
        return prediction, pred_proba
        
    def __vectorize_text__(self, url : str):
        vectorized_text = self.data_preproces.transform(url)
        return vectorized_text
        