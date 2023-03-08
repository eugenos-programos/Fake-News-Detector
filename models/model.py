import mlflow 
from data_preprocess import DataPreprocessor
import os


class Model():
    def __init__(self) -> None: 
        self.catboost_model = mlflow.catboost.load_model('./models/CatBoostClassifier')
        self.lightgbm_model = mlflow.lightgbm.load_model('./models/LGBMClassifier')
        self.lstm_model = mlflow.tensorflow.load_model('./models/LSTM')
        self.data_preproces = DataPreprocessor()

    def predict(self, model_type : str, tokenizer : str, url : str):
        if model_type == 'LightGBM' and tokenizer == 'TF-IDF':
            vectorized_text = self.__vectorize_text_tf_idf__(url)
            prediction = self.lightgbm_model.predict(vectorized_text)
            pred_proba = self.lightgbm_model.predict_proba(vectorized_text)
            pred_proba = round(float(pred_proba[0][prediction]), 4)
        elif model_type == 'CatBoost' and tokenizer == 'TF-IDF':
            vectorized_text = self.__vectorize_text_tf_idf__(url)
            prediction = self.catboost_model.predict(vectorized_text)
            pred_proba = self.catboost_model.predict_proba(vectorized_text)
            pred_proba = round(float(pred_proba[0][prediction]), 4)
        elif model_type == 'LSTM' and tokenizer == 'Word2Vec':
            vectorized_text = self.__vectorize_text_word_2_vec__(url)
            pred_proba = self.lstm_model.predict(vectorized_text)
            prediction = (pred_proba > 0.5).astype(int)
        else:
            raise ValueError(f"Unknown model type or tokenizer - {model_type}, {tokenizer}")
        return prediction, pred_proba
    
    def __vectorize_text_word_2_vec__(self, url : str):
        vectorized_text = self.data_preproces.transform(url, 'word2vec')
        return vectorized_text
        
    def __vectorize_text_tf_idf__(self, url : str):
        vectorized_text = self.data_preproces.transform(url, 'tf-idf')
        return vectorized_text
        