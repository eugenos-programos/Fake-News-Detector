from catboost import CatBoostClassifier
import mlflow as mlf
import pandas as pd
import pickle
import yaml
import os


class CatBoostModel():
    def __init__(self) -> None:
        pass

    def train_and_log(self):
        with mlf.start_run(tags={"version" : "1.0.0", "type" : "boosting"}, run_name='catboost-1.0.0-run') as run:
            
            params = {
                'iterations': 1000,
                'learning_rate': 0.05, 
                'l2_leaf_reg' : 5
            }
            mlf.log_params(params)

            with open("models_config.yml") as confile:
                confs = yaml.safe_load(confile)

            train_file_name = confs['train_data_file']
            val_file_name = confs['val_data_file']
            vectorizer_file_name = confs['vectorizer_file']

            train_data = pd.read_csv(os.path.join('..', train_file_name)).dropna()
            val_data = pd.read_csv(os.path.join('..', val_file_name)).dropna()
            with open(os.path.join('..', vectorizer_file_name, 'rb')) as vecfile:
                vectorizer = pickle.load(vecfile)
            encoded_train_data = vectorizer.transform(train_data.title)
            encoded_val_data = vectorizer.transform(val_data.title)

            model = CatBoostClassifier(**params)
            model.fit(encoded_train_data, train_data.label, eval_set=(encoded_val_data, val_data.label), use_best_model=True)

            score = model.score(encoded_val_data, val_data.label)
            mlf.log_metric("val_accuracy", score)
            signature = mlf.models.signature.infer_signature(encoded_val_data, val_data.label)

            mlf.catboost.log_model(model, "catboost_model", signature=signature, input_example=encoded_val_data, registered_model_name="CatBoostClassifier")

    def save_model(self):
        model = mlf.catboost.load_model('models:/CatBoostClassifier/latest')
        mlf.catboost.save_model(model, './CatBoostClassifier')
        

if __name__ == '__main__':
    model = CatBoostModel()
    model.train_and_log()
    model.save_model()
