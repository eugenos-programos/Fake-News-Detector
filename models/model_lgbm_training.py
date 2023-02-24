from lightgbm import LGBMClassifier
import mlflow as mlf
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


with mlf.start_run(tags={"version" : "1.0.0", "type" : "boosting"}, run_name='lightgbm-1.0.0-run') as run:
    
    params = {
        'learning_rate' : 0.08,
        'n_estimators' : 400
    }
    
    mlf.log_params(params)

    data_file_name = 'preproc_train.csv'
    vectorizer_file_name = 'vectorizer.pkl'

    data = pd.read_csv(data_file_name).dropna()
    with open(vectorizer_file_name, 'rb') as vecfile:
        vectorizer = pickle.load(vecfile)
    encoded_data = vectorizer.transform(data.title)

    model = LGBMClassifier(**params)
    model.fit(encoded_data, data.label)

    score = model.score(encoded_data, data.label)
    mlf.log_metric("accuracy", score)
    signature = mlf.models.signature.infer_signature(encoded_data, data.label)

    mlf.lightgbm.log_model(model, "LGBMClassifier", signature=signature, input_example=encoded_data, registered_model_name="LGBMClassifier")
