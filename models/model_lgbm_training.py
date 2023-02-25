from lightgbm import LGBMClassifier
import mlflow as mlf
import pandas as pd
import pickle
import yaml


with mlf.start_run(tags={"version" : "2.0.0", "type" : "boosting"}, run_name='lightgbm-2.0.0-run') as run:
    
    params = {
        'learning_rate' : 0.08,
        'n_estimators' : 400
    }
    
    mlf.log_params(params)

    with open("models_config.yml") as confile:
        confs = yaml.safe_load(confile)

    train_file_name = confs['train_data_file']
    val_file_name = confs['val_data_file']
    vectorizer_file_name = confs['vectorizer_file']

    train_data = pd.read_csv(train_file_name).dropna()
    val_data = pd.read_csv(val_file_name).dropna()
    with open(vectorizer_file_name, 'rb') as vecfile:
        vectorizer = pickle.load(vecfile)
    encoded_train_data = vectorizer.transform(train_data.title)
    encoded_val_data = vectorizer.transform(val_data.title)

    model = LGBMClassifier(**params)
    model.fit(encoded_train_data, train_data.label)

    score = model.score(encoded_val_data, val_data.label)
    mlf.log_metric("val_accuracy", score)
    signature = mlf.models.signature.infer_signature(encoded_val_data, val_data.label)

    mlf.lightgbm.log_model(model, "lgbm_model", signature=signature, input_example=encoded_val_data, registered_model_name="LGBMClassifier")
