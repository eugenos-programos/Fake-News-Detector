import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# read from yaml
vect_path = 'vectorizer.pkl'
file_path = 'preproc_train.csv'

data = pd.read_csv(file_path).dropna()
vectorizer = TfidfVectorizer()
vectorizer.fit(data.title)

with open(vect_path, 'wb') as vecfile:
    pickle.dump(vectorizer, vecfile)
