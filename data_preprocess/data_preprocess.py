from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem.porter import PorterStemmer
import pickle
from web_scraper import WebScraper
import yaml
import os


class DataPreprocessor():

    tf_idf_vectorizer : TfidfVectorizer = None

    def __init__(self, max_text_size=500) -> None:
        self.web_scaper = WebScraper()
        self.port_stemmer = PorterStemmer()
        self.max_text_size = max_text_size

    def stemming(self, text : str):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower().split()
        text = list(map(self.port_stemmer.stem, text))
        text = ' '.join(text)   
        return text
    
    def vectorize_tf_idf(self, text):
        if self.tf_idf_vectorizer is None:
            with open('../models_config.yml', 'rb') as confile:
                tf_ifd_file_name = yaml.safe_load(confile)['vectorizer_file']
            with open(os.path.join('..', tf_ifd_file_name), 'rb') as vecfile:
                self.tf_idf_vectorizer = pickle.load(vecfile)
        print(text)
        vectorized_text = self.tf_idf_vectorizer.transform([text])
        return vectorized_text
    
    def transform(self, url : str):
        text = self.web_scaper.gather_text(url)
        pruned_text = ' '.join(text.split()[:self.max_text_size])
        transformed_text = self.stemming(pruned_text)
        vectorized_text = self.vectorize_tf_idf(transformed_text)
        return vectorized_text
