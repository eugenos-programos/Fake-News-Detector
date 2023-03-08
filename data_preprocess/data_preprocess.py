from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.stem.porter import PorterStemmer
import pickle
from web_scraper import WebScraper
import yaml
import os


class DataPreprocessor():

    tf_idf_vectorizer : TfidfVectorizer = None
    word2vec_tokenizer : Tokenizer = None

    def __init__(self) -> None:
        self.web_scaper = WebScraper()
        self.port_stemmer = PorterStemmer()
        self.conf_file = 'models_config.yml'
        with open(self.conf_file, 'rb') as confile:
            confs = yaml.safe_load(confile)
            self.max_text_size = confs['maxlen_tfidf']
            self.maxlen_word2vec = confs['maxlen_word2vec']

    def stemming(self, text : str):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower().split()
        text = list(map(self.port_stemmer.stem, text))
        text = ' '.join(text)   
        return text
    
    def __vectorize_tf_idf__(self, text : str):
        if self.tf_idf_vectorizer is None:
            with open(self.conf_file, 'rb') as confile:
                tf_ifd_file_name = yaml.safe_load(confile)['tf_idf_vectorizer']
            with open(os.path.join('.', tf_ifd_file_name), 'rb') as vecfile:
                self.tf_idf_vectorizer = pickle.load(vecfile)
        vectorized_text = self.tf_idf_vectorizer.transform([text])
        return vectorized_text
    
    def __vectorize_word2vec__(self, text : str):
        if self.word2vec_tokenizer is None:
            with open(self.conf_file, 'rb') as confile:
                confs = yaml.safe_load(confile)
                word2vec_file_name = confs['word2vec_tokenizer']
            with open(os.path.join('.', word2vec_file_name), 'rb') as vecfile:
                self.word2vec_tokenizer = pickle.load(vecfile)
        tokenized_text = self.word2vec_tokenizer.texts_to_sequences([text.split()])
        seqs = pad_sequences(tokenized_text, maxlen=self.maxlen_word2vec)
        return seqs

    def transform(self, url : str, vectorizer : str):
        text = self.web_scaper.gather_text(url)
        if vectorizer == 'tf-idf':
            pruned_text = ' '.join(text.split()[:self.max_text_size])
            transformed_text = self.stemming(pruned_text)
        else:
            transformed_text = text
        if vectorizer.lower() == 'tf-idf':
            vectorized_text = self.__vectorize_tf_idf__(transformed_text)
        elif vectorizer.lower() == 'word2vec':
            vectorized_text = self.__vectorize_word2vec__(transformed_text)
        return vectorized_text
    