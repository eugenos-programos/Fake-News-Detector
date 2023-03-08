from model import Model


class Controller():
    def __init__(self) -> None:
        self.model = Model()
    
    def predict(self, model_type, vectorizer, url) -> None:
        label = self.model.predict(model_type, vectorizer, url)
        return label
    