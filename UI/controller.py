from model import Model


class Controller():
    def __init__(self) -> None:
        self.model = Model()
    
    def predict(self, model_type, url) -> None:
        label = self.model.predict(model_type, url)
        return label
    