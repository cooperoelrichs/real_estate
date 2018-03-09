from sklearn.neural_networks import MLPRegressor
from real_estate.models.price_model import PriceModel


class Perceptron(PriceModel):
    MODEL_CLASS = MLPRegressor
    def __init__(self, ):
        pass
