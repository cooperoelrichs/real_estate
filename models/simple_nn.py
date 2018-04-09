from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from real_estate.models.price_model import PriceModel


class SimpleNeuralNetworkModel(object):
    def __init__(self, input_dim, nb_epoch, batch_size):
        self.input_dim = input_dim
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

        self.model = Sequential()
        self.model.add(Dense(
            input_dim=self.input_dim, output_dim=64,
            init='normal',
            activation='relu'
        ))
        self.model.add(Dense(
            output_dim=10,
            init='normal',
            activation='relu'
        ))
        self.model.add(Dense(
            output_dim=1,
            init='normal'
        ))

        self.model.compile(
            loss='mean_squared_error',
            optimizer='sgd',
            metrics=['mean_squared_error']
        )

        # self.model = KerasRegressor(
        #     build_fn=model,
        #     nb_epoch=nb_epoch, batch_size=batch_size,
        #     verbose=1  # , x=
        # )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, nb_epoch=self.nb_epoch, batch_size=self.batch_size)
        # self.model.fit(X_train, y_train)

    def predict(self, X_pred):
        return self.model.predict(X_pred, batch_size=self.batch_size)
        # self.model.predict(X_pred)

    def score(self, X_test, y_test):
        loss_and_metrics = self.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print(loss_and_metrics)
        return loss_and_metrics
        # self.model.score(X_test, y_test)


class NN(PriceModel):
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = False
    MODEL_CLASS = SimpleNeuralNetworkModel

    PARAMS = {
        'input_dim': 10,
        'nb_epoch': 5,
        'batch_size': 256,
    }

    def __init__(self, X, y, X_labels, params=None):
        super().__init__(X, y, X_labels, params)
