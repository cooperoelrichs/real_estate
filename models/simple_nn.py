from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, TFOptimizer
from keras.regularizers import l1, l2
from keras import backend as K
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from real_estate.models.price_model import PriceModel


class EmptyKerasModel(object):
    def __init__(
        self, input_dim, epochs, batch_size,
        learning_rate, verbosity
    ):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbosity = verbosity

    def compile_model(self):
        raise NotImplementedError('This class should not be used directly.')

    def fit(self, X_train, y_train):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)

        self.model = self.compile_model()
        self.model.fit(
            X_scaled, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbosity,
            # validation_split=0.3
        )

    def predict(self, X_pred):
        X_scaled = self.scaler.transform(X_pred)
        return self.model.predict(
            X_scaled, batch_size=self.batch_size,
            verbose=0
        )[:, 0]

    def score(self, X_test, y_test):
        X_scaled = self.scaler.transform(X_test)
        loss_and_metrics = self.model.evaluate(
            X_scaled, y_test, batch_size=self.batch_size,
            verbose=0
        )
        return loss_and_metrics[1]

    def r2(y_true, y_pred):
        '''Keras r2 score.'''
        ss_res = K.sum(K.square(y_true - y_pred))
        ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - ss_res/(ss_tot + K.epsilon())


class LinearNN(EmptyKerasModel):
    def compile_model(self):
        model = Sequential()
        model.add(Dense(
            input_dim=self.input_dim, units=1,
            kernel_initializer='normal',
            # kernel_regularizer=l2(self.LAMBDA_L2)
        ))

        gd = TFOptimizer(tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate
        ))

        model.compile(
            loss='mean_squared_error',
            optimizer=gd,
            metrics=[SimpleNeuralNetworkModel.r2]
        )
        return model


class SimpleNeuralNetworkModel(EmptyKerasModel):
    def __init__(
        self, input_dim, epochs, batch_size,
        learning_rate, lambda_l2, dropout_fraction,
        verbosity
    ):
        super().__init__(
            input_dim, epochs, batch_size, learning_rate, verbosity
        )
        self.lambda_l2 = lambda_l2
        self.dropout_fraction = dropout_fraction

    def compile_model(self):
        model_width = 32
        model = Sequential()
        model.add(Dense(
            input_dim=self.input_dim, units=32,
            kernel_initializer='normal',
            activation='relu',
            kernel_regularizer=l2(self.lambda_l2)
        ))
        model.add(Dense(
            input_dim=self.input_dim, units=model_width,
            kernel_initializer='normal',
            activation='relu',
            kernel_regularizer=l2(self.lambda_l2)
        ))
        model.add(Dense(
            input_dim=self.input_dim, units=model_width,
            kernel_initializer='normal',
            activation='relu',
            kernel_regularizer=l2(self.lambda_l2)
        ))
        model.add(Dense(
            input_dim=self.input_dim, units=model_width,
            kernel_initializer='normal',
            activation='relu',
            kernel_regularizer=l2(self.lambda_l2)
        ))
        model.add(Dense(
            input_dim=self.input_dim, units=model_width,
            kernel_initializer='normal',
            activation='relu',
            kernel_regularizer=l2(self.lambda_l2)
        ))
        model.add(Dense(
            input_dim=self.input_dim, units=16,
            kernel_initializer='normal',
            activation='relu',
            kernel_regularizer=l2(self.lambda_l2)
        ))
        model.add(Dense(
            units=1,
            kernel_initializer='normal'
        ))

        adam = Adam(
            lr=self.learning_rate  # 0.001,
            # beta_1=0.9,
            # beta_2=0.999,
            # epsilon=None,
            # decay=0.0,
            # amsgrad=False
        )
        model.compile(
            loss='mean_squared_error',
            optimizer=adam,
            metrics=[SimpleNeuralNetworkModel.r2]
        )
        return model


class LNN(PriceModel):
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = False
    MODEL_CLASS = LinearNN

    PARAMS = {
        'epochs': 100,
        'batch_size': 10000,
        'learning_rate': 0.01,
        'verbosity': 0,
    }

    def __init__(self, X, y, X_labels, params=None):
        if params is None:
            params = self.PARAMS
            params['input_dim'] = X.shape[1]
        super().__init__(X, y, X_labels, params)


class NN(PriceModel):
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = False
    MODEL_CLASS = SimpleNeuralNetworkModel

    PARAMS = {
        'epochs': 200,
        'batch_size': 1024,
        'learning_rate': 0.0002,
        'verbosity': 2,
        'lambda_l2': 1e7,
        'dropout_fraction': 0,
    }

    def __init__(self, X, y, X_labels, params=None):
        if params is None:
            params = self.PARAMS
            params['input_dim'] = X.shape[1]
        super().__init__(X, y, X_labels, params)
