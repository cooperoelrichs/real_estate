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
        learning_rate, validation_split, verbosity
    ):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
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
            validation_split=self.validation_split
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
        self, input_dim, layers, epochs, batch_size,
        learning_rate, lambda_l1, lambda_l2, dropout_fraction,
        validation_split, verbosity
    ):
        super().__init__(
            input_dim, epochs, batch_size, learning_rate, validation_split,
            verbosity
        )
        self.layers = layers
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.dropout_fraction = dropout_fraction

        if self.lambda_l2 is not None:
            raise ValueError(
                'L2 regulisation is temporaly disabled, see commented code below.'
            )

    def compile_model(self):
        model = Sequential()
        for i, width in enumerate(self.layers):
            if i == 0:
                model.add(Dense(
                    input_dim=self.input_dim, units=width,
                    kernel_initializer='normal',
                    activation='relu',
                    kernel_regularizer=l1(self.lambda_l1)
                    # kernel_regularizer=l2(self.lambda_l2)
                ))
            else:
                model.add(Dense(
                    units=width,
                    kernel_initializer='normal',
                    activation='relu',
                    kernel_regularizer=l1(self.lambda_l1)
                    # kernel_regularizer=l2(self.lambda_l2)
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

    # Try with model with a significantly smaller learning rate (0.00001?)
    # PARAMS = {
    #     'layers': (2048, 1024, 512, 256, 256) + (256,) * 15,
    #     'epochs': 600,
    #     'batch_size': 1024,
    #     'learning_rate': 0.0001,
    #     'verbosity': 2,
    #     'lambda_l1': 100,
    #     'lambda_l2': None,
    #     'dropout_fraction': 0,
    #     'validation_split': 0.3
    # }

    PARAMS = {
        # 'layers': (2048, 512, 512, 512, 256),  # Epoch 73, r2: 0.5092, val_r2: 0.3379
        # 'layers': (128 ,)*10,  # Epoch 116, r2: 0.5109, val_r2: 0.3914
        'layers': (1024, 1024, 256, 256, 256),
        'epochs': 600,
        'batch_size': 1024,
        'learning_rate': 0.0001,
        'verbosity': 2,
        'lambda_l1': 100,
        'lambda_l2': None,
        'dropout_fraction': 0,
        'validation_split': 0.3
    }

    def __init__(self, X, y, X_labels, params=None):
        if params is None:
            params = self.PARAMS
            params['input_dim'] = X.shape[1]
        else:
            params['input_dim'] = X.shape[1]
        super().__init__(X, y, X_labels, params)
