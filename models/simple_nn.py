from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, PReLU
from keras.optimizers import Adam, SGD, TFOptimizer, Nadam
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import LearningRateScheduler
from keras.constraints import max_norm
from keras import backend as K
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import StandardScaler
from real_estate.models.price_model import PriceModel
from real_estate.models.live_keras_plotter import LivePlotter


class EmptyScaler(object):
    def __init__(self):
        self.mean_ = 0
        self.scale_ = 1

    def fit(self, x):
        pass

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


class EmptyKerasModel(object):
    def __init__(
        self, input_dim, epochs, batch_size,
        learning_rate, learning_rate_decay, momentum, validation_split,
        callbacks, verbosity
    ):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.momentum = momentum
        self.validation_split = validation_split
        self.verbosity = verbosity

        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks

    def compile_model(self):
        raise NotImplementedError('This class should not be used directly.')

    def fit(self, X_train, y_train):
        self.x_scaler, X_scaled = self.new_scaler(X_train)
        # self.y_scaler, y_scaled = self.empty_scaler(y_train)

        self.model = self.compile_model()
        self.model.fit(
            X_scaled, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbosity,
            validation_split=self.validation_split,
            callbacks=self.callbacks
        )

    def new_scaler(self, x):
        scaler = StandardScaler()
        scaler.fit(x)
        x_scaled = scaler.transform(x)
        return scaler, x_scaled

    def empty_scaler(self, x):
        return EmptyScaler(), x

    def predict(self, X_pred):
        X_scaled = self.x_scaler.transform(X_pred)
        y_pred = self.model.predict(
            X_scaled, batch_size=self.batch_size,
            verbose=0
        )[:, 0]
        # y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        return y_pred

    def score(self, X_test, y_test):
        X_scaled = self.x_scaler.transform(X_test)
        # y_scaled = self.y_scaler.transform(y_test)
        loss_and_metrics = self.model.evaluate(
            X_scaled, y_test, batch_size=self.batch_size,
            verbose=0
        )
        return loss_and_metrics[1]

    def r2(y_true, y_pred):
        '''Keras backend r2 score.'''
        ss_res = K.sum(K.square(y_true - y_pred))
        ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - ss_res/(ss_tot + K.epsilon())

    def unscale(x, mean, scale):
        return (x * scale) + mean

    def mae(y_true, y_pred):
        '''Keras backend mean absolute error.'''
        return K.mean(K.abs(y_true - y_pred))

    def mse(y_true, y_pred):
        '''Keras backend mean squared error.'''
        return K.mean(K.square(y_true - y_pred))

    def smooth_l1(y_true, y_pred):
        huber_delta = 0.5
        d = K.abs(y_true - y_pred)
        l = tf.where(
            d < huber_delta,
            0.5 * d ** 2,
            huber_delta * (d - 0.5 * huber_delta)
        )
        return  K.sum(l)

    def scaled_mae(y_scaler):
        mean = y_scaler.mean_
        scale = y_scaler.scale_

        def maes(y_true_scaled, y_pred_scaled):
            y_true = EmptyKerasModel.unscale(y_true_scaled, mean, scale)
            y_pred = EmptyKerasModel.unscale(y_pred_scaled, mean, scale)
            return EmptyKerasModel.mae(y_true, y_pred)
        return maes

    def scaled_mse(y_scaler):
        mean = y_scaler.mean_
        scale = y_scaler.scale_

        def mses(y_true_scaled, y_pred_scaled):
            y_true = EmptyKerasModel.unscale(y_true_scaled, mean, scale)
            y_pred = EmptyKerasModel.unscale(y_pred_scaled, mean, scale)
            return EmptyKerasModel.mse(y_true, y_pred)
        return mses

    def simple_lr_scheduler(learning_rate):
        adjust_at_epoch = 20
        adjust_factor = 1/5
        # adjusted_lr = learning_rate * adjust_factor
        def learning_rate_schedule(current_epoch, lr):
            if current_epoch < adjust_at_epoch:
                return lr
            elif current_epoch == adjust_at_epoch:
                print('Altering the learning rate to: %f' %
                      (lr * adjust_factor))
                return lr * adjust_factor
            else:
                return lr * adjust_factor
        return LearningRateScheduler(schedule=learning_rate_schedule)


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
        learning_rate, learning_rate_decay, momentum,
        lambda_l1, lambda_l2, dropout_fractions, max_norm,
        validation_split, callbacks, loss, optimizer, verbosity, activation,
        batch_normalization, kernel_initializer
    ):
        super().__init__(
            input_dim, epochs, batch_size, learning_rate,
            learning_rate_decay, momentum,
            validation_split, callbacks, verbosity
        )
        self.layers = layers
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.dropout_fractions = dropout_fractions
        self.max_norm = max_norm
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.kernel_initializer = kernel_initializer
        self.loss = loss
        self.optimizer = optimizer

    def compile_model(self):
        if self.dropout_fractions is not None and (
            len(self.layers) != len(self.dropout_fractions)
        ):
            raise ValueError(
                'Layers and dropout fractions are not consistant.'
            )

        if self.max_norm is None:
            kernel_constraint = None
        else:
            kernel_constraint = max_norm(self.max_norm)

        model = Sequential()
        for i, width in enumerate(self.layers):
            if i == 0:
                model.add(Dense(
                    input_dim=self.input_dim, units=width,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=l1_l2(self.lambda_l1, self.lambda_l2),
                    kernel_constraint=kernel_constraint
                ))
            else:
                model.add(Dense(
                    units=width,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=l1_l2(self.lambda_l1, self.lambda_l2),
                    kernel_constraint=kernel_constraint
                ))

            if self.activation == 'prelu':
                model.add(PReLU())
            else:
                model.add(Activation(self.activation))

            if self.batch_normalization is True:
                model.add(BatchNormalization())

            if self.dropout_fractions is not None:
                model.add(Dropout(self.dropout_fractions[i]))

        model.add(Dense(
            units=1,
            kernel_initializer='normal',
            kernel_regularizer=l1_l2(self.lambda_l1, self.lambda_l2)
        ))

        if self.optimizer == 'sgd':
            optimizer = SGD(
                lr=self.learning_rate,
                momentum=self.momentum,
                decay=self.learning_rate_decay,
                nesterov=True
            )
        elif self.optimizer == 'adam':
            optimizer = Adam(
                lr=self.learning_rate,
                decay=self.learning_rate_decay,
            )
        elif self.optimizer == 'nadam':
            optimizer = Nadam()

        if self.loss == 'l1':
            loss = EmptyKerasModel.smooth_l1
        elif self.loss == 'l2':
            loss = 'mean_squared_error'

        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=[
                EmptyKerasModel.r2,
                EmptyKerasModel.mae,
            ]
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
        'layers': (2**9,)*5,
        'epochs': 2000,
        'batch_size': 2**9,
        'learning_rate': 1000,
        'learning_rate_decay': 0.1,
        'momentum': None,
        'verbosity': 2,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'max_norm': None,
        'validation_split': 0.3,
        'callbacks': None,
        'dropout_fractions': (0,) + (0.5,)*4,
        'activation': 'prelu',
        'batch_normalization': True,
        'kernel_initializer': 'lecun_uniform',
        'loss': 'l1',
        'optimizer': 'nadam'
    }

    def __init__(self, X, y, X_labels, params=None):
        if params is None:
            params = self.PARAMS
            params['input_dim'] = X.shape[1]
        else:
            params['input_dim'] = X.shape[1]
        super().__init__(X, y, X_labels, params)

    def model_summary(self):
        self.model.compile_model().summary()

    def show_live_results(self, outputs_folder, name):
        self.model.callbacks = [
            a for a in self.model.callbacks
        ] + [
            LivePlotter((25, 10), self.model.epochs, outputs_folder, name)
        ]
