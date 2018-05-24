import tensorflow as tf
import tensorflow
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam, SGD, Nadam
from tensorflow.python.keras.regularizers import l1, l2, l1_l2
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.constraints import max_norm
from tensorflow.python.keras.layers import (
    Dense, Dropout, Activation, BatchNormalization, PReLU)

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
        ss_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
        ss_tot = tf.keras.backend.sum(
            tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true))
        )
        return 1 - ss_res/(ss_tot + tf.keras.backend.epsilon())

    def unscale(x, mean, scale):
        return (x * scale) + mean

    def mae(y_true, y_pred):
        '''Keras backend mean absolute error.'''
        return tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_pred))

    def mse(y_true, y_pred):
        '''Keras backend mean squared error.'''
        return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))

    def smooth_l1(y_true, y_pred):
        huber_delta = 0.5
        d = tf.keras.backend.abs(y_true - y_pred)
        l = tf.where(
            d < huber_delta,
            0.5 * d ** 2,
            huber_delta * (d - 0.5 * huber_delta)
        )
        return  tf.keras.backend.sum(l)

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

        sgd = SGD(
            lr=self.learning_rate,
            momentum=self.momentum,
            decay=self.learning_rate_decay,
            nesterov=True
        )

        model.compile(
            loss='mean_squared_error',
            optimizer=sgd,
            metrics=[SimpleNeuralNetworkModel.r2]
        )
        return model


class LNN(PriceModel):
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = False
    MODEL_CLASS = LinearNN

    PARAMS = {
        'epochs': 100,
        'batch_size': 1000,
        'learning_rate': 0.01,
        'verbosity': 0,
    }

    def __init__(self, X, y, X_labels, params=None):
        if params is None:
            params = self.PARAMS
            params['input_dim'] = X.shape[1]
        super().__init__(X, y, X_labels, params)
