import os
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam, SGD, Nadam
from tensorflow.python.keras.regularizers import l1, l2, l1_l2
from tensorflow.python.keras.constraints import max_norm
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import (
    Dense, Dropout, Activation, BatchNormalization, PReLU)

from real_estate.models.price_model import PriceModel
from real_estate.models.nn_base import EmptyKerasModel
from real_estate.models.live_keras_plotter import LivePlotter


class SimpleNeuralNetworkModel(EmptyKerasModel):
    def __init__(
        self, input_dim, layers, epochs, batch_size,
        learning_rate, learning_rate_decay, momentum,
        lambda_l1, lambda_l2, dropout_fractions, max_norm,
        validation_split, callbacks, loss, optimizer, verbosity, activation,
        batch_normalization, kernel_initializer, outputs_dir
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
        self.outputs_dir = outputs_dir

        self.callbacks.append(TensorBoard(
            log_dir=os.path.join(self.outputs_dir, 'model')
        ))

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
