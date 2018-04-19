from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, TFOptimizer
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import LearningRateScheduler
from keras.constraints import max_norm
from keras import backend as K
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from real_estate.models.price_model import PriceModel


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
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.verbosity = verbosity

    def compile_model(self):
        raise NotImplementedError('This class should not be used directly.')

    def fit(self, X_train, y_train):
        self.x_scaler, X_scaled = self.new_scaler(X_train)
        self.y_scaler, y_scaled = self.new_scaler(y_train)

        self.model = self.compile_model()
        self.model.fit(
            X_scaled, y_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbosity,
            validation_split=self.validation_split,
            callbacks=self.prep_callbacks(self.callbacks)
        )

    def new_scaler(self, x):
        scaler = StandardScaler()
        scaler.fit(x)
        x_scaled = scaler.transform(x)
        return scaler, x_scaled

    def predict(self, X_pred):
        X_scaled = self.x_scaler.transform(X_pred)
        return self.model.predict(
            X_scaled, batch_size=self.batch_size,
            verbose=0
        )[:, 0]

    def score(self, X_test, y_test):
        X_scaled = self.x_scaler.transform(X_test)
        y_scaled = self.y_scaler.transform(y_test)
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

    def prep_callbacks(self, callback_names):
        callback_objects = {
            'simple_lr_schedule': EmptyKerasModel.simple_lr_scheduler(
                self.learning_rate)
        }
        return [callback_objects[a] for a in callback_names]

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
        validation_split, callbacks, verbosity
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

    def compile_model(self):
        if self.dropout_fractions is not None and (
            len(self.layers) != len(self.dropout_fractions)
        ):
            raise ValueError(
                'Layers and dropout fractions are not consistant.'
            )

        model = Sequential()
        for i, width in enumerate(self.layers):
            if i == 0:
                model.add(Dense(
                    input_dim=self.input_dim, units=width,
                    kernel_initializer='truncated_normal',
                    activation='tanh',
                    kernel_regularizer=l1_l2(self.lambda_l1, self.lambda_l2),
                    kernel_constraint=max_norm(self.max_norm)
                ))
            else:
                model.add(Dense(
                    units=width,
                    kernel_initializer='truncated_normal',
                    activation='tanh',
                    kernel_regularizer=l1_l2(self.lambda_l1, self.lambda_l2),
                    kernel_constraint=max_norm(self.max_norm)
                ))
            if self.dropout_fractions is not None:
                model.add(Dropout(self.dropout_fractions[i]))

        model.add(Dense(
            units=1,
            kernel_initializer='truncated_normal'
        ))

        # optimizer = Adam(
        #     lr=self.learning_rate,
        #     decay=self.learning_rate_decay,
        # )
        optimizer = SGD(
            lr=self.learning_rate,
            momentum=self.momentum,
            decay=self.learning_rate_decay,
            nesterov=False
        )

        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
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
    # and non-categorical parameters
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

    # (2048, 512, 512, 512, 256), Epoch 73, r2: 0.5092, val_r2: 0.3379
    # (128 ,)*10,  # Epoch 116, r2: 0.5109, val_r2: 0.3914

    # This model without dropout:
    #  - Average cv score - simple_nn_model_test: 0.495; and
    #  - Fold 0: 0.566346.

    PARAMS = {
        # 'layers': (2**8,) * 5,  # r2 = -0.06
        'layers': (2**9,) * 5,  # (2**8,) * 5,
        'epochs': 50,
        'batch_size': 2**10,
        'learning_rate': 1e-3,
        'learning_rate_decay': 0.1,
        'momentum': 0.9,
        'verbosity': 2,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'max_norm': 4,
        'validation_split': 0.3,
        'callbacks': [],
        'dropout_fractions': (0,) * 5,
    }

    def __init__(self, X, y, X_labels, params=None):
        if params is None:
            params = self.PARAMS
            params['input_dim'] = X.shape[1]
        else:
            params['input_dim'] = X.shape[1]
        super().__init__(X, y, X_labels, params)
