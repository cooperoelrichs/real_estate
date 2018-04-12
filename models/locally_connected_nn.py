from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Flatten, LocallyConnected1D
from keras.optimizers import Adam, SGD, TFOptimizer
from keras.regularizers import l1, l2
from keras import backend as K
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from real_estate.models.price_model import PriceModel
from real_estate.models.simple_nn import SimpleNeuralNetworkModel, NN


class LocallyConnectedNeuralNetworkModel(SimpleNeuralNetworkModel):
    DENSE_LAYER = 'dense'
    LOCALLY_CONNECTED_LAYER = 'locally_connected'
    RESHAPE_LAYER = 'reshape'
    FLATTEN_LAYER = 'flatten'
    ACTIVATION = 'relu'
    KERNEL_INITIALIZER = 'normal'

    def compile_model(self):
        model = Sequential()
        for i, layer in enumerate(self.layers):
            layer_type, layer_params = layer
            model = self.add_layer_of_type(model, i, layer_type, layer_params)

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

    def add_layer_of_type(self, model, i, layer_type, layer_params):
        if i == 0:
            if layer_type != self.DENSE_LAYER:
                raise ValueError('First layer must be %s.' % self.DENSE_LAYER)
            model.add(Dense(
                input_dim=self.input_dim,
                units=layer_params[0],
                kernel_initializer=self.KERNEL_INITIALIZER,
                activation=self.ACTIVATION,
                kernel_regularizer=l2(self.lambda_l2)
            ))
        elif layer_type == self.DENSE_LAYER:
            model.add(Dense(
                units=layer_params[0],
                kernel_initializer=self.KERNEL_INITIALIZER,
                activation=self.ACTIVATION,
                kernel_regularizer=l2(self.lambda_l2)
            ))
        elif layer_type == self.LOCALLY_CONNECTED_LAYER:
            filters, kernel_size, strides = layer_params
            model.add(LocallyConnected1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                kernel_initializer=self.KERNEL_INITIALIZER,
                activation=self.ACTIVATION,
                kernel_regularizer=l2(self.lambda_l2)
            ))
        elif layer_type == self.RESHAPE_LAYER:
            target_shape, input_shape = layer_params
            model.add(Reshape(
                target_shape=target_shape,
                input_shape=input_shape
            ))
        elif layer_type == self.FLATTEN_LAYER:
            model.add(Flatten())
        else:
            raise ValueError('Layer type not understood: %s' % layer_type)

        return model


class LCNN(NN):
    MODEL_CLASS = LocallyConnectedNeuralNetworkModel
    PARAMS = {
        'layers': [
            # ('dense', (256,)),
            # ('reshape', ((256, 1), (256,))),
            # ('locally_connected', (3, 10, 1)),
            # ('dense', (256,)),
            # ('flatten', None),
            # ('reshape', ((256, 1), (256,))),
            # ('locally_connected', (3, 10, 1)),
            # ('dense', (256,)),
            # ('flatten', None),

            ('dense', (5,)),
            ('reshape', ((5, 1), (5,))),
            ('locally_connected', (2, 5, 1)),
            ('flatten', None),
            ('dense', (5,)),
            ('flatten', None),
        ],
        'epochs': 2,  # 100,
        'batch_size': 1024,
        'learning_rate': 0.0001,
        'verbosity': 2,
        'lambda_l2': 1e6,
        'dropout_fraction': 0,
        'validation_split': 0.3,
    }
